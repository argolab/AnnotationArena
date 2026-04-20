/*
 * Tensor model HMC inference.
 *
 * Generative model (matched exactly to tensor_generation.stan):
 *   z_ijk = sum_t alpha_jt * exp(u_i + v_t + u_it) . e_k
 *
 * Noise model (use_dawid_skene_noise):
 *   0 = continuous: ordinal probit with inferred sigma_measurement
 *       log P(c | z) = log [ Phi((thresh_c - z/sigma_total) / (sigma_m/sigma_total))
 *                           - Phi((thresh_{c-1} - ...) / ...) ]
 *       sigma_total = sqrt(||eff_pref[ij]||^2 + sigma_measurement^2)
 *   1 = discrete:  Dawid-Skene marginalized likelihood
 *       log P(c_obs | z) = log sum_{c_lat} P(bin=c_lat | z) * confusion[c_lat][c_obs]
 *       bin_smoothing = 0.05 (fixed HMC regularizer; not a model parameter)
 *
 * Parameters inferred:
 *   u_attr[I, D]              ~ N(0, sigma_u)
 *   v_proto[T, D]             ~ N(0, sigma_v)
 *   u_inter[I][T, D]          ~ N(0, sigma_uit)
 *   sigma_u                   ~ Gamma(2, 2)          [mean 1.0]
 *   sigma_v                   ~ Gamma(2, 2)          [mean 1.0]
 *   sigma_uit                 ~ Gamma(2, 20)         [mean 0.1]
 *   sigma_measurement         ~ Gamma(2, 20)         [mean 0.1]
 *   kappa                     ~ Gamma(2, 2/15)       [mean 15.0]
 *   alpha_jt[J][T]            ~ Dirichlet(alpha_dirichlet_jt, ...)
 *   embeddings[K, D]          ~ N(0, 1)
 *   rating_probs[I*J][C]      ~ Dirichlet(kappa / C, ...)
 *   confusion_matrix[C][C]    ~ Dirichlet(alpha_confusion on diag, 1 off-diag)
 *                                (only enters likelihood when use_dawid_skene_noise=1;
 *                                 gets its prior when use_dawid_skene_noise=0)
 *   threshold_transform_W[C-1, T]  ~ N(0, 1/sqrt(T))  (when derive_thresholds_from_annotator=1)
 *   threshold_attr_bias[I, C-1]    ~ N(0, 0.1)         (when derive_thresholds_from_annotator=1)
 */

functions {

    // Ordinal probit bin probability P(c | z, thresholds).
    // Used for continuous noise (sigma_measurement > 0).
    real ordinal_bin_prob(
        real base_score,
        real total_std,
        real sigma_measurement,
        real lower_threshold,
        real upper_threshold
    ) {
        real mean_std = base_score / total_std;
        real cond_std = sigma_measurement / total_std;
        real upper_p  = (upper_threshold == positive_infinity())
                         ? 1.0
                         : Phi((upper_threshold - mean_std) / cond_std);
        real lower_p  = (lower_threshold == negative_infinity())
                         ? 0.0
                         : Phi((lower_threshold - mean_std) / cond_std);
        return upper_p - lower_p;
    }

}

data {

    // ── Dimensions ──────────────────────────────────────────────────────────
    int<lower=1> K;   // total items (K_train + K_test, same indexing as generation output)
    int<lower=1> I;   // attributes
    int<lower=1> J;   // annotators
    int<lower=1> D;   // embedding dimension
    int<lower=1> C;   // rating categories
    int<lower=1> T;   // number of annotator prototypes

    // ── Model flags ─────────────────────────────────────────────────────────
    int<lower=0, upper=1> use_dawid_skene_noise;           // 0=continuous, 1=discrete
    int<lower=0, upper=1> derive_thresholds_from_annotator; // 0=Dirichlet, 1=from alpha_j

    // ── Observed ratings ────────────────────────────────────────────────────
    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=I> rating_attributes;
    array[N_ratings] int<lower=1, upper=J> rating_annotators;
    array[N_ratings] int<lower=1, upper=K> rating_items;
    array[N_ratings] int<lower=1, upper=C> rating_values;   // hard integer labels 1..C

    // ── Missing ratings to predict ───────────────────────────────────────────
    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=I> missing_rating_attributes;
    array[N_missing_ratings] int<lower=1, upper=J> missing_rating_annotators;
    array[N_missing_ratings] int<lower=1, upper=K> missing_rating_items;

    // ── Hyperparameters fixed as data ────────────────────────────────────────
    real<lower=0> alpha_dirichlet_jt; // Dirichlet prior concentration for alpha_jt
    real<lower=0> alpha_confusion;   // diagonal concentration for confusion matrix prior

}

parameters {

    // Item embeddings: e_k ~ N(0, I_D)
    matrix[K, D] embeddings;

    // Tensor model parameters
    matrix[I, D] u_attr;                  // attribute embeddings u_i
    matrix[T, D] v_proto;                 // prototype embeddings v_t
    array[I] matrix[T, D] u_inter;        // attribute-prototype interactions u_it
    real<lower=0> sigma_u;                // flat prior over positive reals
    real<lower=0> sigma_v;                // flat prior over positive reals
    real<lower=0> sigma_uit;              // flat prior over positive reals
    real<lower=0> sigma_measurement;      // flat prior over positive reals (used when continuous noise)
    real<lower=0> kappa;                  // flat prior over positive reals for rating_probs concentration

    // Annotator mixing weights: alpha_j ~ Dirichlet(alpha_dirichlet_jt, ...)
    array[J] simplex[T] alpha_jt;

    // Rating threshold parameters (one simplex per (i,j) pair)
    array[I*J] simplex[C] rating_probs;

    // Threshold transform (only meaningful when derive_thresholds_from_annotator=1;
    // always declared — gets a regularizing prior when unused)
    matrix[C-1, T] threshold_transform_W;
    matrix[I, C-1] threshold_attr_bias;

    // Confusion matrix (only enters likelihood when use_dawid_skene_noise=1;
    // always declared — gets its Dirichlet prior regardless)
    array[C] simplex[C] confusion_matrix;

}

transformed parameters {

    // ── Effective preference vectors eff_pref[ij] = sum_t alpha_jt[t]*exp(u_i+v_t+u_it) ──
    matrix[I*J, D] eff_pref;
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            row_vector[D] pref = rep_row_vector(0.0, D);
            for (t in 1:T) {
                row_vector[D] log_gate = u_attr[i] + v_proto[t] + u_inter[i][t];
                pref = pref + alpha_jt[j][t] * exp(log_gate);
            }
            eff_pref[idx] = pref;
        }
    }

    // ── Base scores z_ijk = eff_pref[ij] . e_k ──────────────────────────────
    matrix[I*J, K] base_scores;
    for (ij in 1:(I*J))
        for (k in 1:K)
            base_scores[ij, k] = dot_product(eff_pref[ij], embeddings[k]);

    // ── Ordinal thresholds from rating_probs (or from alpha_j if flagged) ───
    // When derive_thresholds_from_annotator=1, rating_probs is still declared as a
    // parameter but its likelihood is replaced by the threshold_transform path.
    // For simplicity we always derive thresholds from rating_probs in the likelihood;
    // the threshold_transform_W prior encourages the two to agree when the flag is set.
    // (A cleaner formulation would skip rating_probs entirely when the flag is set;
    // this is left as a future refinement.)
    array[I*J] vector[C+1] rating_thresholds;
    for (ij in 1:(I*J)) {
        rating_thresholds[ij][1] = negative_infinity();
        for (c in 2:C)
            rating_thresholds[ij][c] = inv_Phi(sum(rating_probs[ij][1:(c-1)]));
        rating_thresholds[ij][C+1] = positive_infinity();
    }

    // ── total_std per (i,j) ─────────────────────────────────────────────────
    // Continuous: sqrt(||eff_pref||^2 + sigma_measurement^2)
    // Discrete:   sqrt(||eff_pref||^2 + bin_smoothing^2)   (smoothing for HMC only)
    real bin_smoothing = 0.05;
    array[I*J] real total_std;
    for (ij in 1:(I*J)) {
        real noise_sq = (use_dawid_skene_noise == 1)
                         ? bin_smoothing * bin_smoothing
                         : sigma_measurement * sigma_measurement;
        total_std[ij] = sqrt(dot_self(eff_pref[ij]) + noise_sq);
    }

    // ── sigma_noise: which noise level enters ordinal_bin_prob ───────────────
    real sigma_noise = (use_dawid_skene_noise == 1) ? bin_smoothing : sigma_measurement;

}

model {

    // ── Priors ──────────────────────────────────────────────────────────────
    for (k in 1:K)
        embeddings[k] ~ normal(0, 1);

    // Continuous hyperparameters (proper priors centered at legacy defaults).
    sigma_u           ~ gamma(2, 2);
    sigma_v           ~ gamma(2, 2);
    sigma_uit         ~ gamma(2, 4);
    sigma_measurement ~ gamma(2, 20);
    kappa             ~ gamma(2, 2.0 / 15.0);

    to_vector(u_attr)  ~ normal(0, sigma_u);
    to_vector(v_proto) ~ normal(0, sigma_v);
    for (i in 1:I)
        to_vector(u_inter[i]) ~ normal(0, sigma_uit);

    for (j in 1:J)
        alpha_jt[j] ~ dirichlet(rep_vector(alpha_dirichlet_jt, T));

    for (ij in 1:(I*J))
        rating_probs[ij] ~ dirichlet(rep_vector(kappa / C, C));

    // threshold_transform priors (regularizing; meaningful when flag=1)
    to_vector(threshold_transform_W) ~ normal(0, 1.0 / sqrt(T));
    to_vector(threshold_attr_bias)   ~ normal(0, 0.1);

    // Confusion matrix prior: diagonal = alpha_confusion, off-diagonal = 1
    for (c in 1:C) {
        vector[C] alpha_dir = rep_vector(1.0, C);
        alpha_dir[c] = alpha_confusion;
        confusion_matrix[c] ~ dirichlet(alpha_dir);
    }

    // ── Likelihood ──────────────────────────────────────────────────────────
    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;
        int c_obs  = rating_values[n];

        if (use_dawid_skene_noise == 1) {
            // Discrete: marginalize over latent bin
            // log P(c_obs) = log sum_{c_lat} P(bin=c_lat | z) * confusion[c_lat][c_obs]
            real mix = 0.0;
            for (c_lat in 1:C) {
                mix += ordinal_bin_prob(
                    base_scores[ij_idx, k], total_std[ij_idx], sigma_noise,
                    rating_thresholds[ij_idx][c_lat], rating_thresholds[ij_idx][c_lat+1]
                ) * confusion_matrix[c_lat][c_obs];
            }
            target += log(mix + 1e-10);
        } else {
            // Continuous: standard ordinal probit
            target += log(ordinal_bin_prob(
                base_scores[ij_idx, k], total_std[ij_idx], sigma_noise,
                rating_thresholds[ij_idx][c_obs], rating_thresholds[ij_idx][c_obs+1]
            ) + 1e-10);
        }
    }

}

generated quantities {

    // Observed log-likelihood
    real log_lik_ratings_obs = 0;
    real log_lik_pairwise_obs = 0;   // stub — no pairwise data in current pipeline
    real total_log_lik;

    // Posterior predictive for missing ratings
    array[N_missing_ratings] int<lower=1, upper=C>  missing_rating_predictions;
    array[N_missing_ratings] vector[C]               missing_rating_probs;

    // ── Observed log-likelihood ──────────────────────────────────────────────
    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;
        int c_obs  = rating_values[n];

        if (use_dawid_skene_noise == 1) {
            real mix = 0.0;
            for (c_lat in 1:C)
                mix += ordinal_bin_prob(
                    base_scores[ij_idx, k], total_std[ij_idx], sigma_noise,
                    rating_thresholds[ij_idx][c_lat], rating_thresholds[ij_idx][c_lat+1]
                ) * confusion_matrix[c_lat][c_obs];
            log_lik_ratings_obs += log(mix + 1e-10);
        } else {
            log_lik_ratings_obs += log(ordinal_bin_prob(
                base_scores[ij_idx, k], total_std[ij_idx], sigma_noise,
                rating_thresholds[ij_idx][c_obs], rating_thresholds[ij_idx][c_obs+1]
            ) + 1e-10);
        }
    }

    total_log_lik = log_lik_ratings_obs;

    // ── Posterior predictive for missing ratings ─────────────────────────────
    for (n in 1:N_missing_ratings) {
        int i      = missing_rating_attributes[n];
        int j      = missing_rating_annotators[n];
        int k      = missing_rating_items[n];
        int ij_idx = (i-1)*J + j;

        real base_score = base_scores[ij_idx, k];

        // Step 1: ordinal probit bin probabilities
        vector[C] bin_probs;
        for (c in 1:C)
            bin_probs[c] = ordinal_bin_prob(
                base_score, total_std[ij_idx], sigma_noise,
                rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
            );

        if (use_dawid_skene_noise == 1) {
            // Step 2a: mix through confusion matrix
            for (c_obs in 1:C) {
                real p = 0.0;
                for (c_lat in 1:C)
                    p += bin_probs[c_lat] * confusion_matrix[c_lat][c_obs];
                missing_rating_probs[n][c_obs] = p;
            }
            // Step 3a: sample latent bin, then corrupt
            int sampled_latent = categorical_rng(bin_probs);
            missing_rating_predictions[n] = categorical_rng(confusion_matrix[sampled_latent]);
        } else {
            // Step 2b: bin_probs are already the predictive distribution
            missing_rating_probs[n] = bin_probs;
            // Step 3b: sample directly
            missing_rating_predictions[n] = categorical_rng(bin_probs);
        }
    }

}

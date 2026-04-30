/*
 * Tensor model HMC inference with attribute-level shared thresholds on raw z.
 *
 * Misspecification:
 *   For each attribute i, all annotators share one threshold vector tau_i.
 *   A Dirichlet-induced prior is applied to per-attribute bin masses implied by
 *   averaging ordinal-bin probabilities across annotators j (Jacobian included).
 */

functions {

    // Ordinal probit bin probability on raw z-score scale.
    real ordinal_bin_prob_raw_z(
        real base_score,
        real sigma_link,
        real lower_threshold,
        real upper_threshold
    ) {
        real upper_p  = (upper_threshold == positive_infinity())
                         ? 1.0
                         : Phi((upper_threshold - base_score) / sigma_link);
        real lower_p  = (lower_threshold == negative_infinity())
                         ? 0.0
                         : Phi((lower_threshold - base_score) / sigma_link);
        return upper_p - lower_p;
    }

    real annotator_mixture_cdf_value(real x, vector scales) {
        real acc = 0.0;
        int J_local = num_elements(scales);
        for (j in 1:J_local)
            acc += Phi(x / scales[j]);
        return acc / J_local;
    }

    real annotator_mixture_log_density(real x, vector scales) {
        int J_local = num_elements(scales);
        vector[J_local] log_terms;
        for (j in 1:J_local)
            log_terms[j] = normal_lpdf(x | 0, scales[j]);
        return log_sum_exp(log_terms) - log(J_local);
    }
}

data {
    int<lower=1> K;
    int<lower=1> I;
    int<lower=1> J;
    int<lower=1> D;
    int<lower=1> C;
    int<lower=1> T;

    int<lower=0, upper=1> use_dawid_skene_noise;
    int<lower=0, upper=1> derive_thresholds_from_annotator;

    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=I> rating_attributes;
    array[N_ratings] int<lower=1, upper=J> rating_annotators;
    array[N_ratings] int<lower=1, upper=K> rating_items;
    array[N_ratings] int<lower=1, upper=C> rating_values;

    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=I> missing_rating_attributes;
    array[N_missing_ratings] int<lower=1, upper=J> missing_rating_annotators;
    array[N_missing_ratings] int<lower=1, upper=K> missing_rating_items;

    real<lower=0> alpha_dirichlet_jt;
    real<lower=0> alpha_confusion;
}

parameters {
    matrix[K, D] embeddings;

    matrix[I, D] u_attr;
    matrix[T, D] v_proto;
    array[I] matrix[T, D] u_inter;
    real<lower=0> sigma_u;
    real<lower=0> sigma_v;
    real<lower=0> sigma_uit;
    real<lower=0> sigma_measurement;
    real<lower=0> kappa;

    array[J] simplex[T] alpha_jt;

    // Attribute-level thresholds shared across annotators.
    array[I] ordered[C-1] attribute_thresholds;

    array[C] simplex[C] confusion_matrix;
}

transformed parameters {
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

    matrix[I*J, K] base_scores;
    for (ij in 1:(I*J))
        for (k in 1:K)
            base_scores[ij, k] = dot_product(eff_pref[ij], embeddings[k]);

    array[I] vector[C+1] rating_thresholds;
    for (i in 1:I) {
        rating_thresholds[i][1] = negative_infinity();
        for (c in 2:C)
            rating_thresholds[i][c] = attribute_thresholds[i][c-1];
        rating_thresholds[i][C+1] = positive_infinity();
    }

    real bin_smoothing = 0.05;
    array[I*J] real total_std;
    for (ij in 1:(I*J)) {
        real noise_sq = (use_dawid_skene_noise == 1)
                         ? bin_smoothing * bin_smoothing
                         : sigma_measurement * sigma_measurement;
        total_std[ij] = sqrt(dot_self(eff_pref[ij]) + noise_sq);
    }

    real sigma_noise = (use_dawid_skene_noise == 1) ? bin_smoothing : sigma_measurement;
}

model {
    for (k in 1:K)
        embeddings[k] ~ normal(0, 1);

    sigma_u           ~ gamma(2, 2.5);
    sigma_v           ~ gamma(2, 0.25);
    sigma_uit         ~ gamma(2, 2.5);
    sigma_measurement ~ gamma(2, 20);
    kappa             ~ gamma(2, 2.0 / 15.0);

    to_vector(u_attr)  ~ normal(0, sigma_u);
    to_vector(v_proto) ~ normal(0, sigma_v);
    for (i in 1:I)
        to_vector(u_inter[i]) ~ normal(0, sigma_uit);

    for (j in 1:J)
        alpha_jt[j] ~ dirichlet(rep_vector(alpha_dirichlet_jt, T));

    // Dirichlet-induced prior over average-j masses at fixed attribute i.
    for (i in 1:I) {
        vector[J] annotator_scales;
        vector[C] induced_probs;
        real prev_q = 0.0;
        for (j in 1:J)
            annotator_scales[j] = total_std[(i-1)*J + j];
        for (c in 1:(C-1)) {
            real q = annotator_mixture_cdf_value(attribute_thresholds[i][c], annotator_scales);
            induced_probs[c] = q - prev_q;
            prev_q = q;
            target += annotator_mixture_log_density(attribute_thresholds[i][c], annotator_scales);
        }
        induced_probs[C] = 1.0 - prev_q;
        target += dirichlet_lpdf(induced_probs | rep_vector(kappa / C, C));
    }

    for (c in 1:C) {
        vector[C] alpha_dir = rep_vector(1.0, C);
        alpha_dir[c] = alpha_confusion;
        confusion_matrix[c] ~ dirichlet(alpha_dir);
    }

    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;
        int c_obs  = rating_values[n];

        if (use_dawid_skene_noise == 1) {
            real mix = 0.0;
            for (c_lat in 1:C) {
                mix += ordinal_bin_prob_raw_z(
                    base_scores[ij_idx, k], sigma_noise,
                    rating_thresholds[i][c_lat], rating_thresholds[i][c_lat+1]
                ) * confusion_matrix[c_lat][c_obs];
            }
            target += log(mix + 1e-10);
        } else {
            target += log(ordinal_bin_prob_raw_z(
                base_scores[ij_idx, k], sigma_noise,
                rating_thresholds[i][c_obs], rating_thresholds[i][c_obs+1]
            ) + 1e-10);
        }
    }
}

generated quantities {
    real log_lik_ratings_obs = 0;
    real log_lik_pairwise_obs = 0;
    real total_log_lik;

    array[N_missing_ratings] int<lower=1, upper=C>  missing_rating_predictions;
    array[N_missing_ratings] vector[C]               missing_rating_probs;

    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;
        int c_obs  = rating_values[n];

        if (use_dawid_skene_noise == 1) {
            real mix = 0.0;
            for (c_lat in 1:C)
                mix += ordinal_bin_prob_raw_z(
                    base_scores[ij_idx, k], sigma_noise,
                    rating_thresholds[i][c_lat], rating_thresholds[i][c_lat+1]
                ) * confusion_matrix[c_lat][c_obs];
            log_lik_ratings_obs += log(mix + 1e-10);
        } else {
            log_lik_ratings_obs += log(ordinal_bin_prob_raw_z(
                base_scores[ij_idx, k], sigma_noise,
                rating_thresholds[i][c_obs], rating_thresholds[i][c_obs+1]
            ) + 1e-10);
        }
    }

    total_log_lik = log_lik_ratings_obs;

    for (n in 1:N_missing_ratings) {
        int i      = missing_rating_attributes[n];
        int j      = missing_rating_annotators[n];
        int k      = missing_rating_items[n];
        int ij_idx = (i-1)*J + j;

        real base_score = base_scores[ij_idx, k];
        vector[C] bin_probs;
        for (c in 1:C)
            bin_probs[c] = ordinal_bin_prob_raw_z(
                base_score, sigma_noise,
                rating_thresholds[i][c], rating_thresholds[i][c+1]
            );

        if (use_dawid_skene_noise == 1) {
            for (c_obs in 1:C) {
                real p = 0.0;
                for (c_lat in 1:C)
                    p += bin_probs[c_lat] * confusion_matrix[c_lat][c_obs];
                missing_rating_probs[n][c_obs] = p;
            }
            {
                int sampled_latent = categorical_rng(bin_probs);
                missing_rating_predictions[n] = categorical_rng(confusion_matrix[sampled_latent]);
            }
        } else {
            missing_rating_probs[n] = bin_probs;
            missing_rating_predictions[n] = categorical_rng(bin_probs);
        }
    }
}

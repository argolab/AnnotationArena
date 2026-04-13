/*
 * Dawid-Skene inference model: factored latent space + shared confusion matrix.
 *
 * Generative process (pure discrete noise):
 *   1. Latent score z_ijk = V_ij · e_k
 *      → binned into C categories via ordinal probit (latent_bin)
 *   2. Observed rating c_obs drawn from confusion row:
 *      c_obs ~ Categorical(confusion_matrix[latent_bin])
 *
 * No continuous measurement noise — bin_smoothing = 0.05 is a fixed computational
 * constant that keeps ordinal-probit probabilities differentiable for HMC; it is
 * not a model parameter.
 *
 * Inference marginalizes out latent_bin:
 *   log P(c_obs | model) = log( sum_{c_lat=1}^{C}
 *       P(latent_bin=c_lat | z_ijk) * confusion_matrix[c_lat][c_obs] )
 *
 * LLM ratings are unchanged: q_n ~ Dir(alpha_llm * pi_ijk)
 * where pi_ijk are the pure ordinal-probit bin probabilities (no confusion).
 *
 * The shared confusion matrix has a Dirichlet prior per row:
 *   row c ~ Dirichlet(alpha_dir)  where alpha_dir[c] = alpha_confusion, alpha_dir[c'] = 1
 * Higher alpha_confusion → confusion matrix closer to identity (less confusion noise).
 */

functions {

    // Raw bin probability P(c | base_score, thresholds) via ordinal probit.
    real ordinal_bin_prob(
        real base_score,
        real total_std_ij,
        real sigma_measurement,
        real lower_threshold,
        real upper_threshold
    ) {
        real mean_std = base_score / total_std_ij;
        real cond_std = sigma_measurement / total_std_ij;

        real upper_prob = (upper_threshold == positive_infinity())
                          ? 1.0
                          : Phi((upper_threshold - mean_std) / cond_std);
        real lower_prob = (lower_threshold == negative_infinity())
                          ? 0.0
                          : Phi((lower_threshold - mean_std) / cond_std);

        return upper_prob - lower_prob;
    }

    // Log P(c | base_score, thresholds) — for hard-label human ratings.
    real ordinal_log_prob(
        real base_score,
        real total_std_ij,
        real sigma_measurement,
        real lower_threshold,
        real upper_threshold
    ) {
        return log(ordinal_bin_prob(base_score, total_std_ij, sigma_measurement,
                                    lower_threshold, upper_threshold) + 1e-10);
    }

}

data {

    // ── Dimensions ──────────────────────────────────────────────────────────
    int<lower=1> K;            // total items  (K_train + K_test)
    int<lower=1> I;            // attributes / criteria
    int<lower=1> J;            // annotators  (humans + LLM)
    int<lower=1> D;            // embedding dimension
    int<lower=1> C;            // rating categories
    int<lower=1> d_annotator;  // annotator embedding dimension (used when use_factored_annotator=1)

    // ── Annotator model selection ────────────────────────────────────────────
    // 0 = old spherical model: V_ij ~ N(v_i, sigma^2) independently
    // 1 = new factored model:  V_ij = v_i + u_j * M_i
    int<lower=0, upper=1> use_factored_annotator;

    // ── Observed ratings ────────────────────────────────────────────────────
    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=I> rating_attributes;
    array[N_ratings] int<lower=1, upper=J> rating_annotators;
    array[N_ratings] int<lower=1, upper=K> rating_items;

    // Hard integer labels (1..C): used for human ratings (is_llm_rating==0).
    // For LLM ratings set to the argmax — not used in the likelihood.
    array[N_ratings] int<lower=1, upper=C> rating_values;

    // Full distributions over C categories: used for LLM ratings (is_llm_rating==1).
    // For human ratings set to the one-hot — not used in the likelihood.
    array[N_ratings] simplex[C] rating_dists;

    // Routing indicator: 0 = human (Dawid-Skene marginal), 1 = LLM (Dirichlet).
    array[N_ratings] int<lower=0, upper=1> is_llm_rating;

    // ── Missing ratings to predict (test-set human annotations) ─────────────
    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=I> missing_rating_attributes;
    array[N_missing_ratings] int<lower=1, upper=J> missing_rating_annotators;
    array[N_missing_ratings] int<lower=1, upper=K> missing_rating_items;

    // ── Hyperparameters ─────────────────────────────────────────────────────
    real<lower=0> sigma_annotator;    // std of annotator preference around mean
    real<lower=0> alpha_dirichlet;    // Dirichlet prior concentration for rating_probs
    real<lower=0> temperature;        // unused here; kept for interface consistency
    real<lower=0> alpha_llm;          // Dirichlet likelihood concentration for LLM obs

    // ── Dawid-Skene confusion prior ──────────────────────────────────────────
    // Row c ~ Dirichlet(alpha_dir) where alpha_dir[c] = alpha_confusion, alpha_dir[c']=1
    // Higher alpha_confusion → confusion matrix closer to identity.
    real<lower=0> alpha_confusion;

}

parameters {

    // Item embeddings: e_k ~ N(0, I_D)
    matrix[K, D] embeddings;

    // Mean preferences per attribute: v_i ~ N(0, I_D)
    matrix[I, D] mean_preferences;

    // ── Annotator preference parameterization ────────────────────────────────
    // OLD model (use_factored_annotator=0):
    //   annotator_prefs_direct[idx] ~ N(mean_preferences[i], sigma_annotator^2)
    //   directly used as V_ij.
    // NEW model (use_factored_annotator=1):
    //   annotator_embeddings[j] in R^{d_annotator}: per-annotator style vector
    //   attr_transforms[i] in R^{d_annotator x D}: attribute-specific projection
    //   V_ij = v_i + u_j * M_i  (assembled in transformed parameters)
    //
    // Both sets are always declared; the prior/likelihood below activates only
    // the relevant block via if/else on use_factored_annotator.

    // OLD model parameters
    matrix[I*J, D] annotator_prefs_direct;   // V_ij directly (spherical model)

    // NEW model parameters
    matrix[J, d_annotator] annotator_embeddings;      // u_j
    array[I] matrix[d_annotator, D] attr_transforms;  // M_i

    // Rating probability simplex per (i,j): p_ij ~ Dir(alpha_dirichlet/C, ...)
    array[I*J] simplex[C] rating_probs;

    // ── Dawid-Skene confusion matrix ─────────────────────────────────────────
    // Shared C×C matrix; row c_lat is the distribution over observed categories
    // given latent bin c_lat.
    array[C] simplex[C] confusion_matrix;

}

transformed parameters {

    // Resolved annotator preferences V_ij (I*J x D).
    // For old model: copy of annotator_prefs_direct.
    // For new model: v_i + u_j * M_i.
    matrix[I*J, D] annotator_preferences;

    // Scale for M_i rows to match variance of old model:
    //   Var(u_j * M_i) = sigma_annotator^2 * d_annotator  if rows of M_i ~ N(0, sigma_M^2)
    //   => sigma_M = sigma_annotator / sqrt(d_annotator)
    real sigma_M = sigma_annotator / sqrt(d_annotator);

    if (use_factored_annotator == 1) {
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                // u_j is [1, d_annotator], M_i is [d_annotator, D]
                // annotator_embeddings[j] is a row vector here via row indexing
                annotator_preferences[idx] = mean_preferences[i]
                    + annotator_embeddings[j] * attr_transforms[i];
            }
        }
    } else {
        annotator_preferences = annotator_prefs_direct;
    }

    // Base utility scores: z_ijk = v_ij · e_k
    matrix[I*J, K] base_scores;
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], embeddings[k]);
            }
        }
    }

    // Ordinal thresholds derived from rating_probs via inverse-normal CDF.
    // threshold[ij][1] = -inf,  threshold[ij][C+1] = +inf.
    array[I*J] vector[C+1] rating_thresholds;
    for (ij in 1:(I*J)) {
        rating_thresholds[ij][1] = negative_infinity();
        for (c in 2:C) {
            rating_thresholds[ij][c] = inv_Phi(sum(rating_probs[ij][1:(c-1)]));
        }
        rating_thresholds[ij][C+1] = positive_infinity();
    }

    // Fixed smoothing constant — keeps ordinal-probit gradients non-degenerate for HMC.
    // Not a model parameter; generation has no measurement noise.
    real bin_smoothing = 0.05;

    // total_std[ij] = sqrt(||v_ij||^2 + bin_smoothing^2)
    array[I*J] real total_std;
    for (i in 1:I) {
        for (j in 1:J) {
            int ij_idx = (i-1)*J + j;
            total_std[ij_idx] = sqrt(dot_self(annotator_preferences[ij_idx])
                                     + bin_smoothing * bin_smoothing);
        }
    }

}

model {

    // ── Priors ──────────────────────────────────────────────────────────────
    for (k in 1:K) embeddings[k] ~ normal(0, 1);
    for (i in 1:I) mean_preferences[i] ~ normal(0, 1);

    if (use_factored_annotator == 1) {
        // NEW factored model priors
        // u_j ~ N(0, I_{d_annotator})
        for (j in 1:J) {
            annotator_embeddings[j] ~ normal(0, 1);
        }
        // M_i rows ~ N(0, sigma_M^2)
        for (i in 1:I) {
            for (r in 1:d_annotator) {
                attr_transforms[i][r] ~ normal(0, sigma_M);
            }
        }
        // annotator_prefs_direct unused — give it a weak prior to avoid improper posterior
        for (idx in 1:(I*J)) {
            annotator_prefs_direct[idx] ~ normal(0, 1);
        }
    } else {
        // OLD spherical model prior: V_ij ~ N(v_i, sigma_annotator^2)
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                annotator_prefs_direct[idx] ~ normal(mean_preferences[i], sigma_annotator);
            }
        }
        // annotator_embeddings and attr_transforms unused — weak priors
        for (j in 1:J) {
            annotator_embeddings[j] ~ normal(0, 1);
        }
        for (i in 1:I) {
            for (r in 1:d_annotator) {
                attr_transforms[i][r] ~ normal(0, 1);
            }
        }
    }

    for (ij in 1:(I*J)) {
        rating_probs[ij] ~ dirichlet(rep_vector(alpha_dirichlet / C, C));
    }

    // ── Dawid-Skene confusion matrix prior ───────────────────────────────────
    // Row c ~ Dirichlet(alpha_dir) where alpha_dir[c] = alpha_confusion, alpha_dir[c']=1
    for (c in 1:C) {
        vector[C] alpha_dir = rep_vector(1.0, C);
        alpha_dir[c] = alpha_confusion;
        confusion_matrix[c] ~ dirichlet(alpha_dir);
    }

    // ── Likelihoods ─────────────────────────────────────────────────────────
    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;

        if (is_llm_rating[n] == 0) {
            // Human: Dawid-Skene marginalized likelihood
            // log P(c_obs | model) = log( sum_{c_lat} P(latent_bin=c_lat) * confusion[c_lat][c_obs] )
            int c_obs = rating_values[n];
            real mix = 0.0;
            for (c_lat in 1:C) {
                mix += ordinal_bin_prob(
                    base_scores[ij_idx, k], total_std[ij_idx], bin_smoothing,
                    rating_thresholds[ij_idx][c_lat], rating_thresholds[ij_idx][c_lat+1]
                ) * confusion_matrix[c_lat][c_obs];
            }
            target += log(mix + 1e-10);

        } else {
            // LLM: Dirichlet likelihood  q_n ~ Dir(alpha_llm * pi_ijk)
            // Use pure ordinal-probit bin probabilities (no confusion for LLM)
            vector[C] pi_raw;
            for (c in 1:C) {
                pi_raw[c] = ordinal_bin_prob(
                    base_scores[ij_idx, k], total_std[ij_idx], bin_smoothing,
                    rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
                );
            }
            // Add small epsilon for numerical stability, renormalize to simplex
            vector[C] alpha_vec = alpha_llm * (pi_raw + 1e-6) / sum(pi_raw + 1e-6);
            target += dirichlet_lpdf(rating_dists[n] | alpha_vec);
        }
    }

}

generated quantities {

    // Observed log-likelihood (sum of appropriate likelihood per rating)
    real log_lik_ratings_obs = 0;
    real log_lik_pairwise_obs = 0;   // stub — no pairwise data in LLMRubric
    real total_log_lik;

    // Posterior predictive samples and distributions for missing ratings (all human)
    array[N_missing_ratings] int<lower=1, upper=C>  missing_rating_predictions;
    array[N_missing_ratings] vector[C]              missing_rating_probs;

    // ── Observed log-likelihood ──────────────────────────────────────────────
    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;

        if (is_llm_rating[n] == 0) {
            int c_obs = rating_values[n];
            real mix = 0.0;
            for (c_lat in 1:C) {
                mix += ordinal_bin_prob(
                    base_scores[ij_idx, k], total_std[ij_idx], bin_smoothing,
                    rating_thresholds[ij_idx][c_lat], rating_thresholds[ij_idx][c_lat+1]
                ) * confusion_matrix[c_lat][c_obs];
            }
            log_lik_ratings_obs += log(mix + 1e-10);
        } else {
            vector[C] pi_raw;
            for (c in 1:C) {
                pi_raw[c] = ordinal_bin_prob(
                    base_scores[ij_idx, k], total_std[ij_idx], bin_smoothing,
                    rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
                );
            }
            vector[C] alpha_vec = alpha_llm * (pi_raw + 1e-6) / sum(pi_raw + 1e-6);
            log_lik_ratings_obs += dirichlet_lpdf(rating_dists[n] | alpha_vec);
        }
    }

    total_log_lik = log_lik_ratings_obs;

    // ── Posterior predictive for missing ratings (human, Dawid-Skene) ────────
    for (n in 1:N_missing_ratings) {
        int i      = missing_rating_attributes[n];
        int j      = missing_rating_annotators[n];
        int k      = missing_rating_items[n];
        int ij_idx = (i-1)*J + j;

        real base_score = base_scores[ij_idx, k];

        // Step 1: compute latent bin probabilities from ordinal probit
        vector[C] bin_probs;
        for (c_lat in 1:C) {
            bin_probs[c_lat] = ordinal_bin_prob(
                base_score, total_std[ij_idx], bin_smoothing,
                rating_thresholds[ij_idx][c_lat], rating_thresholds[ij_idx][c_lat+1]
            );
        }

        // Step 2: mix through confusion matrix to get predictive distribution over observed categories
        // P(c_obs | model) = sum_{c_lat} P(latent_bin=c_lat) * confusion_matrix[c_lat][c_obs]
        for (c_obs in 1:C) {
            real p = 0.0;
            for (c_lat in 1:C) {
                p += bin_probs[c_lat] * confusion_matrix[c_lat][c_obs];
            }
            missing_rating_probs[n][c_obs] = p;
        }

        // Step 3: sample — first draw latent bin, then corrupt through confusion row
        int sampled_latent = categorical_rng(bin_probs);
        missing_rating_predictions[n] = categorical_rng(confusion_matrix[sampled_latent]);
    }

}

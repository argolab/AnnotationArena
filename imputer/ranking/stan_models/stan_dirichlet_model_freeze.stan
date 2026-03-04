/*
 * Round 2 variant of stan_dirichlet_model.stan for two-round inference.
 *
 * In Round 1 (stan_dirichlet_model.stan), all parameters are sampled on
 * training-instance ratings.  After Round 1, posterior means of:
 *   - mean_preferences       [I, D]
 *   - annotator_preferences  [I*J, D]
 *   - rating_probs           [I*J, C]
 * are extracted and passed here as fixed data.
 *
 * In Round 2 (this file), only item embeddings are free parameters.
 * Observed ratings are test-instance only, so embeddings[1..K_train]
 * receive no gradient signal and remain near their initialisation;
 * only embeddings[K_train+1..K] are meaningfully updated.
 *
 * Everything else — functions, transformed parameters, model structure,
 * generated quantities — is identical to stan_dirichlet_model.stan.
 */

functions {

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
    int<lower=1> K;   // total items  (K_train + K_test)
    int<lower=1> I;   // attributes / criteria
    int<lower=1> J;   // annotators  (humans + LLM)
    int<lower=1> D;   // embedding dimension
    int<lower=1> C;   // rating categories

    // ── Observed ratings (test-instance only in Round 2) ────────────────────
    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=I> rating_attributes;
    array[N_ratings] int<lower=1, upper=J> rating_annotators;
    array[N_ratings] int<lower=1, upper=K> rating_items;

    array[N_ratings] int<lower=1, upper=C> rating_values;
    array[N_ratings] simplex[C] rating_dists;
    array[N_ratings] int<lower=0, upper=1> is_llm_rating;

    // ── Missing ratings to predict (test-set human annotations) ─────────────
    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=I> missing_rating_attributes;
    array[N_missing_ratings] int<lower=1, upper=J> missing_rating_annotators;
    array[N_missing_ratings] int<lower=1, upper=K> missing_rating_items;

    // ── Hyperparameters ─────────────────────────────────────────────────────
    real<lower=0> sigma_annotator;
    real<lower=0> sigma_measurement;
    real<lower=0> alpha_dirichlet;
    real<lower=0> temperature;
    real<lower=0> alpha_llm;

    // ── Frozen Round 1 posterior means (moved from parameters to data) ───────
    matrix[I, D]     mean_preferences;        // v_i  — fixed from Round 1
    matrix[I*J, D]   annotator_preferences;   // v_ij — fixed from Round 1
    array[I*J] simplex[C] rating_probs;       // p_ij — fixed from Round 1

}

parameters {

    // Only item embeddings are free in Round 2.
    // Train item embeddings (rows 1..K_train) receive no gradient signal
    // because no train ratings are present; only test item embeddings
    // (rows K_train+1..K) are meaningfully updated.
    matrix[K, D] embeddings;

}

transformed parameters {

    matrix[I*J, K] base_scores;
    array[I*J] vector[C+1] rating_thresholds;
    array[I*J] real total_std;

    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], embeddings[k]);
            }
        }
    }

    for (ij in 1:(I*J)) {
        rating_thresholds[ij][1] = negative_infinity();
        for (c in 2:C) {
            rating_thresholds[ij][c] = inv_Phi(sum(rating_probs[ij][1:(c-1)]));
        }
        rating_thresholds[ij][C+1] = positive_infinity();
    }

    for (i in 1:I) {
        for (j in 1:J) {
            int ij_idx = (i-1)*J + j;
            total_std[ij_idx] = sqrt(dot_self(annotator_preferences[ij_idx])
                                     + sigma_measurement * sigma_measurement);
        }
    }

}

model {

    // ── Prior on embeddings only ─────────────────────────────────────────────
    // (mean_preferences, annotator_preferences, rating_probs are fixed data)
    for (k in 1:K) embeddings[k] ~ normal(0, 1);

    // ── Likelihoods ─────────────────────────────────────────────────────────
    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;

        if (is_llm_rating[n] == 0) {
            int c = rating_values[n];
            target += ordinal_log_prob(
                base_scores[ij_idx, k], total_std[ij_idx], sigma_measurement,
                rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
            );

        } else {
            vector[C] pi_raw;
            for (c in 1:C) {
                pi_raw[c] = ordinal_bin_prob(
                    base_scores[ij_idx, k], total_std[ij_idx], sigma_measurement,
                    rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
                );
            }
            vector[C] alpha_vec = alpha_llm * (pi_raw + 1e-6) / sum(pi_raw + 1e-6);
            target += dirichlet_lpdf(rating_dists[n] | alpha_vec);
        }
    }

}

generated quantities {

    real log_lik_ratings_obs = 0;
    real log_lik_pairwise_obs = 0;
    real total_log_lik;

    array[N_missing_ratings] int<lower=1, upper=C>  missing_rating_predictions;
    array[N_missing_ratings] vector[C]              missing_rating_probs;

    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;

        if (is_llm_rating[n] == 0) {
            int c = rating_values[n];
            log_lik_ratings_obs += ordinal_log_prob(
                base_scores[ij_idx, k], total_std[ij_idx], sigma_measurement,
                rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
            );
        } else {
            vector[C] pi_raw;
            for (c in 1:C) {
                pi_raw[c] = ordinal_bin_prob(
                    base_scores[ij_idx, k], total_std[ij_idx], sigma_measurement,
                    rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
                );
            }
            vector[C] alpha_vec = alpha_llm * (pi_raw + 1e-6) / sum(pi_raw + 1e-6);
            log_lik_ratings_obs += dirichlet_lpdf(rating_dists[n] | alpha_vec);
        }
    }

    total_log_lik = log_lik_ratings_obs;

    for (n in 1:N_missing_ratings) {
        int i      = missing_rating_attributes[n];
        int j      = missing_rating_annotators[n];
        int k      = missing_rating_items[n];
        int ij_idx = (i-1)*J + j;

        real base_score = base_scores[ij_idx, k];

        for (c in 1:C) {
            missing_rating_probs[n][c] = ordinal_bin_prob(
                base_score, total_std[ij_idx], sigma_measurement,
                rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
            );
        }

        real noisy_score  = base_score + normal_rng(0, sigma_measurement);
        real standardized = noisy_score / total_std[ij_idx];
        int  rating       = C;
        for (c in 1:C) {
            if (standardized <= rating_thresholds[ij_idx][c+1]) {
                rating = c;
                break;
            }
        }
        missing_rating_predictions[n] = rating;
    }

}
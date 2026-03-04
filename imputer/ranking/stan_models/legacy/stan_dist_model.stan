/*
 * Domain model for annotation inference with soft (distributional) labels.
 *
 * Unified model where ALL observed ratings provide a full distribution over
 * rating categories:
 *   - Human annotators: one-hot(c) — reduces exactly to standard hard-label CE
 *   - LLM annotators:   actual probability distribution over C categories
 *
 * Likelihood for each rating n:
 *   log p(q_n | model) = sum_c  q_n[c] * log P(c | model)
 *
 * This is the expected log-likelihood (cross-entropy) under q_n.  It is a
 * proper scoring rule: optimal when the model matches q_n exactly.
 *
 * Parameters and priors are identical to domain_model.stan.
 * Pairwise rankings are omitted (HANNA / LLMRubric contain none).
 * Stub generated quantities are included for compatibility with
 * evaluate_predictions.py.
 */

functions {
    // Log probability of bin [lower_threshold, upper_threshold) under the
    // ordinal-probit model with given base_score and measurement noise.
    real ordinal_log_prob(
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

        return log(upper_prob - lower_prob + 1e-10);
    }
}

data {
    // ── Dimensions ──────────────────────────────────────────────────────────
    int<lower=1> K;   // total items  (K_train + K_test)
    int<lower=1> I;   // attributes / criteria
    int<lower=1> J;   // annotators  (humans + LLM)
    int<lower=1> D;   // embedding dimension
    int<lower=1> C;   // rating categories

    // ── Observed ratings (all as distributions) ──────────────────────────────
    // For human annotators:  rating_dists[n] = one_hot(value)
    // For LLM annotators:    rating_dists[n] = actual soft distribution
    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=I> rating_attributes;
    array[N_ratings] int<lower=1, upper=J> rating_annotators;
    array[N_ratings] int<lower=1, upper=K> rating_items;
    array[N_ratings] simplex[C]            rating_dists;   // q[C], must sum to 1

    // ── Missing ratings to predict (test-set human annotations) ─────────────
    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=I> missing_rating_attributes;
    array[N_missing_ratings] int<lower=1, upper=J> missing_rating_annotators;
    array[N_missing_ratings] int<lower=1, upper=K> missing_rating_items;

    // ── Hyperparameters ─────────────────────────────────────────────────────
    real<lower=0> sigma_annotator;    // std of annotator pref around mean pref
    real<lower=0> sigma_measurement;  // measurement noise std
    real<lower=0> alpha_dirichlet;    // Dirichlet concentration for rating probs
    real<lower=0> temperature;        // unused here; kept for interface consistency
}

parameters {
    // Item embeddings: e_k ~ N(0, I_D)
    matrix[K, D] embeddings;

    // Mean preferences per attribute: v_i ~ N(0, I_D)
    matrix[I, D] mean_preferences;

    // Per-annotator preferences: v_ij ~ N(v_i, sigma_annotator^2 I_D)
    matrix[I*J, D] annotator_preferences;

    // Rating probability simplex per (i,j): p_ij ~ Dir(alpha/C, ..., alpha/C)
    array[I*J] simplex[C] rating_probs;
}

transformed parameters {
    // Base utility scores: z_ijk = v_ij · e_k
    matrix[I*J, K] base_scores;

    // Ordinal thresholds derived from rating_probs via inverse-normal CDF
    // threshold[ij][1] = -inf,  threshold[ij][C+1] = +inf
    array[I*J] vector[C+1] rating_thresholds;

    // total_std[ij] = sqrt(||v_ij||^2 + sigma_measurement^2)
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
    // ── Priors ──────────────────────────────────────────────────────────────
    for (k in 1:K) embeddings[k] ~ normal(0, 1);
    for (i in 1:I) mean_preferences[i] ~ normal(0, 1);

    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            annotator_preferences[idx] ~ normal(mean_preferences[i], sigma_annotator);
        }
    }

    for (ij in 1:(I*J)) {
        rating_probs[ij] ~ dirichlet(rep_vector(alpha_dirichlet / C, C));
    }

    // ── Soft-label likelihood: expected log-prob under q ─────────────────────
    // log p(q | model) = sum_c  q_c * log P(c | model)
    // For human one-hot(c): reduces to log P(c | model) exactly.
    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;

        real ll = 0;
        for (c in 1:C) {
            ll += rating_dists[n][c] * ordinal_log_prob(
                base_scores[ij_idx, k],
                total_std[ij_idx],
                sigma_measurement,
                rating_thresholds[ij_idx][c],
                rating_thresholds[ij_idx][c+1]
            );
        }
        target += ll;
    }
}

generated quantities {
    // Observed log-likelihood (total across all ratings)
    real log_lik_ratings_obs = 0;
    real log_lik_pairwise_obs = 0;   // stub — no pairwise data
    real total_log_lik;

    // Posterior predictive samples and distributions for missing ratings
    array[N_missing_ratings] int<lower=1, upper=C> missing_rating_predictions;
    array[N_missing_ratings] vector[C]             missing_rating_probs;

    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;
        for (c in 1:C) {
            log_lik_ratings_obs += rating_dists[n][c] * ordinal_log_prob(
                base_scores[ij_idx, k], total_std[ij_idx], sigma_measurement,
                rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
            );
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
        real mean_std   = base_score / total_std[ij_idx];
        real cond_std   = sigma_measurement / total_std[ij_idx];

        for (c in 1:C) {
            real upper_prob = (rating_thresholds[ij_idx][c+1] == positive_infinity())
                              ? 1.0
                              : Phi((rating_thresholds[ij_idx][c+1] - mean_std) / cond_std);
            real lower_prob = (rating_thresholds[ij_idx][c] == negative_infinity())
                              ? 0.0
                              : Phi((rating_thresholds[ij_idx][c] - mean_std) / cond_std);
            missing_rating_probs[n][c] = upper_prob - lower_prob;
        }

        // Sample a rating from the posterior predictive
        real noisy_score = base_score + normal_rng(0, sigma_measurement);
        real standardized = noisy_score / total_std[ij_idx];
        int  rating = C;   // fallback to top category
        for (c in 1:C) {
            if (standardized <= rating_thresholds[ij_idx][c+1]) {
                rating = c;
                break;
            }
        }
        missing_rating_predictions[n] = rating;
    }
}

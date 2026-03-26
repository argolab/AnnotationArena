/*
 * Domain model with Dirichlet likelihood for LLM distributional observations.
 * Extended with factored annotator model (use_factored_annotator flag).
 *
 * Human annotators provide hard ratings:
 *   c_n ~ Categorical(pi_ijk)  →  log P(c_n | model) via ordinal probit
 *
 * LLM annotators provide a full distribution q over categories:
 *   q_n ~ Dir(alpha_llm * pi_ijk)
 *
 * where pi_ijk = [P(1|model), ..., P(C|model)] from the ordinal-probit bin
 * probabilities for annotator j, criterion i, item k.
 *
 * Two annotator models are supported (controlled by use_factored_annotator):
 *
 *   use_factored_annotator = 0  (OLD spherical model):
 *     V_ij ~ N(v_i, sigma_annotator^2 * I_D)  independently
 *     annotator_preferences is directly parameterized.
 *
 *   use_factored_annotator = 1  (NEW factored model):
 *     V_ij = v_i + u_j * M_i
 *     where u_j in R^{d_annotator} is the annotator embedding and
 *     M_i in R^{d_annotator x D} is the attribute-specific transform.
 *     annotator_preferences[idx] is derived as a transformed parameter.
 *
 * The alpha_llm hyperparameter encodes LLM reliability:
 *   - Large alpha_llm → LLM output concentrates tightly around pi_ijk (high signal)
 *   - Small alpha_llm → LLM output is diffuse regardless of pi_ijk (low signal)
 *
 * Parameters and priors follow domain_model.stan / stan_dist_model.stan.
 * Missing-rating generated quantities are identical — posterior predictive is
 * always over human ratings (ordinal probit), so evaluate_predictions.py is reusable.
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

    // Routing indicator: 0 = human (ordinal probit), 1 = LLM (Dirichlet).
    array[N_ratings] int<lower=0, upper=1> is_llm_rating;

    // ── Missing ratings to predict (test-set human annotations) ─────────────
    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=I> missing_rating_attributes;
    array[N_missing_ratings] int<lower=1, upper=J> missing_rating_annotators;
    array[N_missing_ratings] int<lower=1, upper=K> missing_rating_items;

    // ── Hyperparameters ─────────────────────────────────────────────────────
    real<lower=0> sigma_annotator;    // std of annotator preference around mean
    real<lower=0> sigma_measurement;  // measurement noise std
    real<lower=0> alpha_dirichlet;    // Dirichlet prior concentration for rating_probs
    real<lower=0> temperature;        // unused here; kept for interface consistency
    real<lower=0> alpha_llm;          // Dirichlet likelihood concentration for LLM obs

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

    // total_std[ij] = sqrt(||v_ij||^2 + sigma_measurement^2)
    array[I*J] real total_std;
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

    // ── Likelihoods ─────────────────────────────────────────────────────────
    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;

        if (is_llm_rating[n] == 0) {
            // Human: hard-label ordinal probit likelihood  log P(c | model)
            int c = rating_values[n];
            target += ordinal_log_prob(
                base_scores[ij_idx, k], total_std[ij_idx], sigma_measurement,
                rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
            );

        } else {
            // LLM: Dirichlet likelihood  q_n ~ Dir(alpha_llm * pi_ijk)
            // Step 1: compute ordinal-probit bin probabilities pi_ijk
            vector[C] pi_raw;
            for (c in 1:C) {
                pi_raw[c] = ordinal_bin_prob(
                    base_scores[ij_idx, k], total_std[ij_idx], sigma_measurement,
                    rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
                );
            }
            // Step 2: add small epsilon for numerical stability, renormalize to simplex
            vector[C] alpha_vec = alpha_llm * (pi_raw + 1e-6) / sum(pi_raw + 1e-6);
            // Step 3: Dirichlet log-likelihood
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

    // ── Posterior predictive for missing ratings (human, ordinal probit) ─────
    for (n in 1:N_missing_ratings) {
        int i      = missing_rating_attributes[n];
        int j      = missing_rating_annotators[n];
        int k      = missing_rating_items[n];
        int ij_idx = (i-1)*J + j;

        real base_score = base_scores[ij_idx, k];

        // Compute predicted distribution over Likert scale
        for (c in 1:C) {
            missing_rating_probs[n][c] = ordinal_bin_prob(
                base_score, total_std[ij_idx], sigma_measurement,
                rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
            );
        }

        // Sample a rating via noisy score → ordinal bins
        real noisy_score  = base_score + normal_rng(0, sigma_measurement);
        real standardized = noisy_score / total_std[ij_idx];
        int  rating       = C;   // fallback to top category
        for (c in 1:C) {
            if (standardized <= rating_thresholds[ij_idx][c+1]) {
                rating = c;
                break;
            }
        }
        missing_rating_predictions[n] = rating;
    }

}

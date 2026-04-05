/*
 * Domain model for CP tensor decomposition inference using MCMC
 * Learns CP factors (v_i, u_j, e_k) and component weights T_d from ratings and comparisons
 * Score: Z_ijk = sum_d v_id * u_jd * e_kd * T_d where T_d = factor_decay^(d-1)
 * Loadings are real-valued (normal prior) so the model can fit dot-product and other signed data.
 *
 * LLM ratings are handled via a Dirichlet likelihood (as in stan_dirichlet_model.stan):
 *   q_n ~ Dir(alpha_llm * pi_ijk)
 * Human ratings use the standard ordinal probit hard-label likelihood.
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
        real cond_std = fmax(sigma_measurement / total_std_ij, 1e-10);

        real phi_arg_upper, phi_arg_lower;
        real upper_prob, lower_prob;

        if (upper_threshold == positive_infinity()) {
            upper_prob = 1.0;
        } else {
            real raw = (upper_threshold - mean_std) / cond_std;
            phi_arg_upper = fmax(fmin(raw, 20.0), -20.0);
            upper_prob = Phi(phi_arg_upper);
        }

        if (lower_threshold == negative_infinity()) {
            lower_prob = 0.0;
        } else {
            real raw = (lower_threshold - mean_std) / cond_std;
            phi_arg_lower = fmax(fmin(raw, 20.0), -20.0);
            lower_prob = Phi(phi_arg_lower);
        }

        return upper_prob - lower_prob;
    }

}

data {
    // Dimensions
    int<lower=1> K;  // number of items in this instance
    int<lower=1> I;  // number of criteria (attributes)
    int<lower=1> J;  // number of annotators  
    int<lower=1> D;  // CP rank (number of components)
    int<lower=1> C;  // number of rating categories
    
    // Observed ratings
    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=I> rating_attributes;
    array[N_ratings] int<lower=1, upper=J> rating_annotators;
    array[N_ratings] int<lower=1, upper=K> rating_items;

    // Hard integer labels (1..C): used for human ratings (is_llm_rating==0).
    // For LLM ratings set to the argmax value — not used in likelihood.
    array[N_ratings] int<lower=1, upper=C> rating_values;

    // Full distributions over C categories: used for LLM ratings (is_llm_rating==1).
    // For human ratings set to the one-hot — not used in likelihood.
    array[N_ratings] simplex[C] rating_dists;

    // Routing indicator: 0 = human (ordinal probit), 1 = LLM (Dirichlet).
    array[N_ratings] int<lower=0, upper=1> is_llm_rating;
    
    // Missing variables to predict (optional)
    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=I> missing_rating_attributes;
    array[N_missing_ratings] int<lower=1, upper=J> missing_rating_annotators;
    array[N_missing_ratings] int<lower=1, upper=K> missing_rating_items;
    
    
    // Hyperparameters
    real<lower=0> sigma_annotator;    // unused in CP model, kept for pipeline compatibility
    real<lower=0> sigma_measurement;  // measurement noise std
    real<lower=0> kappa;              // Dirichlet concentration for rating thresholds
    real<lower=0> temperature;        // unused, kept for pipeline compatibility
    real<lower=0> factor_decay;       // T_d = factor_decay^(d-1), controls rank structure
    real<lower=0> alpha_llm;          // Dirichlet likelihood concentration for LLM observations
    
    // Pipeline compatibility (d_annotator should equal D for CP)
    int<lower=1> d_annotator;
    // Debug init: set to 1 via --stan-arg DEBUG_INIT=1 to reject() with which variable is non-finite
    int<lower=0,upper=1> DEBUG_INIT;
}

transformed data {
    // Debug printing flag (0 = off, 1 = on). Toggle here.
    int DEBUG_PRINT;
    DEBUG_PRINT = 0;
    
    // Component weights: T_d = factor_decay^(d-1)
    vector[D] T_weights;
    for (d in 1:D) {
        T_weights[d] = pow(factor_decay, d - 1);
    }
    
    real score_center = 0;
}

parameters {
    // CP factor loadings (real-valued)
    matrix[I, D] v_loadings;
    matrix[J, D] u_loadings;
    matrix[K, D] e_loadings;
    
    // Rating probabilities: p_ij ~ Dir(κ/C, ..., κ/C)
    // Shared per annotator j (not per (i,j)) for CP model
    array[J] simplex[C] rating_probs;
}

transformed parameters {
    matrix[I*J, D] annotator_preferences;
    matrix[I*J, K] base_scores;
    array[I*J] vector[C+1] rating_thresholds;
    array[I*J] real total_std;
    
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (d in 1:D) {
                annotator_preferences[idx, d] = v_loadings[i, d] * u_loadings[j, d] * T_weights[d];
            }
        }
    }
    
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], e_loadings[k]) - score_center;
            }
        }
    }
    
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            rating_thresholds[idx][1] = negative_infinity();
            for (c in 2:C) {
                real cum_prob = sum(rating_probs[j][1:(c-1)]);
                cum_prob = fmin(fmax(cum_prob, 1e-6), 1.0 - 1e-6);
                rating_thresholds[idx][c] = inv_Phi(cum_prob);
            }
            rating_thresholds[idx][C+1] = positive_infinity();
        }
    }
    
    for (i in 1:I) {
        for (j in 1:J) {
            int ij_idx = (i-1)*J + j;
            real pref_norm_sq = dot_self(annotator_preferences[ij_idx]);
            total_std[ij_idx] = fmax(sqrt(pref_norm_sq + sigma_measurement * sigma_measurement), 1e-10);
        }
    }

    if (DEBUG_PRINT == 1) {
        int max_j_print = (J < 3) ? J : 3;
        for (j in 1:max_j_print) {
            print("[DEBUG TP] j=", j, " rating_probs=", rating_probs[j]);
            int idx = j;
            print("[DEBUG TP] ij=", idx, " thresholds=", rating_thresholds[idx]);
        }
        print("[DEBUG TP] T_weights=", T_weights);
        print("[DEBUG TP] score_center=", score_center);
    }
}

model {
    // ===== DEBUG INIT =====
    if (DEBUG_INIT == 1) {
        if (temperature <= 0 || is_nan(temperature) || is_inf(temperature))
            reject("DEBUG_INIT: temperature is non-positive or non-finite, value=", temperature);
        if (sigma_measurement <= 0 || is_nan(sigma_measurement) || is_inf(sigma_measurement))
            reject("DEBUG_INIT: sigma_measurement is non-positive or non-finite, value=", sigma_measurement);
        for (i in 1:I)
            for (d in 1:D)
                if (is_nan(v_loadings[i,d]) || is_inf(v_loadings[i,d]))
                    reject("DEBUG_INIT: v_loadings[", i, ",", d, "] non-finite, value=", v_loadings[i,d]);
        for (j in 1:J)
            for (d in 1:D)
                if (is_nan(u_loadings[j,d]) || is_inf(u_loadings[j,d]))
                    reject("DEBUG_INIT: u_loadings[", j, ",", d, "] non-finite, value=", u_loadings[j,d]);
        for (k in 1:K)
            for (d in 1:D)
                if (is_nan(e_loadings[k,d]) || is_inf(e_loadings[k,d]))
                    reject("DEBUG_INIT: e_loadings[", k, ",", d, "] non-finite, value=", e_loadings[k,d]);
        for (idx in 1:(I*J)) {
            if (is_nan(total_std[idx]) || is_inf(total_std[idx]) || total_std[idx] <= 0)
                reject("DEBUG_INIT: total_std[", idx, "] is bad, value=", total_std[idx]);
        }
        for (idx in 1:(I*J))
            for (c in 1:(C+1)) {
                real th = rating_thresholds[idx][c];
                if (c > 1 && c <= C && (is_nan(th) || is_inf(th)))
                    reject("DEBUG_INIT: rating_thresholds[", idx, ",", c, "] is non-finite, value=", th);
            }
        for (idx in 1:(I*J))
            for (k in 1:K) {
                real bs = base_scores[idx, k];
                if (is_nan(bs) || is_inf(bs))
                    reject("DEBUG_INIT: base_scores[", idx, ",", k, "] is non-finite, value=", bs);
            }
        for (j in 1:J)
            for (c in 1:C) {
                real p = rating_probs[j][c];
                if (is_nan(p) || is_inf(p))
                    reject("DEBUG_INIT: rating_probs[", j, ",", c, "] is non-finite, value=", p);
            }
    }

    // ===== PRIORS =====
    for (i in 1:I)
        for (d in 1:D)
            v_loadings[i, d] ~ normal(0, 1);
    for (j in 1:J)
        for (d in 1:D)
            u_loadings[j, d] ~ normal(0, 1);
    for (k in 1:K)
        for (d in 1:D)
            e_loadings[k, d] ~ normal(0, 1);
    for (j in 1:J)
        rating_probs[j] ~ dirichlet(rep_vector(kappa / C, C));
    
    // ===== RATING LIKELIHOOD =====
    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;

        if (is_llm_rating[n] == 0) {
            // Human: hard-label ordinal probit  log P(c | model)
            int c = rating_values[n];
            real bin_prob = ordinal_bin_prob(
                base_scores[ij_idx, k], total_std[ij_idx], sigma_measurement,
                rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
            );
            target += log(fmax(bin_prob, 1e-8));

        } else {
            // LLM: Dirichlet likelihood  q_n ~ Dir(alpha_llm * pi_ijk)
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
    real total_log_lik = 0;
    
    array[N_missing_ratings] int<lower=1, upper=C> missing_rating_predictions;
    array[N_missing_ratings] vector[C] missing_rating_probs;
    
    // Pipeline compatibility
    matrix[I, D] mean_preferences;
    matrix[J, d_annotator] annotator_embeddings;
    
    for (i in 1:I)
        for (d in 1:D)
            mean_preferences[i, d] = v_loadings[i, d];
    for (j in 1:J) {
        for (d in 1:min(d_annotator, D))
            annotator_embeddings[j, d] = u_loadings[j, d];
        for (d in (D+1):d_annotator)
            annotator_embeddings[j, d] = 0;
    }
    
    // ===== OBSERVED LOG-LIKELIHOOD =====
    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;

        if (is_llm_rating[n] == 0) {
            // Human: ordinal probit
            int c = rating_values[n];
            real bin_prob = ordinal_bin_prob(
                base_scores[ij_idx, k], total_std[ij_idx], sigma_measurement,
                rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
            );
            log_lik_ratings_obs += log(bin_prob + 1e-10);

            if (DEBUG_PRINT == 1 && n <= 10)
                print("[DEBUG RATING] n=", n, " human c=", c,
                      " base_score=", base_scores[ij_idx, k],
                      " bin_prob=", bin_prob);
        } else {
            // LLM: Dirichlet
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

    
    
    // ===== POSTERIOR PREDICTIVE FOR MISSING RATINGS =====
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
        
        real noisy_score      = base_score + normal_rng(0, sigma_measurement);
        real standardized     = noisy_score / total_std[ij_idx];
        int  rating           = C;
        for (c in 1:C) {
            if (standardized <= rating_thresholds[ij_idx][c+1]) {
                rating = c;
                break;
            }
        }
        missing_rating_predictions[n] = rating;

        if (DEBUG_PRINT == 1 && n <= 10)
            print("[DEBUG MRATING] n=", n,
                  " base_score=", base_score,
                  " standardized=", standardized,
                  " probs=", missing_rating_probs[n],
                  " sampled=", rating);
    }
    

}

/*
 * Domain model for CP tensor decomposition inference using MCMC
 * Learns CP factors (v_i, u_j, e_k) and component weights T_d from ratings and comparisons
 * Score: Z_ijk = sum_d v_id * u_jd * e_kd * T_d where T_d = factor_decay^(d-1)
 * Loadings are real-valued (normal prior) so the model can fit dot-product and other signed data.
 */

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
    array[N_ratings] int<lower=1, upper=C> rating_values;
    
    // Observed pairwise rankings (Bradley-Terry model)
    int<lower=0> N_pairwise_rankings;
    array[N_pairwise_rankings] int<lower=1, upper=I> pairwise_ranking_attributes;
    array[N_pairwise_rankings] int<lower=1, upper=J> pairwise_ranking_annotators;
    array[N_pairwise_rankings, 2] int<lower=1, upper=K> pairwise_ranking_items;  // [item1, item2] pairs
    array[N_pairwise_rankings] int<lower=1, upper=2> pairwise_ranking_orders;    // 1 if item1 > item2, 2 if item2 > item1
    
    // Missing variables to predict (optional)
    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=I> missing_rating_attributes;
    array[N_missing_ratings] int<lower=1, upper=J> missing_rating_annotators;
    array[N_missing_ratings] int<lower=1, upper=K> missing_rating_items;
    
    int<lower=0> N_missing_pairwise_rankings;
    array[N_missing_pairwise_rankings] int<lower=1, upper=I> missing_pairwise_ranking_attributes;
    array[N_missing_pairwise_rankings] int<lower=1, upper=J> missing_pairwise_ranking_annotators;
    array[N_missing_pairwise_rankings, 2] int<lower=1, upper=K> missing_pairwise_ranking_items;
    
    // Hyperparameters
    real<lower=0> sigma_annotator;    // unused in CP model, kept for pipeline compatibility
    real<lower=0> sigma_measurement;  // measurement noise std
    real<lower=0> kappa;    // Dirichlet concentration for rating thresholds
    real<lower=0> temperature;        // unused, kept for pipeline compatibility
    real<lower=0> factor_decay;       // T_d = factor_decay^(d-1), controls rank structure
    
    // Pipeline compatibility (d_annotator should equal D for CP)
    int<lower=1> d_annotator;
    // Debug init: set to 1 via --stan-arg DEBUG_INIT=1 to reject() with which variable is non-finite
    int<lower=0,upper=1> DEBUG_INIT;
}

transformed data {
    // Debug printing flag (0 = off, 1 = on). Toggle here.
    int DEBUG_PRINT;
    DEBUG_PRINT = 0;
    
    // Component weights: T_d = factor_decay^(d-1) TODO: this is currently known by setting factor_decay, but can also be learned with a exp/flat prior.
    vector[D] T_weights;
    for (d in 1:D) {
        T_weights[d] = pow(factor_decay, d - 1);
    }
    
    // Score centering: with normal(0,1) loadings E[v*u*e]=0 so E[Z]=0
    real score_center = 0;
}

parameters {
    // CP factor loadings (real-valued so model can fit dot-product and other signed data)
    // v_id: attribute loadings
    matrix[I, D] v_loadings;
    
    // u_jd: annotator loadings
    matrix[J, D] u_loadings;
    
    // e_kd: item embeddings
    matrix[K, D] e_loadings;
    
    // Rating probabilities: p_ij ~ Dir(α/C, ..., α/C)
    // Shared per annotator j (not per (i,j)) for CP model
    array[J] simplex[C] rating_probs;
}

transformed parameters {
    // Effective annotator preferences: annotator_preferences[ij, d] = v_i[d] * u_j[d] * T_d
    // This allows base_score = dot_product(annotator_preferences[ij], e_k) = Z_ijk
    matrix[I*J, D] annotator_preferences;
    
    // Base utility scores: z_ijk = sum_d v_id * u_jd * e_kd * T_d - score_center
    matrix[I*J, K] base_scores;
    
    // Rating thresholds: q_ij = Φ⁻¹(cumsum(p_ij))
    array[I*J] vector[C+1] rating_thresholds;
    
    // Total standard deviation per annotator-criterion for rating binning
    // Binning is based on distribution of Z + epsilon with std = sqrt(||annotator_preferences[ij]||^2 + sigma_measurement^2)
    array[I*J] real total_std;
    
    // Compute effective annotator preferences
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (d in 1:D) {
                annotator_preferences[idx, d] = v_loadings[i, d] * u_loadings[j, d] * T_weights[d];
            }
        }
    }
    
    // Compute base scores: z_ijk = sum_d v_id * u_jd * e_kd * T_d - score_center
    // = dot_product(annotator_preferences[ij], e_k) - score_center
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], e_loadings[k]) - score_center;
            }
        }
    }
    
    // Convert probabilities to thresholds using inverse normal CDF
    // Rating probabilities are shared per annotator j (replicated across attributes i)
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            rating_thresholds[idx][1] = negative_infinity();  // -∞ for category 1
            
            // Convert cumulative probabilities to thresholds (clamp to avoid inf gradient at 0/1)
            for (c in 2:C) {
                real cum_prob = sum(rating_probs[j][1:(c-1)]);
                cum_prob = fmin(fmax(cum_prob, 1e-6), 1.0 - 1e-6);
                rating_thresholds[idx][c] = inv_Phi(cum_prob);
            }
            
            rating_thresholds[idx][C+1] = positive_infinity();  // +∞ for category C
        }
    }
    
    // Compute total_std per ij for rating binning (bounded below to avoid 0/0 or NaN in Phi)
    // Binning is based on distribution of Z + epsilon with std = sqrt(||annotator_preferences[ij]||^2 + sigma_measurement^2)
    for (i in 1:I) {
        for (j in 1:J) {
            int ij_idx = (i-1)*J + j;
            real pref_norm_sq = dot_self(annotator_preferences[ij_idx]);
            total_std[ij_idx] = fmax(sqrt(pref_norm_sq + sigma_measurement * sigma_measurement), 1e-10);
        }
    }

    // ===== DEBUG: print a few rating_probs and thresholds per draw =====
    if (DEBUG_PRINT == 1) {
        int max_j_print = (J < 3) ? J : 3;
        for (j in 1:max_j_print) {
            print("[DEBUG TP] j=", j, " rating_probs=", rating_probs[j]);
            int idx = j;  // i=1
            print("[DEBUG TP] ij=", idx, " thresholds=", rating_thresholds[idx]);
        }
        print("[DEBUG TP] T_weights=", T_weights);
        print("[DEBUG TP] score_center=", score_center);
    }
}

model {
    // ===== DEBUG INIT: locate non-finite gradient (set DEBUG_INIT=1 via --stan-arg) =====
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
        for (idx in 1:(I*J)) {
            for (c in 1:(C+1)) {
                real th = rating_thresholds[idx][c];
                if (c > 1 && c <= C && (is_nan(th) || is_inf(th)))
                    reject("DEBUG_INIT: rating_thresholds[", idx, ",", c, "] is non-finite, value=", th);
            }
        }
        for (idx in 1:(I*J)) {
            for (k in 1:K) {
                real bs = base_scores[idx, k];
                if (is_nan(bs) || is_inf(bs))
                    reject("DEBUG_INIT: base_scores[", idx, ",", k, "] is non-finite, value=", bs);
            }
        }
        for (j in 1:J) {
            for (c in 1:C) {
                real p = rating_probs[j][c];
                if (is_nan(p) || is_inf(p))
                    reject("DEBUG_INIT: rating_probs[", j, ",", c, "] is non-finite, value=", p);
            }
        }
    }
    // ===== PRIORS =====
    
    // CP factor loadings: normal(0, 1) so scores can be negative (fits dot-product and misspecified data)
    // Softer than lognormal: finite gradient at init, and all data have positive probability
    for (i in 1:I) {
        for (d in 1:D) {
            v_loadings[i, d] ~ normal(0, 1);
        }
    }
    for (j in 1:J) {
        for (d in 1:D) {
            u_loadings[j, d] ~ normal(0, 1);
        }
    }
    for (k in 1:K) {
        for (d in 1:D) {
            e_loadings[k, d] ~ normal(0, 1);
        }
    }
    
    // Rating probabilities: p_j ~ Dir(α/C, ..., α/C) per annotator
    for (j in 1:J) {
        rating_probs[j] ~ dirichlet(rep_vector(kappa / C, C));
    }
    
    // ===== LIKELIHOODS =====
    
    // 1. RATING LIKELIHOOD
    // Binning is based on distribution of Z + epsilon with std = sqrt(||annotator_preferences[ij]||^2 + sigma_measurement^2)
    // Conditional distribution: Z + epsilon | v, u, e ~ N(Z, sigma_measurement^2)
    // Standardize to match binning space: mean = base_score / total_std, std = sigma_measurement / total_std
    for (n in 1:N_ratings) {
        int i = rating_attributes[n];
        int j = rating_annotators[n]; 
        int k = rating_items[n];
        int c = rating_values[n];
        int ij_idx = (i-1)*J + j;
        
        real base_score = base_scores[ij_idx, k];
        real upper_threshold = rating_thresholds[ij_idx][c+1];  // threshold in z-space (standardized by total_std)
        real lower_threshold = rating_thresholds[ij_idx][c];  // threshold in z-space
        
        // Compute P(category c | base_score) using conditional distribution
        // Conditional: Z + epsilon | v, u, e ~ N(Z, sigma_measurement^2)
        // Standardized: mean = base_score / total_std, std = sigma_measurement / total_std
        real mean_std = base_score / total_std[ij_idx];
        real cond_std = fmax(sigma_measurement / total_std[ij_idx], 1e-10);  // avoid 0 -> inf gradient
        
        real upper_prob, lower_prob;
        // Clamp Phi argument to finite range to avoid NaN/Inf from bad proposals (Phi(-20)~0, Phi(20)~1)
        real raw_upper = (upper_threshold - mean_std) / cond_std;
        real raw_lower = (lower_threshold - mean_std) / cond_std;
        real phi_arg_upper = (upper_threshold == positive_infinity()) ? 20.0 : (is_nan(raw_upper) ? 0.0 : fmax(fmin(raw_upper, 20.0), -20.0));
        real phi_arg_lower = (lower_threshold == negative_infinity()) ? -20.0 : (is_nan(raw_lower) ? 0.0 : fmax(fmin(raw_lower, 20.0), -20.0));
        
        if (upper_threshold == positive_infinity()) {
            upper_prob = 1.0;
        } else {
            upper_prob = Phi(phi_arg_upper);
        }
        
        if (lower_threshold == negative_infinity()) {
            lower_prob = 0.0;
        } else {
            lower_prob = Phi(phi_arg_lower);
        }
        
        real bin_prob = upper_prob - lower_prob;
        
        // Numerical stability
        if (bin_prob > 1e-8) {
            target += log(bin_prob);
        } else {
            target += log(1e-8);
        }
    }
    
    // 2. PAIRWISE RANKING LIKELIHOOD (Bradley-Terry model with Gumbel noise)
    for (n in 1:N_pairwise_rankings) {
        int i = pairwise_ranking_attributes[n];
        int j = pairwise_ranking_annotators[n];
        int ij_idx = (i-1)*J + j;
        
        // Pairwise comparison: [item1, item2] with order 1 or 2
        int item1 = pairwise_ranking_items[n, 1];
        int item2 = pairwise_ranking_items[n, 2];
        int order = pairwise_ranking_orders[n];  // 1 if item1 > item2, 2 if item2 > item1
        
        // Base scores scaled by temperature: z_ijk/T (bound temperature to avoid inf gradient)
        real safe_temp = fmax(temperature, 1e-10);
        real score1 = base_scores[ij_idx, item1] / safe_temp;
        real score2 = base_scores[ij_idx, item2] / safe_temp;
        
        // Bradley-Terry likelihood: P(item1 > item2) = exp(score1) / (exp(score1) + exp(score2))
        // This is equivalent to log_inv_logit(score1 - score2)
        if (order == 1) {  // item1 > item2
            target += log_inv_logit(score1 - score2);
        } else if (order == 2) {  // item2 > item1
            target += log_inv_logit(score2 - score1);
        } else {
            reject("Error: pairwise_ranking_orders[n] must be 1 or 2.");
        }
    }
}

generated quantities {
    // Log-likelihood components for evaluation
    real log_lik_ratings_obs = 0;        // Log-likelihood of observed ratings
    real log_lik_pairwise_obs = 0;       // Log-likelihood of observed pairwise rankings
    real total_log_lik = 0;
    
    // Posterior predictive samples for missing variables
    array[N_missing_ratings] int<lower=1, upper=C> missing_rating_predictions;
    array[N_missing_pairwise_rankings] int<lower=1, upper=2> missing_pairwise_ranking_predictions;
    
    // Predicted distributions for missing variables (for evaluation)
    array[N_missing_ratings] vector[C] missing_rating_probs;        // Predicted probability distribution over Likert scale
    array[N_missing_pairwise_rankings] real missing_pairwise_logits; // Predicted Bradley-Terry logits
    
    // Pipeline compatibility: mean_preferences and annotator_embeddings for bundle compatibility
    matrix[I, D] mean_preferences;              // = v_loadings (attribute loadings)
    matrix[J, d_annotator] annotator_embeddings; // = u_loadings (annotator loadings, truncated to d_annotator)
    
    // Compute compatibility arrays
    for (i in 1:I) {
        for (d in 1:D) {
            mean_preferences[i, d] = v_loadings[i, d];
        }
    }
    for (j in 1:J) {
        for (d in 1:min(d_annotator, D)) {
            annotator_embeddings[j, d] = u_loadings[j, d];
        }
        // Pad if d_annotator > D
        for (d in (D+1):d_annotator) {
            annotator_embeddings[j, d] = 0;
        }
    }
    
    // Compute observed log-likelihoods (same as in model block)
    for (n in 1:N_ratings) {
        int i = rating_attributes[n];
        int j = rating_annotators[n];
        int k = rating_items[n];
        int c = rating_values[n];
        int ij_idx = (i-1)*J + j;
        
        real base_score = base_scores[ij_idx, k];
        real upper_threshold = rating_thresholds[ij_idx][c+1];
        real lower_threshold = rating_thresholds[ij_idx][c];
        
        // Compute P(category c | base_score) using conditional distribution (same as model block)
        real mean_std = base_score / total_std[ij_idx];
        real cond_std = sigma_measurement / total_std[ij_idx];
        
        real upper_prob, lower_prob;
        
        if (upper_threshold == positive_infinity()) {
            upper_prob = 1.0;
        } else {
            upper_prob = Phi((upper_threshold - mean_std) / cond_std);
        }
        
        if (lower_threshold == negative_infinity()) {
            lower_prob = 0.0;
        } else {
            lower_prob = Phi((lower_threshold - mean_std) / cond_std);
        }
        
        real bin_prob = upper_prob - lower_prob;
        log_lik_ratings_obs += log(bin_prob + 1e-10);

        // ===== DEBUG: print the first few observed rating terms per draw =====
        if (DEBUG_PRINT == 1 && n <= 10) {
            print("[DEBUG RATING] n=", n,
                  " i=", i, " j=", j, " k=", k, " ij_idx=", ij_idx,
                  " base_score=", base_score,
                  " lower_th=", lower_threshold, " upper_th=", upper_threshold,
                  " lower_prob=", lower_prob, " upper_prob=", upper_prob,
                  " bin_prob=", bin_prob,
                  " value=", c);
        }
    }
    
    for (n in 1:N_pairwise_rankings) {
        int i = pairwise_ranking_attributes[n];
        int j = pairwise_ranking_annotators[n];
        int ij_idx = (i-1)*J + j;
        
        int item1 = pairwise_ranking_items[n, 1];
        int item2 = pairwise_ranking_items[n, 2];
        int order = pairwise_ranking_orders[n];
        real score1 = base_scores[ij_idx, item1] / temperature;
        real score2 = base_scores[ij_idx, item2] / temperature;
        
        if (order == 1) {  // item1 > item2
            log_lik_pairwise_obs += log_inv_logit(score1 - score2);
        } else {  // item2 > item1
            log_lik_pairwise_obs += log_inv_logit(score2 - score1);
        }

        // ===== DEBUG: print the first few observed pairwise terms per draw =====
        if (DEBUG_PRINT == 1 && n <= 10) {
            real logit12 = score1 - score2;
            real p12 = inv_logit(logit12);
            print("[DEBUG PAIR] n=", n,
                  " i=", i, " j=", j, " ij_idx=", ij_idx,
                  " item1=", item1, " item2=", item2, " order=", order,
                  " score1_T=", score1, " score2_T=", score2,
                  " logit=", logit12, " p12=", p12);
        }
    }
    
    total_log_lik = log_lik_ratings_obs + log_lik_pairwise_obs;

    // ===== DEBUG: summary stats per draw =====
    if (DEBUG_PRINT == 1) {
        // Print brief summary of base_scores for a couple ij indices
        int max_ij_print = (I*J < 2) ? I*J : 2;
        for (m in 1:max_ij_print) {
            real min_bs = positive_infinity();
            real max_bs = negative_infinity();
            real sum_bs = 0;
            for (kk in 1:K) {
                real v = base_scores[m, kk];
                sum_bs += v;
                if (v < min_bs) min_bs = v;
                if (v > max_bs) max_bs = v;
            }
            print("[DEBUG SUMMARY] ij=", m,
                  " base_scores mean=", sum_bs / K,
                  " min=", min_bs, " max=", max_bs);
        }
        print("[DEBUG HYPER] sigma_measurement=", sigma_measurement,
              " factor_decay=", factor_decay,
              " T_weights=", T_weights,
              " temperature=", temperature,
              " N_ratings=", N_ratings,
              " N_pairwise=", N_pairwise_rankings);
    }
    
    // ===== POSTERIOR PREDICTIVE SAMPLING FOR MISSING VARIABLES =====
    
    // 1. Sample missing ratings and compute predicted distributions
    for (n in 1:N_missing_ratings) {
        int i = missing_rating_attributes[n];
        int j = missing_rating_annotators[n];
        int k = missing_rating_items[n];
        int ij_idx = (i-1)*J + j;
        
        // Base score: z_ijk = sum_d v_id * u_jd * e_kd * T_d - score_center
        real base_score = base_scores[ij_idx, k];
        
        // Compute predicted probability distribution over Likert scale
        // Conditional distribution: Z + epsilon | v, u, e ~ N(Z, sigma_measurement^2)
        // Standardize to match binning space: mean = base_score / total_std, std = sigma_measurement / total_std
        real mean_std = base_score / total_std[ij_idx];
        real cond_std = sigma_measurement / total_std[ij_idx];
        
        for (c in 1:C) {
            real upper_threshold = rating_thresholds[ij_idx][c+1];
            real lower_threshold = rating_thresholds[ij_idx][c];
            
            real upper_prob, lower_prob;
            
            if (upper_threshold == positive_infinity()) {
                upper_prob = 1.0;
            } else {
                upper_prob = Phi((upper_threshold - mean_std) / cond_std);
            }
            
            if (lower_threshold == negative_infinity()) {
                lower_prob = 0.0;
            } else {
                lower_prob = Phi((lower_threshold - mean_std) / cond_std);
            }
            
            missing_rating_probs[n][c] = upper_prob - lower_prob;
        }
        
        // Sample a rating from the predicted distribution
        // Generate noisy score and standardize for binning using total_std
        real noisy_score = base_score + normal_rng(0, sigma_measurement);
        real standardized_score = noisy_score / total_std[ij_idx];
        int rating = 1;
        for (c in 1:C) {
            if (standardized_score <= rating_thresholds[ij_idx][c+1]) {
                rating = c;
                break;
            }
        }
        missing_rating_predictions[n] = rating;

        // ===== DEBUG: print first few missing rating predictives per draw =====
        if (DEBUG_PRINT == 1 && n <= 10) {
            print("[DEBUG MRATING] n=", n,
                  " i=", i, " j=", j, " k=", k, " ij_idx=", ij_idx,
                  " base_score=", base_score,
                  " standardized_score=", standardized_score,
                  " thresholds=", rating_thresholds[ij_idx],
                  " probs=", missing_rating_probs[n],
                  " sampled=", rating);
        }
    }
    
    // 2. Sample missing pairwise rankings and compute predicted logits
    for (n in 1:N_missing_pairwise_rankings) {
        int i = missing_pairwise_ranking_attributes[n];
        int j = missing_pairwise_ranking_annotators[n];
        int ij_idx = (i-1)*J + j;
        
        int item1 = missing_pairwise_ranking_items[n, 1];
        int item2 = missing_pairwise_ranking_items[n, 2];
        
        // Get base scores and apply temperature scaling: z_ijk/T
        real score1 = base_scores[ij_idx, item1] / temperature;
        real score2 = base_scores[ij_idx, item2] / temperature;
        
        // Compute predicted Bradley-Terry logit: logit(P(item1 > item2))
        missing_pairwise_logits[n] = score1 - score2;
        
        // Sample ranking using Gumbel noise (same as in data generation)
        real gumbel1 = -log(-log(uniform_rng(0, 1)));  // Gumbel noise G1
        real gumbel2 = -log(-log(uniform_rng(0, 1)));  // Gumbel noise G2
        
        real utility1 = score1 + gumbel1;  // U1 = z_ijk1/T + G1
        real utility2 = score2 + gumbel2;  // U2 = z_ijk2/T + G2
        
        // Determine ranking order: 1 if item1 > item2, 2 if item2 > item1
        missing_pairwise_ranking_predictions[n] = (utility1 > utility2) ? 1 : 2;

        // ===== DEBUG: print first few missing pairwise predictives per draw =====
        if (DEBUG_PRINT == 1 && n <= 10) {
            real logit12 = score1 - score2;
            real p12 = inv_logit(logit12);
            print("[DEBUG MPAIR] n=", n,
                  " i=", i, " j=", j, " ij_idx=", ij_idx,
                  " item1=", item1, " item2=", item2,
                  " score1_T=", score1, " score2_T=", score2,
                  " logit=", logit12, " p12=", p12,
                  " sampled_order=", missing_pairwise_ranking_predictions[n]);
        }
    }
}

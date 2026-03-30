/*
 * Domain model for mixed annotation inference using MCMC
 * Learns embeddings and preferences from ratings, comparisons, and rankings
 */

data {
    // Dimensions
    int<lower=1> K;  // number of items in this instance
    int<lower=1> I;  // number of criteria (attributes)
    int<lower=1> J;  // number of annotators  
    int<lower=1> D;  // embedding dimension
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
    real<lower=0> sigma_annotator;
    real<lower=0> sigma_measurement;
    real<lower=0> alpha_dirichlet;
    real<lower=0> temperature;
    
}

transformed data {
    // Debug printing flag (0 = off, 1 = on). Toggle here.
    int DEBUG_PRINT;
    DEBUG_PRINT = 0;
}

parameters {
    // Item embeddings: e_k ~ N(0, I)
    matrix[K, D] embeddings;
    
    // Mean preferences per criteria: v_i ~ N(0, I)
    matrix[I, D] mean_preferences;
    
    // Annotator-specific preferences: v_ij ~ N(v_i, σ_a²I)
    matrix[I*J, D] annotator_preferences;
    
    // Rating probabilities: p_ij ~ Dir(α/C, ..., α/C)
    array[I*J] simplex[C] rating_probs;
}

transformed parameters {
    // Base utility scores: z_ijk = v_ij · e_k
    matrix[I*J, K] base_scores;
    
    // Rating thresholds: q_ij = Φ⁻¹(cumsum(p_ij))
    array[I*J] vector[C+1] rating_thresholds;
    
    // Total standard deviation per annotator-criterion for rating binning
    // Binning is based on distribution of v*e + epsilon with std = sqrt(||v||^2 + sigma_measurement^2)
    array[I*J] real total_std;  // sqrt(||v_ij||^2 + sigma_measurement^2)
    
    // Compute base scores: z_ij_k = v_ij · e_k  
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], embeddings[k]);
            }
        }
    }
    
    // Convert probabilities to thresholds using inverse normal CDF
    for (ij in 1:(I*J)) {
        rating_thresholds[ij][1] = negative_infinity();  // -∞ for category 1
        
        // Convert cumulative probabilities to thresholds
        for (c in 2:C) {
            real cum_prob = sum(rating_probs[ij][1:(c-1)]);
            cum_prob = fmin(fmax(cum_prob, 1e-6), 1.0 - 1e-6);
            rating_thresholds[ij][c] = inv_Phi(cum_prob);
        }
        
        rating_thresholds[ij][C+1] = positive_infinity();  // +∞ for category C
    }
    
    // Compute total_std per ij for rating binning
    // Binning is based on distribution of v*e + epsilon with std = sqrt(||v||^2 + sigma_measurement^2)
    for (i in 1:I) {
        for (j in 1:J) {
            int ij_idx = (i-1)*J + j;
            real pref_norm_sq = dot_self(annotator_preferences[ij_idx]);
            total_std[ij_idx] = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
        }
    }

    // ===== DEBUG: print a few rating_probs and thresholds per draw =====
    if (DEBUG_PRINT == 1) {
        int max_ij_print = (I*J < 3) ? I*J : 3;
        for (ij in 1:max_ij_print) {
            print("[DEBUG TP] ij=", ij, " rating_probs=", rating_probs[ij]);
            print("[DEBUG TP] ij=", ij, " thresholds=", rating_thresholds[ij]);
        }
    }
}

model {
    // ===== PRIORS =====
    
    // Item embeddings: e_k ~ N(0, I)
    for (k in 1:K) {
        embeddings[k] ~ normal(0, 1);
    }
    
    // Mean preferences: v_i ~ N(0, I) 
    for (i in 1:I) {
        mean_preferences[i] ~ normal(0, 1);
    }

    // Annotator preferences: v_ij ~ N(v_i, σ_a²I)
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            annotator_preferences[idx] ~ normal(mean_preferences[i], sigma_annotator);
        }
    }
    
    // Rating probabilities: p_ij ~ Dir(α/C, ..., α/C)
    for (ij in 1:(I*J)) {
        rating_probs[ij] ~ dirichlet(rep_vector(alpha_dirichlet / C, C));
    }
    
    // ===== LIKELIHOODS =====
    
    // 1. RATING LIKELIHOOD
    // Binning is based on distribution of v*e + epsilon with std = sqrt(||v||^2 + sigma_measurement^2)
    // Conditional distribution: v*e + epsilon | v, e ~ N(v*e, sigma_measurement^2)
    // Standardize to match binning space: mean = (v*e) / total_std, std = sigma_measurement / total_std
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
        // Conditional: v*e + epsilon | v, e ~ N(v*e, sigma_measurement^2)
        // Standardized: mean = base_score / total_std, std = sigma_measurement / total_std
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
        int order = pairwise_ranking_orders[n];  // 1 if item1 > item2, 2 if item2 > item1 TODO: double check this convention.
        
        // Base scores scaled by temperature: z_ijk/T
        real score1 = base_scores[ij_idx, item1] / temperature;
        real score2 = base_scores[ij_idx, item2] / temperature;
        
        // Bradley-Terry likelihood: P(item1 > item2) = exp(score1) / (exp(score1) + exp(score2))
        // This is equivalent to log_inv_logit(score1 - score2)
        if (order == 1) {  // item1 > item2
            target += log_inv_logit(score1 - score2);
        } else if (order==2) {  // item2 > item1
            target += log_inv_logit(score2 - score1);
        }
        else {
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
              " sigma_annotator=", sigma_annotator,
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
        
        // Base score: z_ijk = v_ij · e_k
        real base_score = base_scores[ij_idx, k];
        
        // Compute predicted probability distribution over Likert scale
        // Conditional distribution: v*e + epsilon | v, e ~ N(v*e, sigma_measurement^2)
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
        real gumbel1 = -log(-log(uniform_rng(0, 1)));  // Gumbel noise G1, standard normal since we already normalized.
        real gumbel2 = -log(-log(uniform_rng(0, 1)));  // Gumbel noise G2
        
        real utility1 = score1 + gumbel1;  // U1 = z_ijk1/T + G1
        real utility2 = score2 + gumbel2;  // U2 = z_ijk2/T + G2
        
        // Determine ranking order: 1 if item1 > item2, 2 if item2 > item1
        missing_pairwise_ranking_predictions[n] = (utility1 > utility2) ? 1 : 2;

        // ===== DEBUG: print first few missing pairwise predictives per draw =====
        if (n <= 10) {
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
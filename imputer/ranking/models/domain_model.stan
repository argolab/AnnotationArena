/*
 * Domain model for mixed annotation inference using MCMC
 * Learns embeddings and preferences from ratings, comparisons, and rankings
 */

data {
    // Dimensions
    int<lower=1> K;  // number of items
    int<lower=1> I;  // number of attributes
    int<lower=1> J;  // number of annotators  
    int<lower=1> D;  // embedding dimension
    int<lower=1> C;  // number of rating categories
    int<lower=2> ranking_size;  // size of ranking sets
    
    // Observed ratings
    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=I> rating_attributes;
    array[N_ratings] int<lower=1, upper=J> rating_annotators;
    array[N_ratings] int<lower=1, upper=K> rating_items;
    array[N_ratings] int<lower=1, upper=C> rating_values;
    
    
    // Observed rankings
    int<lower=0> N_rankings;
    array[N_rankings] int<lower=1, upper=I> ranking_attributes;
    array[N_rankings] int<lower=1, upper=J> ranking_annotators;
    array[N_rankings, ranking_size] int<lower=1, upper=K> ranking_items;
    array[N_rankings, ranking_size] int<lower=1, upper=ranking_size> ranking_orders;
    
    // Hyperparameters
    real<lower=0> sigma_annotator;
    real<lower=0> sigma_measurement;
    real<lower=0> alpha_dirichlet;
    real<lower=0> temperature;
    
    // Prior scales
    // Use this as a paramters?
    real<lower=0> sigma_embedding_prior;
    real<lower=0> sigma_preference_prior;
}

parameters {
    // Latent embeddings
    matrix[K, D] embeddings;
    
    // Mean preferences per attribute
    matrix[I, D] mean_preferences;
    
    // Annotator-specific preferences
    matrix[I*J, D] annotator_preferences;
    
    // Rating thresholds (per annotator-attribute pair) as cumulative inverse CDF values
    // TODO - Try with raw probabilities directly
    array[I*J] ordered[C-1] rating_thresholds_raw;
}

transformed parameters {
    // Base utility scores
    matrix[I*J, K] base_scores;
    
    // Rating thresholds with -inf and +inf boundaries
    array[I*J] vector[C+1] rating_thresholds;
    
    // Compute base scores: z_ij_k = v_ij · e_k
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], embeddings[k]);
            }
        }
    }
    
    // Construct full threshold vector with boundaries
    for (ij in 1:(I*J)) {
        rating_thresholds[ij][1] = negative_infinity();  // -∞ for category 1
        for (c in 2:C) {
            rating_thresholds[ij][c] = rating_thresholds_raw[ij][c-1];
        }
        rating_thresholds[ij][C+1] = positive_infinity();  // +∞ for category C
    }
}

model {
    // ===== PRIORS =====
    
    // Embeddings: e_k ~ N(0, σ_e²I)
    for (k in 1:K) {
        embeddings[k] ~ normal(0, sigma_embedding_prior);
    }
    
    // Mean preferences: v_i ~ N(0, σ_v²I) 
    for (i in 1:I) {
        mean_preferences[i] ~ normal(0, sigma_preference_prior);
    }
    
    // Annotator preferences: v_ij ~ N(v_i, σ_a²I)
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            annotator_preferences[idx] ~ normal(mean_preferences[i], sigma_annotator);
        }
    }
    
    // Rating threshold priors: ordered thresholds
    // TODO: Why 2?
    for (ij in 1:(I*J)) {
        for (c in 1:(C-1)) {
            rating_thresholds_raw[ij][c] ~ normal(0, 2.0);  // Diffuse prior on thresholds
        }
    }
    
    // ===== LIKELIHOODS =====
    
    // 1. RATING LIKELIHOOD
    for (n in 1:N_ratings) {
        int i = rating_attributes[n];
        int j = rating_annotators[n]; 
        int k = rating_items[n];
        int c = rating_values[n];
        int ij_idx = (i-1)*J + j;
        
        // Base score: z_ijk = v_ij · e_k
        real base_score = base_scores[ij_idx, k-1];
        
        // Rating likelihood: P(rating = c) = Φ((Q_c - z)/σ_m) - Φ((Q_{c-1} - z)/σ_m)
        // where Q_c are the thresholds and ε ~ N(0, σ_m²)
        real upper_threshold = rating_thresholds[ij_idx][c+1];
        real lower_threshold = rating_thresholds[ij_idx][c];
        
        real upper_prob, lower_prob;
        
        if (upper_threshold == positive_infinity()) {
            upper_prob = 1.0;
        } else {
            upper_prob = Phi((upper_threshold - base_score) / sigma_measurement);
        }
        
        if (lower_threshold == negative_infinity()) {
            lower_prob = 0.0;
        } else {
            lower_prob = Phi((lower_threshold - base_score) / sigma_measurement);
        }
        
        real bin_prob = upper_prob - lower_prob;
        
        // Numerical stability
        if (bin_prob > 1e-8) {
            target += log(bin_prob);
        } else {
            target += log(1e-8);
        }
    }
    
    // 2. RANKING LIKELIHOOD (Plackett-Luce with Gumbel noise)
    for (n in 1:N_rankings) {
        int i = ranking_attributes[n];
        int j = ranking_annotators[n];
        int ij_idx = (i-1)*J + j;
        
        // Extract base scores for ranked items
        vector[ranking_size] item_scores;
        for (r in 1:ranking_size) {
            int k = ranking_items[n, r];
            item_scores[r] = base_scores[ij_idx, k-1] / temperature;
        }
        
        // Plackett-Luce likelihood - fixed version
        // Convert ranking order to proper Plackett-Luce format
        // TODO: What is it doing?
        array[ranking_size] int item_order;
        for (pos in 1:ranking_size) {
            item_order[ranking_orders[n, pos]] = pos;  // Map item rank to position
        }
        
        // Now compute Plackett-Luce likelihood
        for (pos in 1:ranking_size) {
            // Find which item is in position pos
            int chosen_item = 0;
            for (r in 1:ranking_size) {
                if (item_order[r] == pos) {
                    chosen_item = r;
                    break;
                }
            }
            
            if (chosen_item > 0) {
                // Compute log probability
                real chosen_score = item_scores[chosen_item];
                vector[ranking_size - pos + 1] remaining_scores;
                int remaining_count = 0;
                
                for (r in 1:ranking_size) {
                    if (item_order[r] >= pos) {  // Item still available
                        remaining_count += 1;
                        remaining_scores[remaining_count] = item_scores[r];
                    }
                }
                
                if (remaining_count > 0) {
                    real log_sum_exp_remaining = log_sum_exp(remaining_scores[1:remaining_count]);
                    target += chosen_score - log_sum_exp_remaining;
                }
            }
        }
    }
}

generated quantities {
    // Log-likelihood components for evaluation
    real log_lik_ratings = 0;
    real log_lik_rankings = 0;
    real total_log_lik = 0;
    
    // Compute log-likelihoods (same as in model block)
    for (n in 1:N_ratings) {
        int i = rating_attributes[n];
        int j = rating_annotators[n];
        int k = rating_items[n];
        int c = rating_values[n];
        int ij_idx = (i-1)*J + j;
        
        real base_score = base_scores[ij_idx, k-1];
        real upper_threshold = rating_thresholds[ij_idx][c+1];
        real lower_threshold = rating_thresholds[ij_idx][c];
        
        real upper_prob, lower_prob;
        
        if (upper_threshold == positive_infinity()) {
            upper_prob = 1.0;
        } else {
            upper_prob = Phi((upper_threshold - base_score) / sigma_measurement);
        }
        
        if (lower_threshold == negative_infinity()) {
            lower_prob = 0.0;
        } else {
            lower_prob = Phi((lower_threshold - base_score) / sigma_measurement);
        }
        
        real bin_prob = upper_prob - lower_prob;
        log_lik_ratings += log(bin_prob + 1e-10);
    }
    
    for (n in 1:N_rankings) {
        int i = ranking_attributes[n];
        int j = ranking_annotators[n];
        int ij_idx = (i-1)*J + j;
        
        vector[ranking_size] item_scores;
        for (r in 1:ranking_size) {
            int k = ranking_items[n, r];
            item_scores[r] = base_scores[ij_idx, k-1] / temperature;
        }
        
        // Same fixed logic as in model block
        array[ranking_size] int item_order;
        for (pos in 1:ranking_size) {
            item_order[ranking_orders[n, pos]] = pos;
        }
        
        for (pos in 1:ranking_size) {
            int chosen_item = 0;
            for (r in 1:ranking_size) {
                if (item_order[r] == pos) {
                    chosen_item = r;
                    break;
                }
            }
            
            if (chosen_item > 0) {
                real chosen_score = item_scores[chosen_item];
                vector[ranking_size - pos + 1] remaining_scores;
                int remaining_count = 0;
                
                for (r in 1:ranking_size) {
                    if (item_order[r] >= pos) {
                        remaining_count += 1;
                        remaining_scores[remaining_count] = item_scores[r];
                    }
                }
                
                if (remaining_count > 0) {
                    real log_sum_exp_remaining = log_sum_exp(remaining_scores[1:remaining_count]);
                    log_lik_rankings += chosen_score - log_sum_exp_remaining;
                }
            }
        }
    }
    
    total_log_lik = log_lik_ratings + log_lik_rankings;
}
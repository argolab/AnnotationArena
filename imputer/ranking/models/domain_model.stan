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
    
    // Observed comparisons
    int<lower=0> N_comparisons;
    array[N_comparisons] int<lower=1, upper=I> comparison_attributes;
    array[N_comparisons] int<lower=1, upper=J> comparison_annotators;
    array[N_comparisons] int<lower=1, upper=K> comparison_items_a;
    array[N_comparisons] int<lower=1, upper=K> comparison_items_b;
    array[N_comparisons] int<lower=0, upper=1> comparison_results;
    
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
    
    // Rating thresholds (per annotator-attribute pair)
    array[I*J] simplex[C] rating_probs;
}

transformed parameters {
    // Base utility scores
    matrix[I*J, K] base_scores;
    
    // Rating thresholds
    array[I*J] vector[C] rating_thresholds;
    
    // Compute base scores: z_ij_k = v_ij · e_k
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], embeddings[k]);
            }
        }
    }
    
    // Convert rating probabilities to cumulative thresholds
    for (ij in 1:(I*J)) {
        rating_thresholds[ij] = cumulative_sum(rating_probs[ij]);
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
    
    // Rating threshold priors: p_ij ~ Dir(α/C, ..., α/C)
    for (ij in 1:(I*J)) {
        rating_probs[ij] ~ dirichlet(rep_vector(alpha_dirichlet/C, C));
    }
    
    // ===== LIKELIHOODS =====
    
    // 1. RATING LIKELIHOOD
    for (n in 1:N_ratings) {
        int i = rating_attributes[n];
        int j = rating_annotators[n]; 
        int k = rating_items[n];
        int c = rating_values[n];
        int ij_idx = (i-1)*J + j;
        
        // Noisy score: s = z_ijk + ε, ε ~ N(0, σ_m²)
        real base_score = base_scores[ij_idx, k];
        real pref_norm = sqrt(dot_self(annotator_preferences[ij_idx]));
        real total_std = sqrt(pref_norm^2 + sigma_measurement^2);
        real standardized_score = base_score / total_std;
        real cdf_val = Phi(standardized_score);
        
        // Binning likelihood with numerical stability
        real bin_prob;
        if (c == 1) {
            bin_prob = rating_thresholds[ij_idx][c] - cdf_val;
        } else if (c == C) {
            bin_prob = cdf_val - rating_thresholds[ij_idx][c-1];
        } else {
            bin_prob = rating_thresholds[ij_idx][c] - rating_thresholds[ij_idx][c-1];
        }
        
        // Ensure positive probability
        if (bin_prob > 1e-8) {
            target += log(bin_prob);
        } else {
            target += log(1e-8);  // Minimum probability to avoid -inf
        }
    }
    
    // 2. COMPARISON LIKELIHOOD
    for (n in 1:N_comparisons) {
        int i = comparison_attributes[n];
        int j = comparison_annotators[n];
        int k_a = comparison_items_a[n];
        int k_b = comparison_items_b[n];
        int result = comparison_results[n];
        int ij_idx = (i-1)*J + j;
        
        // Score difference with noise
        real score_a = base_scores[ij_idx, k_a];
        real score_b = base_scores[ij_idx, k_b];
        real score_diff = score_a - score_b;
        real noise_std = sigma_measurement * sqrt(2);  // Independent noise on both scores
        
        // Probit likelihood: P(a > b) = Φ((z_a - z_b) / σ_noise)
        real prob = Phi(score_diff / noise_std);
        
        // Ensure probability is in valid range [1e-8, 1-1e-8]
        prob = fmax(1e-8, fmin(1 - 1e-8, prob));
        
        if (result == 1) {
            target += log(prob);
        } else {
            target += log(1 - prob);
        }
    }
    
    // 3. RANKING LIKELIHOOD (Plackett-Luce with Gumbel noise)
    for (n in 1:N_rankings) {
        int i = ranking_attributes[n];
        int j = ranking_annotators[n];
        int ij_idx = (i-1)*J + j;
        
        // Extract base scores for ranked items
        vector[ranking_size] item_scores;
        for (r in 1:ranking_size) {
            int k = ranking_items[n, r];
            item_scores[r] = base_scores[ij_idx, k] / temperature;
        }
        
        // Plackett-Luce likelihood - fixed version
        // Convert ranking order to proper Plackett-Luce format
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
    real log_lik_comparisons = 0;
    real log_lik_rankings = 0;
    real total_log_lik = 0;
    
    // Compute log-likelihoods (same as in model block)
    for (n in 1:N_ratings) {
        int i = rating_attributes[n];
        int j = rating_annotators[n];
        int k = rating_items[n];
        int c = rating_values[n];
        int ij_idx = (i-1)*J + j;
        
        real base_score = base_scores[ij_idx, k];
        real pref_norm = sqrt(dot_self(annotator_preferences[ij_idx]));
        real total_std = sqrt(pref_norm^2 + sigma_measurement^2);
        real standardized_score = base_score / total_std;
        real cdf_val = Phi(standardized_score);
        
        if (c == 1) {
            log_lik_ratings += log(rating_thresholds[ij_idx][c] - cdf_val + 1e-10);
        } else if (c == C) {
            log_lik_ratings += log(cdf_val - rating_thresholds[ij_idx][c-1] + 1e-10);
        } else {
            log_lik_ratings += log(rating_thresholds[ij_idx][c] - rating_thresholds[ij_idx][c-1] + 1e-10);
        }
    }
    
    for (n in 1:N_comparisons) {
        int i = comparison_attributes[n];
        int j = comparison_annotators[n];
        int k_a = comparison_items_a[n];
        int k_b = comparison_items_b[n];
        int result = comparison_results[n];
        int ij_idx = (i-1)*J + j;
        
        real score_a = base_scores[ij_idx, k_a];
        real score_b = base_scores[ij_idx, k_b];
        real score_diff = score_a - score_b;
        real noise_std = sigma_measurement * sqrt(2);
        real prob = Phi(score_diff / noise_std);
        
        if (result == 1) {
            log_lik_comparisons += log(prob + 1e-10);
        } else {
            log_lik_comparisons += log(1 - prob + 1e-10);
        }
    }
    
    for (n in 1:N_rankings) {
        int i = ranking_attributes[n];
        int j = ranking_annotators[n];
        int ij_idx = (i-1)*J + j;
        
        vector[ranking_size] item_scores;
        for (r in 1:ranking_size) {
            int k = ranking_items[n, r];
            item_scores[r] = base_scores[ij_idx, k] / temperature;
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
    
    total_log_lik = log_lik_ratings + log_lik_comparisons + log_lik_rankings;
}
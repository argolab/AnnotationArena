data {
    // Dimensions
    int<lower=1> K;  // number of items
    int<lower=1> I;  // number of attributes
    int<lower=1> J;  // number of annotators  
    int<lower=1> D;  // embedding dimension
    int<lower=1> C;  // number of rating categories
    int<lower=2> ranking_size;  // size of ranking sets
    int<lower=1> rankings_per_annotator_attribute;  // number of rankings per annotator-attribute pair
    
    // Observation fractions
    real<lower=0, upper=1> observed_rating_fraction;
    real<lower=0, upper=1> observed_ranking_fraction;
    
    // Hyperparameters
    real<lower=0> sigma_annotator;
    real<lower=0> sigma_measurement;
    real<lower=0> alpha_dirichlet;
    real<lower=0> temperature;
}

generated quantities {

    // ===== COMPONENTS =====
    matrix[K, D] embeddings;
    matrix[I, D] mean_preferences;
    matrix[I*J, D] annotator_preferences;
    array[I*J] simplex[C] rating_probs;
    array[I*J] vector[C] rating_thresholds;
    matrix[I*J, K] base_scores;
    
    // ===== COMPLETE ANNOTATION SPACE =====
    
    // 1. ALL UNARY RATINGS: I*J*K total
    matrix[I*J, K] all_rating_values;           // All possible ratings
    array[I*J, K] int all_rating_observed;      // 1 = observed, 0 = missing
    
    // 2. Multiple rankings per annotator-attribute pair
    array[I*J*rankings_per_annotator_attribute, ranking_size] int all_ranking_items;   // Items in each ranking set
    array[I*J*rankings_per_annotator_attribute, ranking_size] int all_ranking_orders;  // Ranking orders  
    array[I*J*rankings_per_annotator_attribute] int all_ranking_observed;              // 1 = observed, 0 = missing
    
    // ===== OBSERVED/MISSING COUNTS =====
    int num_observed_ratings;
    int num_missing_ratings; 
    int num_observed_rankings;
    int num_missing_rankings;
    
    // Generate embeddings: eₖ ~ N(0,I)
    for (k in 1:K) {
        for (d in 1:D) {
            embeddings[k, d] = normal_rng(0, 1);
        }
    }
    
    // Generate mean preferences: vᵢ ~ N(0,I)
    for (i in 1:I) {
        for (d in 1:D) {
            mean_preferences[i, d] = normal_rng(0, 1);
        }
    }
    
    // Generate annotator preferences: vᵢⱼ ~ N(vᵢ, σ²I)
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (d in 1:D) {
                annotator_preferences[idx, d] = normal_rng(mean_preferences[i, d], sigma_annotator);
            }
        }
    }
    
    // Generate rating thresholds: pᵢⱼ ~ Dir(α/C, ..., α/C)
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            rating_probs[idx] = dirichlet_rng(rep_vector(alpha_dirichlet/C, C));
            rating_thresholds[idx] = cumulative_sum(rating_probs[idx]);
        }
    }
    
    // Compute base scores: zᵢⱼₖ = vᵢⱼ · eₖ
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], embeddings[k]);
            }
        }
    }
    
    // ===== Generate all unary ratings =====
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                // Generate rating with noise and binning
                real noisy_score = base_scores[idx, k] + normal_rng(0, sigma_measurement);
                real pref_norm = sqrt(dot_self(annotator_preferences[idx]));
                real total_std = sqrt(pref_norm^2 + sigma_measurement^2);
                real standardized_score = noisy_score / total_std;
                real cdf_val = Phi(standardized_score);
                
                // Bin into rating categories
                int rating = 1;
                for (c in 1:C) {
                    if (cdf_val <= rating_thresholds[idx][c]) {
                        rating = c;
                        break;
                    }
                }
                all_rating_values[idx, k] = rating;
                
                // Mark all as generated (split happens in Python)
                all_rating_observed[idx, k] = 1;
            }
        }
    }
    
    // ===== Generate sample rankings (adjacent items by utility) =====
    for (i in 1:I) {
        for (j in 1:J) {
            int ij_idx = (i-1)*J + j;
            
            // Sort items by base utility
            array[K] int sorted_items;
            array[K] real item_utilities;
            for (k in 1:K) {
                sorted_items[k] = k;
                item_utilities[k] = base_scores[ij_idx, k];
            }
            
            // Bubble sort items by utility (descending)
            for (i_sort in 1:(K-1)) {
                for (j_sort in 1:(K-i_sort)) {
                    if (item_utilities[j_sort] < item_utilities[j_sort+1]) {
                        real temp_util = item_utilities[j_sort];
                        item_utilities[j_sort] = item_utilities[j_sort+1];
                        item_utilities[j_sort+1] = temp_util;
                        
                        int temp_item = sorted_items[j_sort];
                        sorted_items[j_sort] = sorted_items[j_sort+1];
                        sorted_items[j_sort+1] = temp_item;
                    }
                }
            }
            
            // Generate sliding window rankings
            for (ranking_idx in 1:rankings_per_annotator_attribute) {
                int global_ranking_idx = (ij_idx-1)*rankings_per_annotator_attribute + ranking_idx;
                int start_pos = ranking_idx;
                
                // Select adjacent items
                for (r in 1:ranking_size) {
                    int pos = start_pos + r - 1;
                    if (pos <= K) {
                        all_ranking_items[global_ranking_idx, r] = sorted_items[pos];
                    } else {
                        // Wrap around to beginning: pos - K gives us how far past the end we are
                        int wrapped_pos = pos - K;
                        all_ranking_items[global_ranking_idx, r] = sorted_items[wrapped_pos];
                    }
                }
                
                // Generate ranking using Gumbel noise
                array[ranking_size] real gumbel_scores;
                for (r in 1:ranking_size) {
                    int k = all_ranking_items[global_ranking_idx, r];
                    real base_score = base_scores[ij_idx, k];
                    real u = uniform_rng(0, 1);
                    real gumbel = -log(-log(u));
                    gumbel_scores[r] = base_score / temperature + gumbel;
                }
                
                // Sort to get ranking order
                array[ranking_size] int sorted_indices;
                for (r in 1:ranking_size) sorted_indices[r] = r;
                
                // Bubble sort
                for (i_sort in 1:(ranking_size-1)) {
                    for (j_sort in 1:(ranking_size-i_sort)) {
                        if (gumbel_scores[sorted_indices[j_sort]] < gumbel_scores[sorted_indices[j_sort+1]]) {
                            int temp = sorted_indices[j_sort];
                            sorted_indices[j_sort] = sorted_indices[j_sort+1];
                            sorted_indices[j_sort+1] = temp;
                        }
                    }
                }
                
                // Store ranking order
                for (r in 1:ranking_size) {
                    all_ranking_orders[global_ranking_idx, r] = sorted_indices[r];
                }
                
                // Mark all as generated (split happens in Python)
                all_ranking_observed[global_ranking_idx] = 1;
            }
        }
    }
    
    // ===== Compute observed/missing counts =====
    num_observed_ratings = 0;
    num_missing_ratings = 0;
    for (i in 1:(I*J)) {
        for (k in 1:K) {
            if (all_rating_observed[i, k] == 1) {
                num_observed_ratings += 1;
            } else {
                num_missing_ratings += 1;
            }
        }
    }
    
    num_observed_rankings = 0;
    num_missing_rankings = 0;
    for (global_ranking_idx in 1:(I*J*rankings_per_annotator_attribute)) {
        if (all_ranking_observed[global_ranking_idx] == 1) {
            num_observed_rankings += 1;
        } else {
            num_missing_rankings += 1;
        }
    }
}
data {
    // Dimensions
    int<lower=1> K;  // number of items
    int<lower=1> I;  // number of attributes
    int<lower=1> J;  // number of annotators  
    int<lower=1> D;  // embedding dimension
    int<lower=1> C;  // number of rating categories
    
    // Pairwise ranking limits
    int<lower=1> max_pairs_per_tied_group;  // Maximum pairwise comparisons per tied group
    int<lower=2> min_group_size;            // Minimum group size to generate pairs  
    int<lower=2> max_group_size;            // Maximum group size for all pairs
    
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
    
    // ===== RATINGS =====
    matrix[I*J, K] all_rating_values;           // All possible ratings
    array[I*J, K] int all_rating_observed;      // 1 = observed, 0 = missing
    
    // ===== PAIRWISE RANKINGS =====
    // Maximum possible pairwise rankings: I*J*C*K*(K-1)/2 (worst case: all items have same rating)
    int max_pairwise_rankings = I*J*C*K*(K-1)/2;
    array[I*J*C*K*(K-1)/2, 2] int pairwise_ranking_items;   // [item1, item2] pairs
    array[I*J*C*K*(K-1)/2] int pairwise_ranking_orders;     // 1 if item1 > item2, 2 if item2 > item1
    array[I*J*C*K*(K-1)/2] int pairwise_ranking_annotator;  // annotator for this ranking
    array[I*J*C*K*(K-1)/2] int pairwise_ranking_attribute;  // attribute for this ranking
    array[I*J*C*K*(K-1)/2] int pairwise_ranking_tied_rating; // the rating value that tied
    array[I*J*C*K*(K-1)/2] int pairwise_ranking_observed;   // 1 = observed, 0 = missing
    
    // Counts
    int num_ratings;
    int num_pairwise_rankings;
    int num_observed_ratings;
    int num_missing_ratings; 
    int num_observed_pairwise_rankings;
    int num_missing_pairwise_rankings;
    
    // ===== GENERATE EMBEDDINGS AND PREFERENCES =====
    
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
    
    // ===== GENERATE ALL RATINGS =====
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
                all_rating_observed[idx, k] = 1;  // All generated for now
            }
        }
    }
    
    // ===== GENERATE PAIRWISE RANKINGS FROM TIED RATINGS =====
    num_pairwise_rankings = 0;
    
    for (i in 1:I) {
        for (j in 1:J) {
            int ij_idx = (i-1)*J + j;
            
            // For each rating category, find tied items
            for (rating_val in 1:C) {
                array[K] int tied_items;
                int num_tied = 0;
                
                // Find all items with this rating value
                for (k in 1:K) {
                    if (all_rating_values[ij_idx, k] == rating_val) {
                        num_tied += 1;
                        tied_items[num_tied] = k;
                    }
                }
                
                // Generate pairwise rankings with limits
                // TODO
                if (num_tied >= min_group_size) {
                    int max_possible_pairs = num_tied * (num_tied - 1) / 2;
                    int pairs_to_generate;
                    
                    // Determine how many pairs to generate based on group size
                    if (num_tied <= max_group_size) {
                        // Small groups: generate all pairs
                        pairs_to_generate = max_possible_pairs;
                    } else {
                        // Large groups: limit to max_pairs_per_tied_group
                        pairs_to_generate = max_pairs_per_tied_group;
                    }
                    
                    // Generate pairs
                    if (pairs_to_generate >= max_possible_pairs) {
                        // Generate all possible pairs
                        for (idx1 in 1:(num_tied-1)) {
                            for (idx2 in (idx1+1):num_tied) {
                                int item1 = tied_items[idx1];
                                int item2 = tied_items[idx2];
                                
                                // Use Gumbel noise for tie-breaking
                                real u1 = uniform_rng(0, 1);
                                real gumbel1 = -log(-log(u1));
                                real utility1 = base_scores[ij_idx, item1] / temperature + gumbel1;
                                
                                real u2 = uniform_rng(0, 1);
                                real gumbel2 = -log(-log(u2));
                                real utility2 = base_scores[ij_idx, item2] / temperature + gumbel2;
                                
                                // Determine ranking order
                                int order = (utility1 > utility2) ? 1 : 2;
                                
                                // Store pairwise ranking
                                num_pairwise_rankings += 1;
                                pairwise_ranking_items[num_pairwise_rankings, 1] = item1;
                                pairwise_ranking_items[num_pairwise_rankings, 2] = item2;
                                pairwise_ranking_orders[num_pairwise_rankings] = order;
                                pairwise_ranking_annotator[num_pairwise_rankings] = j;
                                pairwise_ranking_attribute[num_pairwise_rankings] = i;
                                pairwise_ranking_tied_rating[num_pairwise_rankings] = rating_val;
                                pairwise_ranking_observed[num_pairwise_rankings] = 1;
                            }
                        }
                    } else {
                        // Generate random sample of pairs
                        for (pair_count in 1:pairs_to_generate) {
                            // Randomly select two different items from tied group
                            real u_idx1 = uniform_rng(0, 1);
                            int idx1 = 1 + to_int(floor(u_idx1 * num_tied));
                            if (idx1 > num_tied) idx1 = num_tied;  // Handle edge case
                            
                            real u_idx2 = uniform_rng(0, 1);
                            int idx2 = 1 + to_int(floor(u_idx2 * num_tied));
                            if (idx2 > num_tied) idx2 = num_tied;  // Handle edge case
                            
                            while (idx2 == idx1) {
                                real u_retry = uniform_rng(0, 1);
                                idx2 = 1 + to_int(floor(u_retry * num_tied));
                                if (idx2 > num_tied) idx2 = num_tied;
                            }
                            
                            int item1 = tied_items[idx1];
                            int item2 = tied_items[idx2];
                            
                            // Use Gumbel noise for tie-breaking
                            real u_gumbel1 = uniform_rng(0, 1);
                            real gumbel1 = -log(-log(u_gumbel1));
                            real utility1 = base_scores[ij_idx, item1] / temperature + gumbel1;
                            
                            real u_gumbel2 = uniform_rng(0, 1);
                            real gumbel2 = -log(-log(u_gumbel2));
                            real utility2 = base_scores[ij_idx, item2] / temperature + gumbel2;
                            
                            // Determine ranking order
                            int order = (utility1 > utility2) ? 1 : 2;
                            
                            // Store pairwise ranking
                            num_pairwise_rankings += 1;
                            pairwise_ranking_items[num_pairwise_rankings, 1] = item1;
                            pairwise_ranking_items[num_pairwise_rankings, 2] = item2;
                            pairwise_ranking_orders[num_pairwise_rankings] = order;
                            pairwise_ranking_annotator[num_pairwise_rankings] = j;
                            pairwise_ranking_attribute[num_pairwise_rankings] = i;
                            pairwise_ranking_tied_rating[num_pairwise_rankings] = rating_val;
                            pairwise_ranking_observed[num_pairwise_rankings] = 1;
                        }
                    }
                }
            }
        }
    }
    
    // ===== COMPUTE COUNTS =====
    num_ratings = I * J * K;
    
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
    
    num_observed_pairwise_rankings = 0;
    num_missing_pairwise_rankings = 0;
    for (ranking_idx in 1:num_pairwise_rankings) {
        if (pairwise_ranking_observed[ranking_idx] == 1) {
            num_observed_pairwise_rankings += 1;
        } else {
            num_missing_pairwise_rankings += 1;
        }
    }
}
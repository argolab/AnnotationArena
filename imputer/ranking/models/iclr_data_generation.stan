data {
    // Dimensions
    int<lower=1> K_train;  // number of items in training instance
    int<lower=1> K_test;   // number of items in test instance
    int<lower=1> I;        // number of criteria (attributes)
    int<lower=1> J;         // number of annotators  
    int<lower=1> D;         // embedding dimension
    int<lower=1> C;         // number of rating categories
    
    // Observation protocol controls
    int<lower=0, upper=1> enable_third_annotator;  // Ablation: enable third annotator for disagreement
    int<lower=0, upper=1> enable_pairwise_rankings;  // Ablation: enable pairwise rankings
    int<lower=0> pairwise_cap_per_item;     // Max comparisons per item within its tied group
    
    // Hyperparameters
    real<lower=0> sigma_annotator;
    real<lower=0> sigma_measurement;
    real<lower=0> alpha_dirichlet;
    real<lower=0> temperature;
}

generated quantities {

    // ===== SHARED COMPONENTS (same for train and test) =====
    matrix[I, D] mean_preferences;              // Global criteria embeddings v_i
    matrix[I*J, D] annotator_preferences;      // All annotator preferences v_ij
    array[I*J] simplex[C] rating_probs;        // Rating probabilities p_ij
    array[I*J] vector[C] rating_thresholds;    // Rating thresholds q_ij = prefix sum of p_ij
    
    // ===== TRAINING INSTANCE =====
    matrix[K_train, D] train_embeddings;       // Training item embeddings e_k_train
    matrix[I*J, K_train] train_base_scores;    // Training base scores z_ijk_train = v_ij * z_k_train
    array[I*J, K_train] int train_rating_values;  // Training rating values x_ijk_train = Bin(z_ijk_train + eps_j [sigma_measurement]) against threshold q_ij
    array[I*J, K_train] int train_rating_observed; // Training observation indicators R_ijk_train, whether the data is missing.
    
    // ===== TEST INSTANCE =====
    matrix[K_test, D] test_embeddings;         // Test item embeddings e_k_test
    matrix[I*J, K_test] test_base_scores;     // Test base scores z_ijk_test = v_ij * e_k_test
    array[I*J, K_test] int test_rating_values;   // Test rating values x_ijk_test = Bin(z_ijk_test + eps_j [sigma_measurement]) against threshold q_ij
    array[I*J, K_test] int test_rating_observed; // Test observation indicators R_ijk_test, whether the data is missing
    
    // ===== PAIRWISE RANKINGS =====
    // Training pairwise rankings
    array[I*J*C*K_train*(K_train-1)/2, 2] int train_pairwise_items;     // [k1, k2] pairs for training pairwise comparisons
    array[I*J*C*K_train*(K_train-1)/2] int train_pairwise_orders;       // 1 if k1 > k2, 2 if k2 > k1 (Bradley-Terry)
    array[I*J*C*K_train*(K_train-1)/2] int train_pairwise_annotator;    // annotator j for each training pairwise comparison
    array[I*J*C*K_train*(K_train-1)/2] int train_pairwise_attribute;    // attribute i for each training pairwise comparison
    array[I*J*C*K_train*(K_train-1)/2] int train_pairwise_tied_rating;  // rating value that tied (x_ijk1 = x_ijk2 = rating_val)
    array[I*J*C*K_train*(K_train-1)/2] int train_pairwise_observed;     // 1 if observed, 0 if missing
    
    // Test pairwise rankings
    array[I*J*C*K_test*(K_test-1)/2, 2] int test_pairwise_items;      // [k1, k2] pairs for test pairwise comparisons
    array[I*J*C*K_test*(K_test-1)/2] int test_pairwise_orders;         // 1 if k1 > k2, 2 if k2 > k1 (Bradley-Terry)
    array[I*J*C*K_test*(K_test-1)/2] int test_pairwise_annotator;      // annotator j for each test pairwise comparison
    array[I*J*C*K_test*(K_test-1)/2] int test_pairwise_attribute;      // attribute i for each test pairwise comparison
    array[I*J*C*K_test*(K_test-1)/2] int test_pairwise_tied_rating;    // rating value that tied (x_ijk1 = x_ijk2 = rating_val)
    array[I*J*C*K_test*(K_test-1)/2] int test_pairwise_observed;       // 1 if observed, 0 if missing
    
    // Counts
    int num_train_pairwise_rankings;    // Number of training pairwise comparisons generated
    int num_test_pairwise_rankings;     // Number of test pairwise comparisons generated
    int num_train_observed_ratings;     // Number of observed training ratings (R_ijk_train = 1)
    int num_test_observed_ratings;      // Number of observed test ratings (R_ijk_test = 1)
    int num_train_missing_ratings;      // Number of missing training ratings (R_ijk_train = 0)
    int num_test_missing_ratings;       // Number of missing test ratings (R_ijk_test = 0)
    
    // ===== GENERATE SHARED COMPONENTS =====
    
    // Generate mean preferences: v_i ~ N(0,I) - SHARED across train/test
    for (i in 1:I) {
        for (d in 1:D) {
            mean_preferences[i, d] = normal_rng(0, 1);
        }
    }
    
    // Generate annotator preferences: v_ij ~ N(v_i, σ²I) - SHARED across train/test
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (d in 1:D) {
                annotator_preferences[idx, d] = normal_rng(mean_preferences[i, d], sigma_annotator);
            }
        }
    }
    
    // Generate rating thresholds: p_ij ~ Dir(α/C, ..., α/C) - SHARED across train/test
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            rating_probs[idx] = dirichlet_rng(rep_vector(alpha_dirichlet/C, C));
            rating_thresholds[idx] = cumulative_sum(rating_probs[idx]);
        }
    }
    
    // ===== GENERATE TRAINING INSTANCE =====
    
    // Generate training item embeddings: e_k_train ~ N(0,I)
    for (k in 1:K_train) {
        for (d in 1:D) {
            train_embeddings[k, d] = normal_rng(0, 1);
        }
    }
    
    // Compute training base scores: z_ijk_train = v_ij · e_k_train
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_train) {
                train_base_scores[idx, k] = dot_product(annotator_preferences[idx], train_embeddings[k]);
            }
        }
    }
    
    // ===== GENERATE TEST INSTANCE =====
    
    // Generate test item embeddings: e_k_test ~ N(0,I) - DIFFERENT from training
    for (k in 1:K_test) {
        for (d in 1:D) {
            test_embeddings[k, d] = normal_rng(0, 1);
        }
    }
    
    // Compute test base scores: z_ijk_test = v_ij · e_k_test
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_test) {
                test_base_scores[idx, k] = dot_product(annotator_preferences[idx], test_embeddings[k]);
            }
        }
    }
    
    // ===== GENERATE TRAINING RATINGS =====
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_train) {
                // Generate rating with noise and binning: x_ijk_train = Bin(z_ijk_train + ε_ijk, σ_measurement)
                real noisy_score = train_base_scores[idx, k] + normal_rng(0, sigma_measurement);
                real pref_norm = sqrt(dot_self(annotator_preferences[idx]));
                real total_std = sqrt(pref_norm^2 + sigma_measurement^2);
                real standardized_score = noisy_score / total_std;
                real cdf_val = Phi(standardized_score);
                
                // Bin into rating categories using thresholds q_ij
                int rating = 1;
                for (c in 1:C) {
                    if (cdf_val <= rating_thresholds[idx][c]) {
                        rating = c;
                        break;
                    }
                }
                train_rating_values[idx, k] = rating;
                train_rating_observed[idx, k] = 0;  // Initialize as missing
            }
        }
    }
    
    // ===== GENERATE TEST RATINGS =====
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_test) {
                // Generate rating with noise and binning: x_ijk_test = Bin(z_ijk_test + ε_ijk, σ_measurement)
                real noisy_score = test_base_scores[idx, k] + normal_rng(0, sigma_measurement);
                real pref_norm = sqrt(dot_self(annotator_preferences[idx]));
                real total_std = sqrt(pref_norm^2 + sigma_measurement^2);
                real standardized_score = noisy_score / total_std;
                real cdf_val = Phi(standardized_score);
                
                // Bin into rating categories using thresholds q_ij
                int rating = 1;
                for (c in 1:C) {
                    if (cdf_val <= rating_thresholds[idx][c]) {
                        rating = c;
                        break;
                    }
                }
                test_rating_values[idx, k] = rating;
                test_rating_observed[idx, k] = 0;  // Initialize as missing
            }
        }
    }
    
    // ===== IMPLEMENT OBSERVATION PROTOCOL =====
    // Determine annotator sets: training uses first 2/3, test uses last 2/3
    int train_annotator_start = 1;           // Training annotators: j ∈ {1, ..., 2J/3}
    int train_annotator_end = (2 * J) / 3;
    int test_annotator_start = J / 3 + 1;    // Test annotators: j ∈ {J/3+1, ..., J}
    int test_annotator_end = J;
    
    // ===== TRAINING INSTANCE OBSERVATION PROTOCOL =====
    for (k in 1:K_train) {  // for each training item
        // Sample two different annotators from training set
        real u1 = uniform_rng(0, 1);
        int j1 = train_annotator_start + to_int(floor(u1 * (train_annotator_end - train_annotator_start + 1)));
        if (j1 > train_annotator_end) j1 = train_annotator_end;
        
        // Sample j2 such that j2 != j1
        int j2 = j1;
        while (j2 == j1) {
            real u2 = uniform_rng(0, 1);
            j2 = train_annotator_start + to_int(floor(u2 * (train_annotator_end - train_annotator_start + 1)));
            if (j2 > train_annotator_end) j2 = train_annotator_end;
        }
        
        // Assign both annotators to rate this item on all criteria
        for (i in 1:I) {
            int idx1 = (i-1)*J + j1;
            int idx2 = (i-1)*J + j2;
            train_rating_observed[idx1, k] = 1;
            train_rating_observed[idx2, k] = 1;
            
            // Check for disagreement > 1 and add third annotator if enabled
            if (enable_third_annotator == 1) {
                int rating1 = train_rating_values[idx1, k];
                int rating2 = train_rating_values[idx2, k];
                if (abs(rating1 - rating2) > 1) {
                    // Find a third annotator from the training set, ensuring j3 != j1 and j3 != j2
                    int j3 = j1;
                    while (j3 == j1 || j3 == j2) {
                        real u3 = uniform_rng(0, 1);
                        j3 = train_annotator_start + to_int(floor(u3 * (train_annotator_end - train_annotator_start + 1)));
                        if (j3 > train_annotator_end) j3 = train_annotator_end;
                    }

                    int idx3 = (i-1)*J + j3;
                    train_rating_observed[idx3, k] = 1;
                }
            }
        }
    }
    
    // ===== TEST INSTANCE OBSERVATION PROTOCOL =====
    for (k in 1:K_test) {
        // Sample two different annotators from test set
        real u1 = uniform_rng(0, 1);
        int j1 = test_annotator_start + to_int(floor(u1 * (test_annotator_end - test_annotator_start + 1)));
        if (j1 > test_annotator_end) j1 = test_annotator_end;
        
        int j2 = j1;
        while (j2 == j1) {
            real u2 = uniform_rng(0, 1);
            j2 = test_annotator_start + to_int(floor(u2 * (test_annotator_end - test_annotator_start + 1)));
            if (j2 > test_annotator_end) j2 = test_annotator_end;
        }
        
        // Assign both annotators to rate this item on all criteria
        for (i in 1:I) {
            int idx1 = (i-1)*J + j1;
            int idx2 = (i-1)*J + j2;
            test_rating_observed[idx1, k] = 1;
            test_rating_observed[idx2, k] = 1;
            
            // Check for disagreement > 1 and add third annotator if enabled
            if (enable_third_annotator == 1) {
                int rating1 = test_rating_values[idx1, k];
                int rating2 = test_rating_values[idx2, k];
                if (abs(rating1 - rating2) > 1) {
                    // Find a third annotator from the test set, ensuring j3 != j1 and j3 != j2
                    int j3 = j1;
                    while (j3 == j1 || j3 == j2) {
                        real u3 = uniform_rng(0, 1);
                        j3 = test_annotator_start + to_int(floor(u3 * (test_annotator_end - test_annotator_start + 1)));
                        if (j3 > test_annotator_end) j3 = test_annotator_end;
                    }
                    
                    int idx3 = (i-1)*J + j3;
                    test_rating_observed[idx3, k] = 1;
                }
            }
        }
    }
    
    // ===== GENERATE PAIRWISE RANKINGS FROM TIED RATINGS =====
    num_train_pairwise_rankings = 0;
    num_test_pairwise_rankings = 0;
    
    if (enable_pairwise_rankings == 1) {
    // ===== TRAINING PAIRWISE RANKINGS =====
    for (i in 1:I) {
        for (j in train_annotator_start:train_annotator_end) {
            int ij_idx = (i-1)*J + j;
            
            // For each rating category, find tied items that are observed
            for (rating_val in 1:C) {
                array[K_train] int tied_items;
                int num_tied = 0;
                
                    // Find all training items (i,j,k) with this rating value that are observed (r=1)
                    for (k in 1:K_train) {
                        if (train_rating_values[ij_idx, k] == rating_val && train_rating_observed[ij_idx, k] == 1) {
                        num_tied += 1;
                        tied_items[num_tied] = k;
                    }
                }
                
                    // Generate per-item capped comparisons (sample without replacement)
                    if (num_tied >= 2 && pairwise_cap_per_item > 0) {
                        for (idx1 in 1:num_tied) {
                            int item1 = tied_items[idx1];
                            
                            // Create list of available comparison partners (excluding idx1)
                            array[num_tied-1] int available_indices;
                            int num_available = 0;
                            for (idx in 1:num_tied) {
                                if (idx != idx1) {
                                    num_available += 1;
                                    available_indices[num_available] = idx;
                                }
                            }
                            
                            // Sample without replacement up to cap
                            int max_comparisons = min(pairwise_cap_per_item, num_available);
                            for (comp_idx in 1:max_comparisons) {
                                // Sample from remaining available indices
                                real u = uniform_rng(0, 1);
                                int sample_idx = 1 + to_int(floor(u * (num_available - comp_idx + 1)));
                                if (sample_idx > num_available - comp_idx + 1) sample_idx = num_available - comp_idx + 1;
                                
                                int idx2 = available_indices[sample_idx];
                                int item2 = tied_items[idx2];
                                
                                // Gumbel tie-break: Bradley-Terry model with temperature T
                                real u1 = uniform_rng(0, 1);
                                real g1 = -log(-log(u1));  // Gumbel noise G1
                                real util1 = train_base_scores[ij_idx, item1] / temperature + g1;  // U1 = z_ijk1/T + G1
                                real u2 = uniform_rng(0, 1);
                                real g2 = -log(-log(u2));  // Gumbel noise G2
                                real util2 = train_base_scores[ij_idx, item2] / temperature + g2;  // U2 = z_ijk2/T + G2
                                int order = (util1 > util2) ? 1 : 2;  // 1 if item1 > item2, 2 if item2 > item1
                                
                                // Append training pairwise ranking
                                num_train_pairwise_rankings += 1;
                                train_pairwise_items[num_train_pairwise_rankings, 1] = item1;
                                train_pairwise_items[num_train_pairwise_rankings, 2] = item2;
                                train_pairwise_orders[num_train_pairwise_rankings] = order;
                                train_pairwise_annotator[num_train_pairwise_rankings] = j;
                                train_pairwise_attribute[num_train_pairwise_rankings] = i;
                                train_pairwise_tied_rating[num_train_pairwise_rankings] = rating_val;
                                train_pairwise_observed[num_train_pairwise_rankings] = 1;
                                
                                // Remove selected index from available list (swap with last)
                                available_indices[sample_idx] = available_indices[num_available - comp_idx + 1];
                            }
                        }
                    }
                }
            }
        }
        
        // ===== TEST PAIRWISE RANKINGS =====
        for (i in 1:I) {
            for (j in test_annotator_start:test_annotator_end) {
                int ij_idx = (i-1)*J + j;
                
                // For each rating category, find tied items that are observed
                for (rating_val in 1:C) {
                    array[K_test] int tied_items;
                    int num_tied = 0;
                    
                    // Find all test items with this rating value that are observed
                    for (k in 1:K_test) {
                        if (test_rating_values[ij_idx, k] == rating_val && test_rating_observed[ij_idx, k] == 1) {
                            num_tied += 1;
                            tied_items[num_tied] = k;
                        }
                    }
                
                    // Generate per-item capped comparisons (sample without replacement)
                    if (num_tied >= 2 && pairwise_cap_per_item > 0) {
                        for (idx1 in 1:num_tied) {
                            int item1 = tied_items[idx1];
                            
                            // Create list of available comparison partners (excluding idx1)
                            array[num_tied-1] int available_indices;
                            int num_available = 0;
                            for (idx in 1:num_tied) {
                                if (idx != idx1) {
                                    num_available += 1;
                                    available_indices[num_available] = idx;
                                }
                            }
                            
                            // Sample without replacement up to cap
                            int max_comparisons = min(pairwise_cap_per_item, num_available);
                            for (comp_idx in 1:max_comparisons) {
                                // Sample from remaining available indices
                                real u = uniform_rng(0, 1);
                                int sample_idx = 1 + to_int(floor(u * (num_available - comp_idx + 1)));
                                if (sample_idx > num_available - comp_idx + 1) sample_idx = num_available - comp_idx + 1;
                                
                                int idx2 = available_indices[sample_idx];
                                int item2 = tied_items[idx2];
                                
                                // Gumbel tie-break: Bradley-Terry model with temperature T
                                real u1 = uniform_rng(0, 1);
                                real g1 = -log(-log(u1));  // Gumbel noise G1
                                real util1 = test_base_scores[ij_idx, item1] / temperature + g1;  // U1 = z_ijk1/T + G1
                                real u2 = uniform_rng(0, 1);
                                real g2 = -log(-log(u2));  // Gumbel noise G2
                                real util2 = test_base_scores[ij_idx, item2] / temperature + g2;  // U2 = z_ijk2/T + G2
                                int order = (util1 > util2) ? 1 : 2;  // 1 if item1 > item2, 2 if item2 > item1
                                
                                // Append test pairwise ranking
                                num_test_pairwise_rankings += 1;
                                test_pairwise_items[num_test_pairwise_rankings, 1] = item1;
                                test_pairwise_items[num_test_pairwise_rankings, 2] = item2;
                                test_pairwise_orders[num_test_pairwise_rankings] = order;
                                test_pairwise_annotator[num_test_pairwise_rankings] = j;
                                test_pairwise_attribute[num_test_pairwise_rankings] = i;
                                test_pairwise_tied_rating[num_test_pairwise_rankings] = rating_val;
                                test_pairwise_observed[num_test_pairwise_rankings] = 1;
                                
                                // Remove selected index from available list (swap with last)
                                available_indices[sample_idx] = available_indices[num_available - comp_idx + 1];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // ===== COMPUTE COUNTS =====
    
    // Training instance counts
    num_train_observed_ratings = 0;
    num_train_missing_ratings = 0;
    for (i in 1:(I*J)) {
        for (k in 1:K_train) {
            if (train_rating_observed[i, k] == 1) {
                num_train_observed_ratings += 1;
            } else {
                num_train_missing_ratings += 1;
            }
        }
    }
    
    // Test instance counts
    num_test_observed_ratings = 0;
    num_test_missing_ratings = 0;
    for (i in 1:(I*J)) {
        for (k in 1:K_test) {
            if (test_rating_observed[i, k] == 1) {
                num_test_observed_ratings += 1;
        } else {
                num_test_missing_ratings += 1;
            }
        }
    }
}
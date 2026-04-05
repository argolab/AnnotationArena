data {
    // Dimensions
    int<lower=1> K_train;  // number of items in training instance
    int<lower=1> K_test;   // number of items in test instance
    int<lower=1> K_val;    // number of items in validation instance
    int<lower=1> I;        // number of criteria (attributes)
    int<lower=1> J;         // number of annotators
    int<lower=1> D;         // embedding dimension
    int<lower=1> C;         // number of rating categories
    int<lower=1> d_annotator;  // annotator embedding dimension (can be <= D for low-rank)

    // Observation protocol controls
    int<lower=0, upper=1> enable_pairwise_rankings;  // Ablation: enable pairwise rankings
    int<lower=0> pairwise_cap_per_item;     // Max comparisons per item within its tied group

    // Hyperparameters
    real<lower=0> sigma_annotator;
    real<lower=0> sigma_measurement;
    real<lower=0> kappa;
    real<lower=0> temperature;

    int<lower=1> num_annotate_annotator;

    // Annotator model selection
    // 0 = old spherical model: V_ij ~ N(v_i, sigma^2) independently
    // 1 = new factored model: V_ij = v_i + u_j * M_i (allows covariance learning)
    int<lower=0, upper=1> use_factored_annotator;

    // Rating threshold derivation
    // 0 = independent Dirichlet samples for each (i,j) pair
    // 1 = derive thresholds from annotator embedding u_j (reduces d.f., consistent annotator style)
    // Only meaningful when use_factored_annotator=1; ignored otherwise
    int<lower=0, upper=1> derive_thresholds_from_annotator;
}

generated quantities {

    // ===== SHARED COMPONENTS (same for train, val, and test) =====
    // Debug printing flag (0 = off, 1 = on). Toggle here.
    int DEBUG_PRINT;
    DEBUG_PRINT = 1;

    int train_annotator_start = 1;
    int train_annotator_end = J;
    int val_annotator_start = 1;
    int val_annotator_end = J;
    int test_annotator_start = 1;
    int test_annotator_end = J;

    matrix[I, D] mean_preferences;              // Global criteria embeddings v_i
    matrix[J, d_annotator] annotator_embeddings; // Annotator embeddings u_j
    array[I] matrix[d_annotator, D] attr_transforms; // Attribute-specific transforms M_i
    matrix[I*J, D] annotator_preferences;      // All annotator preferences V_ij = v_i + u_j * M_i
    array[I*J] simplex[C] rating_probs;        // Rating probabilities p_ij
    array[I*J] vector[C] rating_cumprobs;      // Cumulative probabilities q_ij = cumsum(p_ij) in (0,1]
    array[I*J] vector[C+1] rating_thresholds_z; // Z-cutpoints: [-inf, inv_Phi(cumprob[1..C-1]), +inf]

    // Threshold transform matrix (used when derive_thresholds_from_annotator=1)
    matrix[C-1, d_annotator] threshold_transform_W;
    // Per-attribute threshold biases (optional attribute-specific style variation)
    matrix[I, C-1] threshold_attr_bias;
    
    // ===== TRAINING INSTANCE =====
    matrix[K_train, D] train_embeddings;
    matrix[I*J, K_train] train_base_scores;
    array[I*J, K_train] int train_rating_values;
    array[I*J, K_train] int train_rating_observed;
    
    // ===== VALIDATION INSTANCE =====
    matrix[K_val, D] val_embeddings;
    matrix[I*J, K_val] val_base_scores;
    array[I*J, K_val] int val_rating_values;
    array[I*J, K_val] int val_rating_observed;

    // ===== TEST INSTANCE =====
    matrix[K_test, D] test_embeddings;
    matrix[I*J, K_test] test_base_scores;
    array[I*J, K_test] int test_rating_values;
    array[I*J, K_test] int test_rating_observed;
    
    // ===== PAIRWISE RANKINGS =====
    // Training pairwise rankings
    array[I*J*C*(K_train*(K_train-1)%/%2), 2] int train_pairwise_items;
    array[I*J*C*(K_train*(K_train-1)%/%2)] int train_pairwise_orders;
    array[I*J*C*(K_train*(K_train-1)%/%2)] int train_pairwise_annotator;
    array[I*J*C*(K_train*(K_train-1)%/%2)] int train_pairwise_attribute;
    array[I*J*C*(K_train*(K_train-1)%/%2)] int train_pairwise_tied_rating;
    array[I*J*C*(K_train*(K_train-1)%/%2)] int train_pairwise_observed;

    // Validation pairwise rankings
    array[I*J*C*(K_val*(K_val-1)%/%2), 2] int val_pairwise_items;
    array[I*J*C*(K_val*(K_val-1)%/%2)] int val_pairwise_orders;
    array[I*J*C*(K_val*(K_val-1)%/%2)] int val_pairwise_annotator;
    array[I*J*C*(K_val*(K_val-1)%/%2)] int val_pairwise_attribute;
    array[I*J*C*(K_val*(K_val-1)%/%2)] int val_pairwise_tied_rating;
    array[I*J*C*(K_val*(K_val-1)%/%2)] int val_pairwise_observed;

    // Test pairwise rankings
    array[I*J*C*(K_test*(K_test-1)%/%2), 2] int test_pairwise_items;
    array[I*J*C*(K_test*(K_test-1)%/%2)] int test_pairwise_orders;
    array[I*J*C*(K_test*(K_test-1)%/%2)] int test_pairwise_annotator;
    array[I*J*C*(K_test*(K_test-1)%/%2)] int test_pairwise_attribute;
    array[I*J*C*(K_test*(K_test-1)%/%2)] int test_pairwise_tied_rating;
    array[I*J*C*(K_test*(K_test-1)%/%2)] int test_pairwise_observed;

    // Counts
    int num_train_pairwise_rankings;
    int num_val_pairwise_rankings;
    int num_test_pairwise_rankings;
    int num_train_observed_ratings;
    int num_val_observed_ratings;
    int num_test_observed_ratings;
    int num_train_missing_ratings;
    int num_val_missing_ratings;
    int num_test_missing_ratings;
    
    // Posterior rating probabilities (due to measurement error)
    array[I*J, K_train] vector[C] train_posterior_rating_probs;
    array[I*J, K_val] vector[C] val_posterior_rating_probs;
    array[I*J, K_test] vector[C] test_posterior_rating_probs;
    
    // ===== GENERATE SHARED COMPONENTS =====

    real sigma_M = sigma_annotator / sqrt(d_annotator);

    {
        // Step 1: Generate attribute embeddings v_i
        for (i in 1:I) {
            for (d in 1:D) {
                mean_preferences[i, d] = normal_rng(0, 1);
            }
        }

        if (use_factored_annotator == 1) {
            // ===== NEW FACTORED MODEL =====
            for (j in 1:J) {
                for (d in 1:d_annotator) {
                    annotator_embeddings[j, d] = normal_rng(0, 1);
                }
            }
            for (i in 1:I) {
                for (r in 1:d_annotator) {
                    for (c in 1:D) {
                        attr_transforms[i, r, c] = normal_rng(0, sigma_M);
                    }
                }
            }
            for (i in 1:I) {
                for (j in 1:J) {
                    int idx = (i-1)*J + j;
                    annotator_preferences[idx] = mean_preferences[i] + annotator_embeddings[j] * attr_transforms[i];
                }
            }

        } else {
            // ===== OLD SPHERICAL MODEL =====
            for (j in 1:J) {
                for (d in 1:d_annotator) {
                    annotator_embeddings[j, d] = 0;
                }
            }
            for (i in 1:I) {
                for (r in 1:d_annotator) {
                    for (c in 1:D) {
                        attr_transforms[i, r, c] = 0;
                    }
                }
            }
            for (i in 1:I) {
                for (j in 1:J) {
                    int idx = (i-1)*J + j;
                    for (d in 1:D) {
                        annotator_preferences[idx, d] = normal_rng(mean_preferences[i, d], sigma_annotator);
                    }
                }
            }
        }
    }
    
    // ===== GENERATE THRESHOLD TRANSFORM =====
    {
        real sigma_W = 1.0 / sqrt(d_annotator);
        for (c in 1:(C-1)) {
            for (d in 1:d_annotator) {
                threshold_transform_W[c, d] = normal_rng(0, sigma_W);
            }
        }
        real sigma_bias = 0.1;
        for (i in 1:I) {
            for (c in 1:(C-1)) {
                threshold_attr_bias[i, c] = normal_rng(0, sigma_bias);
            }
        }
    }

    if (derive_thresholds_from_annotator == 1 && use_factored_annotator == 1) {
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                vector[C-1] threshold_logits = threshold_transform_W * to_vector(annotator_embeddings[j]);
                threshold_logits = threshold_logits + to_vector(threshold_attr_bias[i]);
                vector[C] logits_full;
                logits_full[1] = 0;
                for (c in 2:C) {
                    logits_full[c] = threshold_logits[c-1];
                }
                real max_logit = max(logits_full);
                vector[C] exp_logits;
                for (c in 1:C) {
                    exp_logits[c] = exp(logits_full[c] - max_logit);
                }
                real sum_exp = sum(exp_logits);
                for (c in 1:C) {
                    rating_probs[idx][c] = exp_logits[c] / sum_exp;
                }
                rating_cumprobs[idx] = cumulative_sum(rating_probs[idx]);
            }
        }
    } else {
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                rating_probs[idx] = dirichlet_rng(rep_vector(kappa/C, C));
                rating_cumprobs[idx] = cumulative_sum(rating_probs[idx]);
            }
        }
    }
    
    // Convert cumulative probabilities to standard normal cutpoints (z-space)
    for (ij in 1:(I*J)) {
        rating_thresholds_z[ij][1] = negative_infinity();
        for (c in 2:C) {
            rating_thresholds_z[ij][c] = inv_Phi(rating_cumprobs[ij][c-1]);
        }
        rating_thresholds_z[ij][C+1] = positive_infinity();
    }
    
    // ===== GENERATE TRAINING INSTANCE =====
    for (k in 1:K_train) {
        for (d in 1:D) {
            train_embeddings[k, d] = normal_rng(0, 1);
        }
    }
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_train) {
                train_base_scores[idx, k] = dot_product(annotator_preferences[idx], train_embeddings[k]);
            }
        }
    }

    // ===== GENERATE VALIDATION INSTANCE =====
    for (k in 1:K_val) {
        for (d in 1:D) {
            val_embeddings[k, d] = normal_rng(0, 1);
        }
    }
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_val) {
                val_base_scores[idx, k] = dot_product(annotator_preferences[idx], val_embeddings[k]);
            }
        }
    }

    // ===== GENERATE TEST INSTANCE =====
    for (k in 1:K_test) {
        for (d in 1:D) {
            test_embeddings[k, d] = normal_rng(0, 1);
        }
    }
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_test) {
                test_base_scores[idx, k] = dot_product(annotator_preferences[idx], test_embeddings[k]);
            }
        }
    }
    
    // ===== GENERATE TRAINING RATINGS =====
    for (k in 1:K_train) {
        if ((train_annotator_end - train_annotator_start + 1) < num_annotate_annotator) {
            reject("Number of available training annotators is less than num_annotate_annotator");
        }
        array[num_annotate_annotator] int selected_annotators;
        int num_selected = 0;
        
        while (num_selected < num_annotate_annotator) {
            real u = uniform_rng(0, 1);
            int candidate = train_annotator_start + to_int(floor(u * (train_annotator_end - train_annotator_start + 1)));
            if (candidate > train_annotator_end) candidate = train_annotator_end;
            int already_selected = 0;
            for (s in 1:num_selected) {
                if (selected_annotators[s] == candidate) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected == 0) {
                num_selected += 1;
                selected_annotators[num_selected] = candidate;
            }
        }
        
        for (i in 1:I) {
            for (s in 1:num_annotate_annotator) {
                int j = selected_annotators[s];
                int idx = (i-1)*J + j;
                train_rating_observed[idx, k] = 1;
                real noisy_score = train_base_scores[idx, k] + normal_rng(0, sigma_measurement);
                real pref_norm_sq = dot_self(annotator_preferences[idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real standardized_score = noisy_score / total_std;
                real cdf_val = Phi(standardized_score);
                int rating = 1;
                for (c in 1:C) {
                    if (cdf_val <= rating_cumprobs[idx][c]) {
                        rating = c;
                        break;
                    }
                }
                train_rating_values[idx, k] = rating;
            }
        }
    }

    // ===== GENERATE VALIDATION RATINGS =====
    for (k in 1:K_val) {
        if ((val_annotator_end - val_annotator_start + 1) < num_annotate_annotator) {
            reject("Number of available validation annotators is less than num_annotate_annotator");
        }
        array[num_annotate_annotator] int selected_annotators;
        int num_selected = 0;
        
        while (num_selected < num_annotate_annotator) {
            real u = uniform_rng(0, 1);
            int candidate = val_annotator_start + to_int(floor(u * (val_annotator_end - val_annotator_start + 1)));
            if (candidate > val_annotator_end) candidate = val_annotator_end;
            int already_selected = 0;
            for (s in 1:num_selected) {
                if (selected_annotators[s] == candidate) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected == 0) {
                num_selected += 1;
                selected_annotators[num_selected] = candidate;
            }
        }
        
        for (i in 1:I) {
            for (s in 1:num_annotate_annotator) {
                int j = selected_annotators[s];
                int idx = (i-1)*J + j;
                val_rating_observed[idx, k] = 1;
                real noisy_score = val_base_scores[idx, k] + normal_rng(0, sigma_measurement);
                real pref_norm_sq = dot_self(annotator_preferences[idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real standardized_score = noisy_score / total_std;
                real cdf_val = Phi(standardized_score);
                int rating = 1;
                for (c in 1:C) {
                    if (cdf_val <= rating_cumprobs[idx][c]) {
                        rating = c;
                        break;
                    }
                }
                val_rating_values[idx, k] = rating;
            }
        }
    }
    
    // ===== GENERATE TEST RATINGS =====
    for (k in 1:K_test) {
        if ((test_annotator_end - test_annotator_start + 1) < num_annotate_annotator) {
            reject("Number of available test annotators is less than num_annotate_annotator");
        }
        array[num_annotate_annotator] int selected_annotators;
        int num_selected = 0;
        
        while (num_selected < num_annotate_annotator) {
            real u = uniform_rng(0, 1);
            int candidate = test_annotator_start + to_int(floor(u * (test_annotator_end - test_annotator_start + 1)));
            if (candidate > test_annotator_end) candidate = test_annotator_end;
            int already_selected = 0;
            for (s in 1:num_selected) {
                if (selected_annotators[s] == candidate) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected == 0) {
                num_selected += 1;
                selected_annotators[num_selected] = candidate;
            }
        }
        
        for (i in 1:I) {
            for (s in 1:num_annotate_annotator) {
                int j = selected_annotators[s];
                int idx = (i-1)*J + j;
                test_rating_observed[idx, k] = 1;
                real noisy_score = test_base_scores[idx, k] + normal_rng(0, sigma_measurement);
                real pref_norm_sq = dot_self(annotator_preferences[idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real standardized_score = noisy_score / total_std;
                real cdf_val = Phi(standardized_score);
                int rating = 1;
                for (c in 1:C) {
                    if (cdf_val <= rating_cumprobs[idx][c]) {
                        rating = c;
                        break;
                    }
                }
                test_rating_values[idx, k] = rating;
            }
        }
    }
    
    // ===== COMPUTE POSTERIOR RATING PROBABILITIES =====

    // Training posterior rating probabilities
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_train) {
                real true_base_score = train_base_scores[idx, k];
                real pref_norm_sq = dot_self(annotator_preferences[idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real mean_std = true_base_score / total_std;
                real cond_std = sigma_measurement / total_std;
                if (cond_std == 0) {
                    int one_c = 1;
                    for (c in 1:C) {
                        if (mean_std <= rating_thresholds_z[idx][c+1]) {
                            one_c = c;
                            break;
                        }
                    }
                    for (c in 1:C) {
                        train_posterior_rating_probs[idx, k][c] = (c == one_c) ? 1 : 0;
                    }
                } else {
                    for (c in 1:C) {
                        real upper_z = rating_thresholds_z[idx][c+1];
                        real lower_z = rating_thresholds_z[idx][c];
                        real upper_prob = Phi((upper_z - mean_std) / cond_std);
                        real lower_prob = Phi((lower_z - mean_std) / cond_std);
                        train_posterior_rating_probs[idx, k][c] = upper_prob - lower_prob;
                    }
                }
            }
        }
    }

    // Validation posterior rating probabilities
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_val) {
                real true_base_score = val_base_scores[idx, k];
                real pref_norm_sq = dot_self(annotator_preferences[idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real mean_std = true_base_score / total_std;
                real cond_std = sigma_measurement / total_std;
                if (cond_std == 0) {
                    int one_c = 1;
                    for (c in 1:C) {
                        if (mean_std <= rating_thresholds_z[idx][c+1]) {
                            one_c = c;
                            break;
                        }
                    }
                    for (c in 1:C) {
                        val_posterior_rating_probs[idx, k][c] = (c == one_c) ? 1 : 0;
                    }
                } else {
                    for (c in 1:C) {
                        real upper_z = rating_thresholds_z[idx][c+1];
                        real lower_z = rating_thresholds_z[idx][c];
                        real upper_prob = Phi((upper_z - mean_std) / cond_std);
                        real lower_prob = Phi((lower_z - mean_std) / cond_std);
                        val_posterior_rating_probs[idx, k][c] = upper_prob - lower_prob;
                    }
                }
            }
        }
    }
    
    // Test posterior rating probabilities
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_test) {
                real true_base_score = test_base_scores[idx, k];
                real pref_norm_sq = dot_self(annotator_preferences[idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real mean_std = true_base_score / total_std;
                real cond_std = sigma_measurement / total_std;
                if (cond_std == 0) {
                    int one_c = 1;
                    for (c in 1:C) {
                        if (mean_std <= rating_thresholds_z[idx][c+1]) {
                            one_c = c;
                            break;
                        }
                    }
                    for (c in 1:C) {
                        test_posterior_rating_probs[idx, k][c] = (c == one_c) ? 1 : 0;
                    }
                } else {
                    for (c in 1:C) {
                        real upper_z = rating_thresholds_z[idx][c+1];
                        real lower_z = rating_thresholds_z[idx][c];
                        real upper_prob = Phi((upper_z - mean_std) / cond_std);
                        real lower_prob = Phi((lower_z - mean_std) / cond_std);
                        test_posterior_rating_probs[idx, k][c] = upper_prob - lower_prob;
                    }
                }
            }
        }
    }

    
    // ===== GENERATE PAIRWISE RANKINGS FROM TIED RATINGS =====
    num_train_pairwise_rankings = 0;
    num_val_pairwise_rankings = 0;
    num_test_pairwise_rankings = 0;
    
    if (enable_pairwise_rankings == 1) {

        // ===== TRAINING PAIRWISE RANKINGS =====
        for (i in 1:I) {
            for (j in train_annotator_start:train_annotator_end) {
                int ij_idx = (i-1)*J + j;
                for (rating_val in 1:C) {
                    array[K_train] int tied_items;
                    int num_tied = 0;
                    for (k in 1:K_train) {
                        if (train_rating_values[ij_idx, k] == rating_val && train_rating_observed[ij_idx, k] == 1) {
                            num_tied += 1;
                            tied_items[num_tied] = k;
                        }
                    }
                    if (num_tied >= 2 && pairwise_cap_per_item > 0) {
                        for (idx1 in 1:num_tied) {
                            int item1 = tied_items[idx1];
                            array[num_tied-1] int available_indices;
                            int num_available = 0;
                            for (idx in 1:num_tied) {
                                if (idx != idx1) {
                                    num_available += 1;
                                    available_indices[num_available] = idx;
                                }
                            }
                            int max_comparisons = min(pairwise_cap_per_item, num_available);
                            for (comp_idx in 1:max_comparisons) {
                                real u = uniform_rng(0, 1);
                                int sample_idx = 1 + to_int(floor(u * (num_available - comp_idx + 1)));
                                if (sample_idx > num_available - comp_idx + 1) sample_idx = num_available - comp_idx + 1;
                                int idx2 = available_indices[sample_idx];
                                int item2 = tied_items[idx2];
                                real u1 = uniform_rng(0, 1);
                                real g1 = -log(-log(u1));
                                real util1 = train_base_scores[ij_idx, item1] / temperature + g1;
                                real u2 = uniform_rng(0, 1);
                                real g2 = -log(-log(u2));
                                real util2 = train_base_scores[ij_idx, item2] / temperature + g2;
                                int order = (util1 > util2) ? 1 : 2;
                                num_train_pairwise_rankings += 1;
                                train_pairwise_items[num_train_pairwise_rankings, 1] = item1;
                                train_pairwise_items[num_train_pairwise_rankings, 2] = item2;
                                train_pairwise_orders[num_train_pairwise_rankings] = order;
                                train_pairwise_annotator[num_train_pairwise_rankings] = j;
                                train_pairwise_attribute[num_train_pairwise_rankings] = i;
                                train_pairwise_tied_rating[num_train_pairwise_rankings] = rating_val;
                                train_pairwise_observed[num_train_pairwise_rankings] = 1;
                                available_indices[sample_idx] = available_indices[num_available - comp_idx + 1];
                            }
                        }
                    }
                }
            }
        }

        // ===== VALIDATION PAIRWISE RANKINGS =====
        for (i in 1:I) {
            for (j in val_annotator_start:val_annotator_end) {
                int ij_idx = (i-1)*J + j;
                for (rating_val in 1:C) {
                    array[K_val] int tied_items;
                    int num_tied = 0;
                    for (k in 1:K_val) {
                        if (val_rating_values[ij_idx, k] == rating_val && val_rating_observed[ij_idx, k] == 1) {
                            num_tied += 1;
                            tied_items[num_tied] = k;
                        }
                    }
                    if (num_tied >= 2 && pairwise_cap_per_item > 0) {
                        for (idx1 in 1:num_tied) {
                            int item1 = tied_items[idx1];
                            array[num_tied-1] int available_indices;
                            int num_available = 0;
                            for (idx in 1:num_tied) {
                                if (idx != idx1) {
                                    num_available += 1;
                                    available_indices[num_available] = idx;
                                }
                            }
                            int max_comparisons = min(pairwise_cap_per_item, num_available);
                            for (comp_idx in 1:max_comparisons) {
                                real u = uniform_rng(0, 1);
                                int sample_idx = 1 + to_int(floor(u * (num_available - comp_idx + 1)));
                                if (sample_idx > num_available - comp_idx + 1) sample_idx = num_available - comp_idx + 1;
                                int idx2 = available_indices[sample_idx];
                                int item2 = tied_items[idx2];
                                real u1 = uniform_rng(0, 1);
                                real g1 = -log(-log(u1));
                                real util1 = val_base_scores[ij_idx, item1] / temperature + g1;
                                real u2 = uniform_rng(0, 1);
                                real g2 = -log(-log(u2));
                                real util2 = val_base_scores[ij_idx, item2] / temperature + g2;
                                int order = (util1 > util2) ? 1 : 2;
                                num_val_pairwise_rankings += 1;
                                val_pairwise_items[num_val_pairwise_rankings, 1] = item1;
                                val_pairwise_items[num_val_pairwise_rankings, 2] = item2;
                                val_pairwise_orders[num_val_pairwise_rankings] = order;
                                val_pairwise_annotator[num_val_pairwise_rankings] = j;
                                val_pairwise_attribute[num_val_pairwise_rankings] = i;
                                val_pairwise_tied_rating[num_val_pairwise_rankings] = rating_val;
                                val_pairwise_observed[num_val_pairwise_rankings] = 1;
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
                for (rating_val in 1:C) {
                    array[K_test] int tied_items;
                    int num_tied = 0;
                    for (k in 1:K_test) {
                        if (test_rating_values[ij_idx, k] == rating_val && test_rating_observed[ij_idx, k] == 1) {
                            num_tied += 1;
                            tied_items[num_tied] = k;
                        }
                    }
                    if (num_tied >= 2 && pairwise_cap_per_item > 0) {
                        for (idx1 in 1:num_tied) {
                            int item1 = tied_items[idx1];
                            array[num_tied-1] int available_indices;
                            int num_available = 0;
                            for (idx in 1:num_tied) {
                                if (idx != idx1) {
                                    num_available += 1;
                                    available_indices[num_available] = idx;
                                }
                            }
                            int max_comparisons = min(pairwise_cap_per_item, num_available);
                            for (comp_idx in 1:max_comparisons) {
                                real u = uniform_rng(0, 1);
                                int sample_idx = 1 + to_int(floor(u * (num_available - comp_idx + 1)));
                                if (sample_idx > num_available - comp_idx + 1) sample_idx = num_available - comp_idx + 1;
                                int idx2 = available_indices[sample_idx];
                                int item2 = tied_items[idx2];
                                real u1 = uniform_rng(0, 1);
                                real g1 = -log(-log(u1));
                                real util1 = test_base_scores[ij_idx, item1] / temperature + g1;
                                real u2 = uniform_rng(0, 1);
                                real g2 = -log(-log(u2));
                                real util2 = test_base_scores[ij_idx, item2] / temperature + g2;
                                int order = (util1 > util2) ? 1 : 2;
                                num_test_pairwise_rankings += 1;
                                test_pairwise_items[num_test_pairwise_rankings, 1] = item1;
                                test_pairwise_items[num_test_pairwise_rankings, 2] = item2;
                                test_pairwise_orders[num_test_pairwise_rankings] = order;
                                test_pairwise_annotator[num_test_pairwise_rankings] = j;
                                test_pairwise_attribute[num_test_pairwise_rankings] = i;
                                test_pairwise_tied_rating[num_test_pairwise_rankings] = rating_val;
                                test_pairwise_observed[num_test_pairwise_rankings] = 1;
                                available_indices[sample_idx] = available_indices[num_available - comp_idx + 1];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // ===== DEBUG =====
    if (DEBUG_PRINT == 1) {
        print("kappa=", kappa);
        print("rating_probs[1]=", rating_probs[1]);
        print("rating_cumprobs[1]=", rating_cumprobs[1]);
        print("rating_thresholds_z[1]=", rating_thresholds_z[1]);
    }

    if (DEBUG_PRINT == 1) {
        int max_i = (I < 1) ? I : 1;
        int max_j = (J < 2) ? J : 2;
        int max_k_train = (K_train < 10) ? K_train : 10;
        int max_k_val = (K_val < 10) ? K_val : 10;
        int max_k_test = (K_test < 30) ? K_test : 30;
        for (i in 1:max_i) {
            for (j in 1:max_j) {
                int ij_idx = (i-1)*J + j;
                for (k in 1:max_k_train) {
                    print("[DEBUG TRAIN POST] i=", i, " j=", j, " k=", k,
                          " probs=", train_posterior_rating_probs[ij_idx, k]);
                }
                for (k in 1:max_k_val) {
                    print("[DEBUG VAL POST]   i=", i, " j=", j, " k=", k,
                          " probs=", val_posterior_rating_probs[ij_idx, k]);
                }
                for (k in 1:max_k_test) {
                    print("[DEBUG TEST POST]  i=", i, " j=", j, " k=", k,
                          " probs=", test_posterior_rating_probs[ij_idx, k]);
                }
            }
        }
    }

    // ===== COMPUTE COUNTS =====
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

    num_val_observed_ratings = 0;
    num_val_missing_ratings = 0;
    for (i in 1:(I*J)) {
        for (k in 1:K_val) {
            if (val_rating_observed[i, k] == 1) {
                num_val_observed_ratings += 1;
            } else {
                num_val_missing_ratings += 1;
            }
        }
    }
    
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
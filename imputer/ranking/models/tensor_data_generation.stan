data {
    // Dimensions
    int<lower=1> K_train;  // number of items in training instance
    int<lower=1> K_test;   // number of items in test instance
    int<lower=1> I;        // number of criteria (attributes)
    int<lower=1> J;         // number of annotators
    int<lower=1> D;         // CP rank (number of components)
    int<lower=1> C;         // number of rating categories
    int<lower=1> d_annotator;  // kept for pipeline compatibility (should equal D)

    // Observation protocol controls
    int<lower=0, upper=1> enable_pairwise_rankings;  // kept for compatibility (pairwise not generated)
    int<lower=0> pairwise_cap_per_item;     // kept for compatibility

    // Hyperparameters
    real<lower=0> sigma_annotator;    // unused in CP model, kept for pipeline compatibility
    real<lower=0> sigma_measurement;  // measurement noise std
    real<lower=0> alpha_dirichlet;    // Dirichlet concentration for global thresholds
    real<lower=0> temperature;        // unused, kept for pipeline compatibility
    real<lower=0> factor_decay;       // T_d = factor_decay^(d-1), controls rank structure

    // Annotator model flags (unused in CP model, kept for pipeline compatibility)
    int<lower=0, upper=1> use_factored_annotator;
    int<lower=0, upper=1> derive_thresholds_from_annotator;

    // Misspecification flags (for testing Stan robustness)
    int<lower=0, upper=1> use_log_scores;     // Apply log() to raw CP scores before binning
    int<lower=0, upper=1> use_logistic_link;   // Use inv_logit instead of Phi for binning

    // Loading distribution: 0 = Exp(1) (default, all positive), 1 = N(0,1) (signed, cancellation)
    int<lower=0, upper=1> use_normal_loadings;
}

generated quantities {

    // ===== DEBUG =====
    int DEBUG_PRINT;
    DEBUG_PRINT = 1;

    // ===== CP DECOMPOSITION =====
    // Z_ijk = sum_d v_id * u_jd * e_kd * T_d
    // v_id, u_jd, e_kd ~ Exp(1) (always positive, no cancellation)
    // T_d = factor_decay^(d-1) (first component strongest)

    // Component weights
    vector[D] T_weights;

    // Factor loadings
    matrix[I, D] v_loadings;       // attribute loadings
    matrix[J, D] u_loadings;       // annotator loadings

    // ===== SHARED COMPONENTS (same variable names as old model for pipeline compatibility) =====
    matrix[I, D] mean_preferences;              // = v_loadings (attribute loadings)
    matrix[J, d_annotator] annotator_embeddings; // = u_loadings (annotator loadings)
    array[I] matrix[d_annotator, D] attr_transforms; // dummy (not used in CP model)
    matrix[I*J, D] annotator_preferences;       // = v_i .* u_j .* T (effective preference vector)
    array[I*J] simplex[C] rating_probs;         // Global rating probabilities (shared across all i,j)
    array[I*J] vector[C] rating_cumprobs;       // Cumulative probabilities
    array[I*J] vector[C+1] rating_thresholds_z; // Z-cutpoints

    // Threshold transform (dummy, kept for compatibility)
    matrix[C-1, d_annotator] threshold_transform_W;
    matrix[I, C-1] threshold_attr_bias;

    // ===== TRAINING INSTANCE =====
    matrix[K_train, D] train_embeddings;       // e_train loadings
    matrix[I*J, K_train] train_base_scores;    // Z_ijk_train
    array[I*J, K_train] int train_rating_values;
    array[I*J, K_train] int train_rating_observed;

    // ===== TEST INSTANCE =====
    matrix[K_test, D] test_embeddings;
    matrix[I*J, K_test] test_base_scores;
    array[I*J, K_test] int test_rating_values;
    array[I*J, K_test] int test_rating_observed;

    // ===== PAIRWISE RANKINGS (size-1 placeholders to avoid K^2 memory blow-up) =====
    array[1, 2] int train_pairwise_items;
    array[1] int train_pairwise_orders;
    array[1] int train_pairwise_annotator;
    array[1] int train_pairwise_attribute;
    array[1] int train_pairwise_tied_rating;
    array[1] int train_pairwise_observed;

    array[1, 2] int test_pairwise_items;
    array[1] int test_pairwise_orders;
    array[1] int test_pairwise_annotator;
    array[1] int test_pairwise_attribute;
    array[1] int test_pairwise_tied_rating;
    array[1] int test_pairwise_observed;

    int num_train_pairwise_rankings;
    int num_test_pairwise_rankings;
    int num_train_observed_ratings;
    int num_test_observed_ratings;
    int num_train_missing_ratings;
    int num_test_missing_ratings;

    // Posterior rating probabilities
    array[I*J, K_train] vector[C] train_posterior_rating_probs;
    array[I*J, K_test] vector[C] test_posterior_rating_probs;

    // ===== INITIALIZE PAIRWISE (no generation, just placeholders) =====
    num_train_pairwise_rankings = 0;
    num_test_pairwise_rankings = 0;
    train_pairwise_items[1, 1] = 0;
    train_pairwise_items[1, 2] = 0;
    train_pairwise_orders[1] = 0;
    train_pairwise_annotator[1] = 0;
    train_pairwise_attribute[1] = 0;
    train_pairwise_tied_rating[1] = 0;
    train_pairwise_observed[1] = 0;
    test_pairwise_items[1, 1] = 0;
    test_pairwise_items[1, 2] = 0;
    test_pairwise_orders[1] = 0;
    test_pairwise_annotator[1] = 0;
    test_pairwise_attribute[1] = 0;
    test_pairwise_tied_rating[1] = 0;
    test_pairwise_observed[1] = 0;

    // ===== INITIALIZE DUMMY COMPATIBILITY ARRAYS =====
    for (c in 1:(C-1)) {
        for (d in 1:d_annotator) {
            threshold_transform_W[c, d] = 0;
        }
    }
    for (i in 1:I) {
        for (c in 1:(C-1)) {
            threshold_attr_bias[i, c] = 0;
        }
        for (r in 1:d_annotator) {
            for (c2 in 1:D) {
                attr_transforms[i, r, c2] = 0;
            }
        }
    }

    // ===== GENERATE COMPONENT WEIGHTS =====
    for (d in 1:D) {
        T_weights[d] = pow(factor_decay, d - 1);
    }

    if (DEBUG_PRINT == 1) {
        print("CP Tensor Decomposition: Z_ijk = sum_d v_id * u_jd * e_kd * T_d");
        print("  D (rank) = ", D);
        print("  factor_decay = ", factor_decay);
        print("  T_weights = ", T_weights);
        print("  use_log_scores = ", use_log_scores);
        print("  use_logistic_link = ", use_logistic_link);
    }

    // ===== GENERATE ATTRIBUTE LOADINGS v_i =====
    {
        for (i in 1:I) {
            for (d in 1:D) {
                shared_v[d] = (use_normal_loadings == 1) ? normal_rng(0, 1) : exponential_rng(1.0);
            }
            for (i in 1:I) {
                v_loadings[i] = shared_v;
            }
        } else {
            for (i in 1:I) {
                for (d in 1:D) {
                    v_loadings[i, d] = (use_normal_loadings == 1) ? normal_rng(0, 1) : exponential_rng(1.0);
                }
            }
        }
        mean_preferences = v_loadings;
    }

    // ===== GENERATE ANNOTATOR LOADINGS u_j =====
    {
        for (j in 1:J) {
            for (d in 1:D) {
                shared_u[d] = (use_normal_loadings == 1) ? normal_rng(0, 1) : exponential_rng(1.0);
            }
            for (j in 1:J) {
                u_loadings[j] = shared_u;
            }
        } else {
            for (j in 1:J) {
                for (d in 1:D) {
                    u_loadings[j, d] = (use_normal_loadings == 1) ? normal_rng(0, 1) : exponential_rng(1.0);
                }
            }
        }
        // Store in annotator_embeddings for pipeline compatibility
        // Note: d_annotator should equal D for this model
        for (j in 1:J) {
            for (d in 1:d_annotator) {
                if (d <= D) {
                    annotator_embeddings[j, d] = u_loadings[j, d];
                } else {
                    annotator_embeddings[j, d] = 0;
                }
            }
        }
    }

    // ===== COMPUTE ANNOTATOR PREFERENCES (effective preference vectors) =====
    // annotator_preferences[ij] = v_i .* u_j .* T
    // So that base_score = dot_product(annotator_preferences[ij], e_k) = Z_ijk
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (d in 1:D) {
                annotator_preferences[idx, d] = v_loadings[i, d] * u_loadings[j, d] * T_weights[d];
            }
        }
    }

    if (DEBUG_PRINT == 1) {
        print("v_loadings[1] = ", v_loadings[1]);
        print("u_loadings[1] = ", u_loadings[1]);
        print("annotator_preferences[1] = ", annotator_preferences[1]);
    }

    // ===== GENERATE PER-(i,j) THRESHOLDS =====
    // Each (attribute, annotator) pair gets an independent Dirichlet sample.
    // This maximizes threshold variability — same annotator rates different
    // attributes with different scales.
    {
        // Full (i,j) dependence — independent Dirichlet for each pair
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                rating_probs[idx] = dirichlet_rng(rep_vector(alpha_dirichlet / C, C));
                rating_cumprobs[idx] = cumulative_sum(rating_probs[idx]);
            }
        }
    }

    // Convert cumulative probabilities to cutpoints
    // Probit link: z = inv_Phi(p)   -> cdf = Phi(z)
    // Logistic link: z = logit(p)   -> cdf = inv_logit(z)
    for (ij in 1:(I*J)) {
        rating_thresholds_z[ij][1] = negative_infinity();
        for (c in 2:C) {
            if (use_logistic_link == 1) {
                rating_thresholds_z[ij][c] = logit(rating_cumprobs[ij][c-1]);
            } else {
                rating_thresholds_z[ij][c] = inv_Phi(rating_cumprobs[ij][c-1]);
            }
        }
        rating_thresholds_z[ij][C+1] = positive_infinity();
    }

    if (DEBUG_PRINT == 1) {
        for (j in 1:min(3, J)) {
            int idx = j;  // i=1
            print("Annotator ", j, " rating_probs = ", rating_probs[idx]);
            print("Annotator ", j, " thresholds_z = ", rating_thresholds_z[idx]);
        }
    }

    // ===== GENERATE TRAINING ITEM EMBEDDINGS =====
    for (k in 1:K_train) {
        for (d in 1:D) {
            shared_e[d] = (use_normal_loadings == 1) ? normal_rng(0, 1) : exponential_rng(1.0);
        }
        for (k in 1:K_train) {
            train_embeddings[k] = shared_e;
        }
    } else {
        for (k in 1:K_train) {
            for (d in 1:D) {
                train_embeddings[k, d] = (use_normal_loadings == 1) ? normal_rng(0, 1) : exponential_rng(1.0);
            }
        }
    }

    // Compute training base scores: z_ijk = v_ij · e_k
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K_train) {
                train_base_scores[idx, k] = dot_product(annotator_preferences[idx], train_embeddings[k]);
            }
        }
    }

    // ===== GENERATE TEST ITEM EMBEDDINGS =====
    for (k in 1:K_test) {
        for (d in 1:D) {
            shared_e_test[d] = (use_normal_loadings == 1) ? normal_rng(0, 1) : exponential_rng(1.0);
        }
        for (k in 1:K_test) {
            test_embeddings[k] = shared_e_test;
        }
    } else {
        for (k in 1:K_test) {
            for (d in 1:D) {
                test_embeddings[k, d] = (use_normal_loadings == 1) ? normal_rng(0, 1) : exponential_rng(1.0);
            }
        }
    }

    // Compute test base scores: z_ijk = v_ij · e_k
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
                real noisy_score = train_base_scores[idx, k] + normal_rng(0, sigma_measurement);
                real pref_norm_sq = dot_self(annotator_preferences[idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real standardized_score = noisy_score / total_std;
                real cdf_val;
                if (use_logistic_link == 1) {
                    cdf_val = inv_logit(standardized_score);
                } else {
                    cdf_val = Phi(standardized_score);
                }

                int rating = 1;
                for (c in 1:C) {
                    if (cdf_val <= rating_cumprobs[idx][c]) {
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
                real noisy_score = test_base_scores[idx, k] + normal_rng(0, sigma_measurement);
                real pref_norm_sq = dot_self(annotator_preferences[idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real standardized_score = noisy_score / total_std;
                real cdf_val;
                if (use_logistic_link == 1) {
                    cdf_val = inv_logit(standardized_score);
                } else {
                    cdf_val = Phi(standardized_score);
                }

                int rating = 1;
                for (c in 1:C) {
                    if (cdf_val <= rating_cumprobs[idx][c]) {
                        rating = c;
                        break;
                    }
                }
                test_rating_values[idx, k] = rating;
                test_rating_observed[idx, k] = 0;  // Initialize as missing
            }
        }
    }

    // ===== COMPUTE POSTERIOR RATING PROBABILITIES =====
    // For logistic link, posterior uses logistic CDF instead of Phi.
    // The posterior P(rating=c | base_score) integrates over measurement noise.
    // For probit: P(z < t) = Phi((t - mean)/cond_std)
    // For logistic: no clean closed form for integral of logistic over Gaussian noise,
    //   so we approximate with the same Gaussian integral (Phi) — this is fine since
    //   the posterior probs are only used as ground truth for evaluation, not for training.

    // Training
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

    // Test
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

    // ===== IMPLEMENT OBSERVATION PROTOCOL =====
    // Same as old model: training uses first 2/3 annotators, test uses last 2/3
    int train_annotator_start = 1;
    int train_annotator_end = (2 * J) / 3;
    int test_annotator_start = J / 3 + 1;
    int test_annotator_end = J;

    // Training observation protocol
    for (k in 1:K_train) {
        if ((train_annotator_end - train_annotator_start + 1) < 4) {
            reject("Number of available training annotators is less than 4");
        }
        array[4] int selected_annotators;
        int num_selected = 0;

        while (num_selected < 4) {
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
            for (s in 1:4) {
                int j = selected_annotators[s];
                int idx = (i-1)*J + j;
                train_rating_observed[idx, k] = 1;
            }
        }
    }

    // Test observation protocol
    for (k in 1:K_test) {
        if ((test_annotator_end - test_annotator_start + 1) < 4) {
            reject("Number of available test annotators is less than 4");
        }
        array[4] int selected_annotators;
        int num_selected = 0;

        while (num_selected < 4) {
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
            for (s in 1:4) {
                int j = selected_annotators[s];
                int idx = (i-1)*J + j;
                test_rating_observed[idx, k] = 1;
            }
        }
    }

    // ===== DEBUG =====
    if (DEBUG_PRINT == 1) {
        int max_i = (I < 1) ? I : 1;
        int max_j = (J < 2) ? J : 2;
        int max_k_train = (K_train < 5) ? K_train : 5;
        int max_k_test = (K_test < 5) ? K_test : 5;
        for (i in 1:max_i) {
            for (j in 1:max_j) {
                int ij_idx = (i-1)*J + j;
                for (k in 1:max_k_train) {
                    print("[DEBUG TRAIN] i=", i, " j=", j, " k=", k,
                          " base_score=", train_base_scores[ij_idx, k],
                          " rating=", train_rating_values[ij_idx, k],
                          " posterior=", train_posterior_rating_probs[ij_idx, k]);
                }
                for (k in 1:max_k_test) {
                    print("[DEBUG TEST]  i=", i, " j=", j, " k=", k,
                          " base_score=", test_base_scores[ij_idx, k],
                          " rating=", test_rating_values[ij_idx, k],
                          " posterior=", test_posterior_rating_probs[ij_idx, k]);
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

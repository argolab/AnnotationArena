data {
    // Dimensions
    int<lower=1> K_train;  // number of items in training instance
    int<lower=1> K_test;   // number of items in test instance
    int<lower=1> I;        // number of criteria (attributes)
    int<lower=1> J;        // number of annotators  
    int<lower=1> D;        // embedding dimension (kept for compatibility; not used for generation)
    int<lower=1> C;        // number of rating categories

    // Prototype/style dimensions for Version A generator
    int<lower=1> M;        // number of item prototypes
    int<lower=1> S;        // number of annotator styles
    
    // Observation protocol controls
    int<lower=0, upper=1> enable_pairwise_rankings;  // Ablation: enable pairwise rankings
    int<lower=0> pairwise_cap_per_item;              // Max comparisons per item within its tied group
    
    // Hyperparameters
    real<lower=0> sigma_annotator;
    real<lower=0> sigma_measurement;
    real<lower=0> kappa;
    real<lower=0> temperature;

}

generated quantities {

    // ===== DEBUG / CONSTANTS =====
    int DEBUG_PRINT;
    DEBUG_PRINT = 1;

    // Rubric structure hyperparameters (Version A knobs)
    real sigma_proto = 1.0;   // prototype separability (bigger → easier transfer)
    real sigma_style = 0.6;   // style strength (bigger → easier style learning)
    real sigma_delta = 0.2;   // residual interaction scale  δ_{i,m,s} (bigger → harder transfer)
    real sigma_rubric_fuzz = 0.2; // small latent rubric fuzz at the time of retrieving score.

    // ===== SHARED COMPONENTS (same for train and test) =====

    // Rating thresholds (used for ordinal probit binning)
    // NOTE: for Version A these thresholds are defined in the *score* space
    // of noisy_score = μ_{i,z_k,s_j} + rubric_fuzz + measurement_noise.
    array[I*J] simplex[C] rating_probs;        // Rating probabilities p_ij
    array[I*J] vector[C] rating_cumprobs;      // Cumulative probabilities q_ij = cumsum(p_ij) in (0,1]
    array[I*J] vector[C+1] rating_thresholds_z; // Score cutpoints: [-inf, t_1, ..., t_{C-1}, +inf]

    // Version A latent structure (prototype–style rubric ground truth)
    array[K_train] int z_train;  // item prototype for each training item (1..M)
    array[K_test]  int z_test;   // item prototype for each test item (1..M)
    array[J] int s_of_j;         // annotator style index (1..S)

    array[I] real a_attr;        // criterion bias a_i
    array[M] real u_proto;       // prototype quality u_m
    array[S] real v_style;       // style bias v_s

    // Flattened over (i,m,s): index = ((i-1)*M + (m-1))*S + s
    array[I*M*S] real delta_ims; // residual interaction δ_{i,m,s}
    array[I*M*S] real mu_ims;    // full rubric means μ_{i,m,s}

    // ===== TRAINING INSTANCE =====
    matrix[I*J, K_train] train_base_scores;    // base_score_{i,j,k} from rubric table
    array[I*J, K_train] int train_rating_values;
    array[I*J, K_train] int train_rating_observed;
    
    // ===== TEST INSTANCE =====
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

    // Test pairwise rankings
    array[I*J*C*(K_test*(K_test-1)%/%2), 2] int test_pairwise_items;
    array[I*J*C*(K_test*(K_test-1)%/%2)] int test_pairwise_orders;
    array[I*J*C*(K_test*(K_test-1)%/%2)] int test_pairwise_annotator;
    array[I*J*C*(K_test*(K_test-1)%/%2)] int test_pairwise_attribute;
    array[I*J*C*(K_test*(K_test-1)%/%2)] int test_pairwise_tied_rating;
    array[I*J*C*(K_test*(K_test-1)%/%2)] int test_pairwise_observed;

    // Counts
    int num_train_pairwise_rankings;
    int num_test_pairwise_rankings;
    int num_train_observed_ratings;
    int num_test_observed_ratings;
    int num_train_missing_ratings;
    int num_test_missing_ratings;
    
    // Posterior rating probabilities (due to measurement error)
    array[I*J, K_train] vector[C] train_posterior_rating_probs;
    array[I*J, K_test]  vector[C] test_posterior_rating_probs;

    // ===== GENERATE RUBRIC STRUCTURE =====
    {
        // Criterion biases a_i
        for (i in 1:I) {
            a_attr[i] = normal_rng(0, 1);
        }

        // Prototype qualities u_m
        for (m in 1:M) {
            u_proto[m] = normal_rng(0, sigma_proto);
        }

        // Style biases v_s
        for (s in 1:S) {
            v_style[s] = normal_rng(0, sigma_style);
        }

        // Residual interactions δ_{i,m,s} and construct μ_{i,m,s}
        {
            int idx;
            for (i in 1:I) {
                for (m in 1:M) {
                    for (s in 1:S) {
                        idx = ((i-1) * M + (m-1)) * S + s;
                        delta_ims[idx] = normal_rng(0, sigma_delta);
                        mu_ims[idx] = a_attr[i] + u_proto[m] + v_style[s] + delta_ims[idx];
                    }
                }
            }
        }
    }

    // ===== ASSIGN PROTOTYPES AND STYLES (FULL I/J/K DEPENDENCE) =====
    {
        // Item prototypes for train and test
        for (k in 1:K_train) {
            z_train[k] = categorical_rng(rep_vector(1.0 / M, M));
        }
        for (k in 1:K_test) {
            z_test[k] = categorical_rng(rep_vector(1.0 / M, M));
        }

        // Annotator styles
        for (j in 1:J) {
            s_of_j[j] = categorical_rng(rep_vector(1.0 / S, S));
        }
    }

    // ===== GENERATE RATING THRESHOLDS VIA GMM QUANTILES (FULL I,J DEPENDENCE) =====
    //
    // For a fixed (i,j), style s = s_of_j[j] is fixed. Items have prototypes z_k ∈ {1..M}
    // with approx. uniform weights, and we add rubric fuzz + measurement noise:
    //
    //   noisy_score | (i,j,m) ~ Normal(μ_{i,m,s}, sigma_tot^2),
    //   sigma_tot^2 = sigma_rubric_fuzz^2 + sigma_measurement^2
    //
    // Across items, noisy_score is a mixture of M Gaussians with equal weights.
    // For each (i,j), we choose thresholds t_c such that the mixture CDF matches
    // the cumulative probabilities rating_cumprobs[idx][c].
    {
        real sigma_tot = sqrt(sigma_rubric_fuzz * sigma_rubric_fuzz +
                              sigma_measurement * sigma_measurement);

        // First set rating_probs / rating_cumprobs (here: uniform over classes)
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1) * J + j;
                rating_probs[idx] = rep_vector(1.0 / C, C);
                rating_cumprobs[idx] = cumulative_sum(rating_probs[idx]);
            }
        }

        // Now compute thresholds per (i,j) using bisection on the mixture CDF
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1) * J + j;
                int s = s_of_j[j];

                rating_thresholds_z[idx][1] = negative_infinity();

                for (c in 2:C) {
                    real target = rating_cumprobs[idx][c-1];  // desired CDF mass
                    real lo = -10;
                    real hi =  10;

                    // Bisection iterations
                    for (iter in 1:40) {
                        real mid = 0.5 * (lo + hi);
                        real F = 0;

                        // Mixture CDF over prototypes m
                        for (m in 1:M) {
                            int idx_ims = ((i-1) * M + (m-1)) * S + s;
                            real mu = mu_ims[idx_ims];
                            F += Phi((mid - mu) / sigma_tot);
                        }
                        F /= M;

                        if (F < target) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }

                    rating_thresholds_z[idx][c] = 0.5 * (lo + hi);
                }

                rating_thresholds_z[idx][C+1] = positive_infinity();
            }
        }
    }

    // ===== COMPUTE BASE SCORES FROM RUBRIC TABLE =====
    {
        int idx_ims;
        for (i in 1:I) {
            for (j in 1:J) {
                int ij_idx = (i-1) * J + j;
                int s = s_of_j[j];

                // Training items
                for (k in 1:K_train) {
                    int m = z_train[k];
                    idx_ims = ((i-1) * M + (m-1)) * S + s;
                    train_base_scores[ij_idx, k] =
                        mu_ims[idx_ims] + normal_rng(0, sigma_rubric_fuzz);
                }

                // Test items
                for (k in 1:K_test) {
                    int m = z_test[k];
                    idx_ims = ((i-1) * M + (m-1)) * S + s;
                    test_base_scores[ij_idx, k] =
                        mu_ims[idx_ims] + normal_rng(0, sigma_rubric_fuzz);
                }
            }
        }
    }

    // ===== GENERATE TRAINING RATINGS (ORDINAL PROBIT, SCORE-SPACE THRESHOLDS) =====
    // For each (i,j,k), we draw:
    //   base_score = μ_{i,z_k,s_j} + rubric_fuzz
    //   noisy_score = base_score + measurement_noise
    // then bin noisy_score directly against the score-space thresholds.
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1) * J + j;
            for (k in 1:K_train) {
                real base_score = train_base_scores[idx, k];
                real noisy_score = base_score + normal_rng(0, sigma_measurement);
                int rating = 1;
                for (c in 1:C) {
                    if (noisy_score <= rating_thresholds_z[idx][c+1]) {
                        rating = c;
                        break;
                    }
                }
                train_rating_values[idx, k] = rating;
                train_rating_observed[idx, k] = 0;  // initialize as missing
            }
        }
    }

    // ===== GENERATE TEST RATINGS =====
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1) * J + j;
            for (k in 1:K_test) {
                real base_score = test_base_scores[idx, k];
                real noisy_score = base_score + normal_rng(0, sigma_measurement);
                int rating = 1;
                for (c in 1:C) {
                    if (noisy_score <= rating_thresholds_z[idx][c+1]) {
                        rating = c;
                        break;
                    }
                }
                test_rating_values[idx, k] = rating;
                test_rating_observed[idx, k] = 0;  // initialize as missing
            }
        }
    }

    // ===== COMPUTE POSTERIOR RATING PROBABILITIES =====
    // For a fixed item k with known base_score, the conditional distribution is:
    //   noisy_score | base_score ~ N(base_score, sigma_measurement^2)
    // We compute P(y=c | base_score) using the score-space thresholds.

    // Training posterior rating probabilities
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1) * J + j;
            for (k in 1:K_train) {
                real true_base_score = train_base_scores[idx, k];
                real mean = true_base_score;
                real cond_std = sigma_measurement;

                if (cond_std == 0) {
                    int one_c = 1;
                    for (c in 1:C) {
                        if (mean <= rating_thresholds_z[idx][c+1]) {
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
                        real upper_prob = Phi((upper_z - mean) / cond_std);
                        real lower_prob = Phi((lower_z - mean) / cond_std);
                        train_posterior_rating_probs[idx, k][c] = upper_prob - lower_prob;
                    }
                }
            }
        }
    }
    
    // Test posterior rating probabilities
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1) * J + j;
            for (k in 1:K_test) {
                real true_base_score = test_base_scores[idx, k];
                real mean = true_base_score;
                real cond_std = sigma_measurement;

                if (cond_std == 0) {
                    int one_c = 1;
                    for (c in 1:C) {
                        if (mean <= rating_thresholds_z[idx][c+1]) {
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
                        real upper_prob = Phi((upper_z - mean) / cond_std);
                        real lower_prob = Phi((lower_z - mean) / cond_std);
                        test_posterior_rating_probs[idx, k][c] = upper_prob - lower_prob;
                    }
                }
            }
        }
    }

    // ===== OBSERVATION PROTOCOL (UNCHANGED) =====
    {
        int train_annotator_start = 1;
        int train_annotator_end = (2 * J) / 3;
        int test_annotator_start = J / 3 + 1;
        int test_annotator_end = J;

        // Training instance observation protocol
        if ((train_annotator_end - train_annotator_start + 1) >= 4) {
            for (k in 1:K_train) {
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
                        int idx = (i-1) * J + j;
                        train_rating_observed[idx, k] = 1;
                    }
                }
            }
        }

        // Test instance observation protocol
        if ((test_annotator_end - test_annotator_start + 1) >= 4) {
            for (k in 1:K_test) {
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
                        int idx = (i-1) * J + j;
                        test_rating_observed[idx, k] = 1;
                    }
                }
            }
        }
    }

    // ===== GENERATE PAIRWISE RANKINGS FROM TIED RATINGS (UNCHANGED, NEW BASE SCORES) =====
    num_train_pairwise_rankings = 0;
    num_test_pairwise_rankings = 0;

    if (enable_pairwise_rankings == 1) {
        int train_annotator_start = 1;
        int train_annotator_end = (2 * J) / 3;
        int test_annotator_start = J / 3 + 1;
        int test_annotator_end = J;

        // Training pairwise rankings
        for (i in 1:I) {
            for (j in train_annotator_start:train_annotator_end) {
                int ij_idx = (i-1) * J + j;

                for (rating_val in 1:C) {
                    array[K_train] int tied_items;
                    int num_tied = 0;
                    
                    for (k in 1:K_train) {
                        if (train_rating_values[ij_idx, k] == rating_val &&
                            train_rating_observed[ij_idx, k] == 1) {
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
                                if (sample_idx > num_available - comp_idx + 1)
                                    sample_idx = num_available - comp_idx + 1;
                                
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
                                
                                available_indices[sample_idx] =
                                    available_indices[num_available - comp_idx + 1];
                            }
                        }
                    }
                }
            }
        }

        // Test pairwise rankings
        for (i in 1:I) {
            for (j in test_annotator_start:test_annotator_end) {
                int ij_idx = (i-1) * J + j;
                
                for (rating_val in 1:C) {
                    array[K_test] int tied_items;
                    int num_tied = 0;
                    
                    for (k in 1:K_test) {
                        if (test_rating_values[ij_idx, k] == rating_val &&
                            test_rating_observed[ij_idx, k] == 1) {
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
                                if (sample_idx > num_available - comp_idx + 1)
                                    sample_idx = num_available - comp_idx + 1;
                                
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
                                
                                available_indices[sample_idx] =
                                    available_indices[num_available - comp_idx + 1];
                            }
                        }
                    }
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

    // ===== OPTIONAL DEBUG PRINTS =====
    if (DEBUG_PRINT == 1) {
        print("sigma_proto=", sigma_proto,
              " sigma_style=", sigma_style,
              " sigma_delta=", sigma_delta,
              " sigma_measurement=", sigma_measurement);
        print("a_attr[1]=", a_attr[1]);
        print("u_proto=", u_proto);
        print("v_style=", v_style);
        print("z_train[1]=", z_train[1], " z_test[1]=", z_test[1]);
        print("s_of_j[1]=", s_of_j[1]);
        print("rating_probs[1]=", rating_probs[1]);
        print("rating_cumprobs[1]=", rating_cumprobs[1]);
        print("rating_thresholds_z[1]=", rating_thresholds_z[1]);
    }
}


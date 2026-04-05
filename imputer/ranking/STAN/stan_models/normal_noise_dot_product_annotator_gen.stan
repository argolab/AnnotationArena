data {
    // Dimensions
    int<lower=1> K;          // number of shared items (same across all instances)
    int<lower=1> J;          // total number of annotators (J_train + J_val + J_test)
    int<lower=1> J_train;    // number of training annotators  (IDs 1..J_train)
    int<lower=1> J_val;      // number of validation annotators (IDs J_train+1..J_train+J_val)
    int<lower=1> J_test;     // number of test annotators      (IDs J_train+J_val+1..J)
    int<lower=1> I;          // number of criteria (attributes)
    int<lower=1> D;          // embedding dimension
    int<lower=1> C;          // number of rating categories
    int<lower=1> d_annotator; // annotator embedding dimension

    // Hyperparameters
    real<lower=0> sigma_annotator;
    real<lower=0> sigma_measurement;
    real<lower=0> kappa;
    real<lower=0> temperature;

    // Annotator model selection (identical semantics to normal_noise_dot_product_generation.stan)
    int<lower=0, upper=1> use_factored_annotator;
    int<lower=0, upper=1> derive_thresholds_from_annotator;
}

generated quantities {

    int DEBUG_PRINT;
    DEBUG_PRINT = 1;

    // ===== SHARED GENERATIVE COMPONENTS =====
    matrix[I, D] mean_preferences;               // Global criteria embeddings v_i
    matrix[J, d_annotator] annotator_embeddings;  // Annotator embeddings u_j (all J)
    array[I] matrix[d_annotator, D] attr_transforms; // Attribute-specific transforms M_i
    matrix[I*J, D] annotator_preferences;         // V_ij for all (i,j)
    array[I*J] simplex[C] rating_probs;
    array[I*J] vector[C] rating_cumprobs;
    array[I*J] vector[C+1] rating_thresholds_z;

    matrix[C-1, d_annotator] threshold_transform_W;
    matrix[I, C-1] threshold_attr_bias;

    // ===== SHARED ITEM SET =====
    matrix[K, D] embeddings;       // Item embeddings e_k  (shared across all instances)
    matrix[I*J, K] base_scores;    // z_ijk = v_ij . e_k  (for all annotators x items)

    // ===== RATINGS (all annotators x all items) =====
    // Shape [I*J, K]; instance membership is determined by annotator ID in Python.
    array[I*J, K] int rating_values;
    array[I*J, K] int rating_observed;  // always 1: every annotator rates every item

    // ===== POSTERIOR RATING PROBABILITIES =====
    // Split by annotator group for downstream convenience.
    array[I*J_train, K] vector[C] train_posterior_rating_probs;
    array[I*J_val,   K] vector[C] val_posterior_rating_probs;
    array[I*J_test,  K] vector[C] test_posterior_rating_probs;

    // ===== COUNTS =====
    int num_train_observed_ratings;
    int num_val_observed_ratings;
    int num_test_observed_ratings;

    // ===== GENERATE SHARED ANNOTATOR COMPONENTS =====
    real sigma_M = sigma_annotator / sqrt(d_annotator);

    {
        for (i in 1:I) {
            for (d in 1:D) {
                mean_preferences[i, d] = normal_rng(0, 1);
            }
        }

        if (use_factored_annotator == 1) {
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

    for (ij in 1:(I*J)) {
        rating_thresholds_z[ij][1] = negative_infinity();
        for (c in 2:C) {
            rating_thresholds_z[ij][c] = inv_Phi(rating_cumprobs[ij][c-1]);
        }
        rating_thresholds_z[ij][C+1] = positive_infinity();
    }

    // ===== GENERATE SHARED ITEM EMBEDDINGS AND BASE SCORES =====
    for (k in 1:K) {
        for (d in 1:D) {
            embeddings[k, d] = normal_rng(0, 1);
        }
    }
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], embeddings[k]);
            }
        }
    }

    // ===== GENERATE RATINGS: every annotator rates every item =====
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                rating_observed[idx, k] = 1;
                real noisy_score = base_scores[idx, k] + normal_rng(0, sigma_measurement);
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
                rating_values[idx, k] = rating;
            }
        }
    }

    // ===== COMPUTE POSTERIOR RATING PROBABILITIES (split by annotator group) =====

    // Training annotators: 1..J_train
    for (i in 1:I) {
        for (j in 1:J_train) {
            int global_idx = (i-1)*J + j;           // index into full [I*J, K] arrays
            int split_idx  = (i-1)*J_train + j;     // index into [I*J_train, K] output
            for (k in 1:K) {
                real true_base_score = base_scores[global_idx, k];
                real pref_norm_sq = dot_self(annotator_preferences[global_idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real mean_std = true_base_score / total_std;
                real cond_std = sigma_measurement / total_std;
                if (cond_std == 0) {
                    int one_c = 1;
                    for (c in 1:C) {
                        if (mean_std <= rating_thresholds_z[global_idx][c+1]) {
                            one_c = c;
                            break;
                        }
                    }
                    for (c in 1:C) {
                        train_posterior_rating_probs[split_idx, k][c] = (c == one_c) ? 1 : 0;
                    }
                } else {
                    for (c in 1:C) {
                        real upper_z = rating_thresholds_z[global_idx][c+1];
                        real lower_z = rating_thresholds_z[global_idx][c];
                        train_posterior_rating_probs[split_idx, k][c] =
                            Phi((upper_z - mean_std) / cond_std) - Phi((lower_z - mean_std) / cond_std);
                    }
                }
            }
        }
    }

    // Validation annotators: J_train+1..J_train+J_val
    for (i in 1:I) {
        for (jv in 1:J_val) {
            int j = J_train + jv;
            int global_idx = (i-1)*J + j;
            int split_idx  = (i-1)*J_val + jv;
            for (k in 1:K) {
                real true_base_score = base_scores[global_idx, k];
                real pref_norm_sq = dot_self(annotator_preferences[global_idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real mean_std = true_base_score / total_std;
                real cond_std = sigma_measurement / total_std;
                if (cond_std == 0) {
                    int one_c = 1;
                    for (c in 1:C) {
                        if (mean_std <= rating_thresholds_z[global_idx][c+1]) {
                            one_c = c;
                            break;
                        }
                    }
                    for (c in 1:C) {
                        val_posterior_rating_probs[split_idx, k][c] = (c == one_c) ? 1 : 0;
                    }
                } else {
                    for (c in 1:C) {
                        real upper_z = rating_thresholds_z[global_idx][c+1];
                        real lower_z = rating_thresholds_z[global_idx][c];
                        val_posterior_rating_probs[split_idx, k][c] =
                            Phi((upper_z - mean_std) / cond_std) - Phi((lower_z - mean_std) / cond_std);
                    }
                }
            }
        }
    }

    // Test annotators: J_train+J_val+1..J
    for (i in 1:I) {
        for (jt in 1:J_test) {
            int j = J_train + J_val + jt;
            int global_idx = (i-1)*J + j;
            int split_idx  = (i-1)*J_test + jt;
            for (k in 1:K) {
                real true_base_score = base_scores[global_idx, k];
                real pref_norm_sq = dot_self(annotator_preferences[global_idx]);
                real total_std = sqrt(pref_norm_sq + sigma_measurement * sigma_measurement);
                real mean_std = true_base_score / total_std;
                real cond_std = sigma_measurement / total_std;
                if (cond_std == 0) {
                    int one_c = 1;
                    for (c in 1:C) {
                        if (mean_std <= rating_thresholds_z[global_idx][c+1]) {
                            one_c = c;
                            break;
                        }
                    }
                    for (c in 1:C) {
                        test_posterior_rating_probs[split_idx, k][c] = (c == one_c) ? 1 : 0;
                    }
                } else {
                    for (c in 1:C) {
                        real upper_z = rating_thresholds_z[global_idx][c+1];
                        real lower_z = rating_thresholds_z[global_idx][c];
                        test_posterior_rating_probs[split_idx, k][c] =
                            Phi((upper_z - mean_std) / cond_std) - Phi((lower_z - mean_std) / cond_std);
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

    // ===== COMPUTE COUNTS =====
    num_train_observed_ratings = 0;
    num_val_observed_ratings = 0;
    num_test_observed_ratings = 0;
    for (i in 1:I) {
        for (j in 1:J_train) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                if (rating_observed[idx, k] == 1) num_train_observed_ratings += 1;
            }
        }
        for (jv in 1:J_val) {
            int j = J_train + jv;
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                if (rating_observed[idx, k] == 1) num_val_observed_ratings += 1;
            }
        }
        for (jt in 1:J_test) {
            int j = J_train + J_val + jt;
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                if (rating_observed[idx, k] == 1) num_test_observed_ratings += 1;
            }
        }
    }
}
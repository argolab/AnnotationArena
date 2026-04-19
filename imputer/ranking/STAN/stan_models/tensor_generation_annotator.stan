/*
 * Tensor model data generator — annotator split.
 *
 * Same generative model as tensor_generation.stan:
 *   z_ijk = sum_t alpha_jt * exp(u_i + v_t + u_it) . e_k
 *
 * Key difference from tensor_generation.stan:
 *   - Items are shared: single set of K items rated by all J annotators.
 *   - Annotators are split: train (1..J_train), val (J_train+1..J_train+J_val),
 *     test (J_train+J_val+1..J).
 *   - Sparsity: only num_annotate_annotator annotators (sampled from ALL J) rate
 *     each item (same random-subsample scheme as tensor_generation.stan).
 *   - No pairwise rankings.
 *   - Posterior arrays are split by annotator group instead of item group.
 */

data {
    int<lower=1> K;           // number of shared items
    int<lower=1> J;           // total annotators (J_train + J_val + J_test)
    int<lower=1> J_train;     // training annotators   (IDs 1..J_train)
    int<lower=1> J_val;       // validation annotators (IDs J_train+1..J_train+J_val)
    int<lower=1> J_test;      // test annotators       (IDs J_train+J_val+1..J)
    int<lower=1> I;           // number of attributes
    int<lower=1> D;           // embedding dimension
    int<lower=1> C;           // number of rating categories
    int<lower=1> T;           // number of annotator prototypes

    int<lower=1> num_annotate_annotator;  // annotators sampled per item

    // Embedding priors
    real<lower=0> sigma_u;
    real<lower=0> sigma_v;
    real<lower=0> sigma_uit;

    // Noise / threshold hyperparameters
    real<lower=0> sigma_measurement;
    real<lower=0> kappa;
    real<lower=1> alpha_confusion;
    real<lower=0> temperature;   // kept for config parity; unused without pairwise

    int<lower=0, upper=1> use_dawid_skene_noise;            // 0=continuous, 1=discrete
    int<lower=0, upper=1> derive_thresholds_from_annotator; // 0=Dirichlet, 1=linear from alpha_j
}

generated quantities {

    int DEBUG_PRINT = 1;

    // ===== TENSOR MODEL PARAMETERS =====
    matrix[I, D] u_attr;               // attribute embeddings u_i
    matrix[T, D] v_proto;              // prototype embeddings v_t
    array[I] matrix[T, D] u_inter;    // attribute-prototype interactions u_it
    array[J] simplex[T] alpha_jt;     // annotator mixing weights
    matrix[I*J, D] eff_pref;          // effective preference vectors for (i,j)

    // Rating threshold parameters
    array[I*J] simplex[C]    rating_probs;
    array[I*J] vector[C]     rating_cumprobs;
    array[I*J] vector[C+1]   rating_thresholds_z;
    matrix[C-1, T] threshold_transform_W;
    matrix[I, C-1] threshold_attr_bias;

    // Confusion matrix (meaningful only when use_dawid_skene_noise=1)
    array[C] simplex[C] confusion_matrix;

    // ===== SHARED ITEM SET =====
    matrix[K, D] embeddings;       // item embeddings e_k
    matrix[I*J, K] base_scores;    // z_ijk = eff_pref[ij] . e_k

    // ===== RATINGS (all annotators x all items) =====
    // rating_observed[idx,k]=1 iff annotator j was selected for item k.
    array[I*J, K] int rating_values;
    array[I*J, K] int rating_observed;

    // ===== POSTERIOR RATING PROBABILITIES (split by annotator group) =====
    array[I*J_train, K] vector[C] train_posterior_rating_probs;
    array[I*J_val,   K] vector[C] val_posterior_rating_probs;
    array[I*J_test,  K] vector[C] test_posterior_rating_probs;

    // ===== COUNTS =====
    int num_train_observed_ratings;
    int num_val_observed_ratings;
    int num_test_observed_ratings;

    // ===== GENERATE TENSOR PARAMETERS =====
    {
        for (i in 1:I)
            for (d in 1:D)
                u_attr[i, d] = normal_rng(0, sigma_u);

        for (t in 1:T)
            for (d in 1:D)
                v_proto[t, d] = normal_rng(0, sigma_v);

        for (i in 1:I)
            for (t in 1:T)
                for (d in 1:D)
                    u_inter[i][t, d] = normal_rng(0, sigma_uit);

        for (j in 1:J)
            alpha_jt[j] = dirichlet_rng(rep_vector(1.0, T));

        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                row_vector[D] pref = rep_row_vector(0.0, D);
                for (t in 1:T) {
                    row_vector[D] log_gate = u_attr[i] + v_proto[t] + u_inter[i][t];
                    pref = pref + alpha_jt[j][t] * exp(log_gate);
                }
                eff_pref[idx] = pref;
            }
        }
    }

    // ===== GENERATE THRESHOLD TRANSFORM =====
    {
        real sigma_W = 1.0 / sqrt(T);
        for (c in 1:(C-1))
            for (t in 1:T)
                threshold_transform_W[c, t] = normal_rng(0, sigma_W);
        for (i in 1:I)
            for (c in 1:(C-1))
                threshold_attr_bias[i, c] = normal_rng(0, 0.1);
    }

    // ===== GENERATE RATING THRESHOLDS =====
    if (derive_thresholds_from_annotator == 1) {
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                vector[C-1] threshold_logits = threshold_transform_W * alpha_jt[j]
                                               + to_vector(threshold_attr_bias[i]);
                vector[C] logits_full;
                logits_full[1] = 0;
                for (c in 2:C)
                    logits_full[c] = threshold_logits[c-1];
                real max_logit = max(logits_full);
                vector[C] exp_logits;
                for (c in 1:C)
                    exp_logits[c] = exp(logits_full[c] - max_logit);
                real sum_exp = sum(exp_logits);
                for (c in 1:C)
                    rating_probs[idx][c] = exp_logits[c] / sum_exp;
                rating_cumprobs[idx] = cumulative_sum(rating_probs[idx]);
            }
        }
    } else {
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                rating_probs[idx] = dirichlet_rng(rep_vector(kappa / C, C));
                rating_cumprobs[idx] = cumulative_sum(rating_probs[idx]);
            }
        }
    }

    for (ij in 1:(I*J)) {
        rating_thresholds_z[ij][1] = negative_infinity();
        for (c in 2:C)
            rating_thresholds_z[ij][c] = inv_Phi(rating_cumprobs[ij][c-1]);
        rating_thresholds_z[ij][C+1] = positive_infinity();
    }

    // ===== GENERATE CONFUSION MATRIX =====
    for (c in 1:C) {
        vector[C] alpha_dir = rep_vector(1.0, C);
        if (use_dawid_skene_noise == 1)
            alpha_dir[c] = alpha_confusion;
        confusion_matrix[c] = dirichlet_rng(alpha_dir);
    }

    // ===== GENERATE ITEM EMBEDDINGS AND BASE SCORES =====
    for (k in 1:K) {
        for (d in 1:D)
            embeddings[k, d] = normal_rng(0, 1);
        for (i in 1:I)
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                base_scores[idx, k] = dot_product(eff_pref[idx], embeddings[k]);
            }
    }

    // ===== INITIALISE RATING ARRAYS =====
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                rating_observed[idx, k] = 0;
                rating_values[idx, k]   = 1;
            }
        }
    }

    // ===== GENERATE RATINGS: num_annotate_annotator sampled from ALL J per item =====
    for (k in 1:K) {
        if (J < num_annotate_annotator)
            reject("J is less than num_annotate_annotator");
        array[num_annotate_annotator] int selected_annotators;
        int num_selected = 0;
        while (num_selected < num_annotate_annotator) {
            real u = uniform_rng(0, 1);
            int candidate = 1 + to_int(floor(u * J));
            if (candidate > J) candidate = J;
            int already_selected = 0;
            for (s in 1:num_selected)
                if (selected_annotators[s] == candidate) { already_selected = 1; break; }
            if (already_selected == 0) {
                num_selected += 1;
                selected_annotators[num_selected] = candidate;
            }
        }
        for (i in 1:I) {
            for (s in 1:num_annotate_annotator) {
                int j   = selected_annotators[s];
                int idx = (i-1)*J + j;
                rating_observed[idx, k] = 1;
                real base_score = base_scores[idx, k];
                real pref_norm  = sqrt(dot_self(eff_pref[idx]));
                if (use_dawid_skene_noise == 1) {
                    real norm_score = base_score / pref_norm;
                    int latent_bin = C;
                    for (c in 1:C)
                        if (norm_score <= rating_thresholds_z[idx][c+1]) { latent_bin = c; break; }
                    rating_values[idx, k] = categorical_rng(confusion_matrix[latent_bin]);
                } else {
                    real noisy_score = base_score + normal_rng(0, sigma_measurement);
                    real total_std   = sqrt(pref_norm * pref_norm
                                           + sigma_measurement * sigma_measurement);
                    real cdf_val = Phi(noisy_score / total_std);
                    int rating = C;
                    for (c in 1:C)
                        if (cdf_val <= rating_cumprobs[idx][c]) { rating = c; break; }
                    rating_values[idx, k] = rating;
                }
            }
        }
    }

    // ===== COMPUTE ORACLE POSTERIOR PROBABILITIES (split by annotator group) =====

    // Training annotators: 1..J_train
    for (i in 1:I) {
        for (j in 1:J_train) {
            int global_idx = (i-1)*J + j;
            int split_idx  = (i-1)*J_train + j;
            real pref_norm = sqrt(dot_self(eff_pref[global_idx]));
            for (k in 1:K) {
                real base_score = base_scores[global_idx, k];
                if (use_dawid_skene_noise == 1) {
                    real norm_score = base_score / pref_norm;
                    int latent_bin = C;
                    for (c in 1:C)
                        if (norm_score <= rating_thresholds_z[global_idx][c+1]) { latent_bin = c; break; }
                    for (c in 1:C)
                        train_posterior_rating_probs[split_idx, k][c] = confusion_matrix[latent_bin][c];
                } else {
                    real total_std = sqrt(pref_norm*pref_norm + sigma_measurement*sigma_measurement);
                    real z_std     = base_score / total_std;
                    real sigma_std = sigma_measurement / total_std;
                    for (c in 1:C) {
                        real p_upper = (rating_thresholds_z[global_idx][c+1] == positive_infinity()) ? 1.0
                                     : Phi((rating_thresholds_z[global_idx][c+1] - z_std) / sigma_std);
                        real p_lower = (rating_thresholds_z[global_idx][c] == negative_infinity()) ? 0.0
                                     : Phi((rating_thresholds_z[global_idx][c] - z_std) / sigma_std);
                        train_posterior_rating_probs[split_idx, k][c] = p_upper - p_lower;
                    }
                }
            }
        }
    }

    // Validation annotators: J_train+1..J_train+J_val
    for (i in 1:I) {
        for (jv in 1:J_val) {
            int j          = J_train + jv;
            int global_idx = (i-1)*J + j;
            int split_idx  = (i-1)*J_val + jv;
            real pref_norm = sqrt(dot_self(eff_pref[global_idx]));
            for (k in 1:K) {
                real base_score = base_scores[global_idx, k];
                if (use_dawid_skene_noise == 1) {
                    real norm_score = base_score / pref_norm;
                    int latent_bin = C;
                    for (c in 1:C)
                        if (norm_score <= rating_thresholds_z[global_idx][c+1]) { latent_bin = c; break; }
                    for (c in 1:C)
                        val_posterior_rating_probs[split_idx, k][c] = confusion_matrix[latent_bin][c];
                } else {
                    real total_std = sqrt(pref_norm*pref_norm + sigma_measurement*sigma_measurement);
                    real z_std     = base_score / total_std;
                    real sigma_std = sigma_measurement / total_std;
                    for (c in 1:C) {
                        real p_upper = (rating_thresholds_z[global_idx][c+1] == positive_infinity()) ? 1.0
                                     : Phi((rating_thresholds_z[global_idx][c+1] - z_std) / sigma_std);
                        real p_lower = (rating_thresholds_z[global_idx][c] == negative_infinity()) ? 0.0
                                     : Phi((rating_thresholds_z[global_idx][c] - z_std) / sigma_std);
                        val_posterior_rating_probs[split_idx, k][c] = p_upper - p_lower;
                    }
                }
            }
        }
    }

    // Test annotators: J_train+J_val+1..J
    for (i in 1:I) {
        for (jt in 1:J_test) {
            int j          = J_train + J_val + jt;
            int global_idx = (i-1)*J + j;
            int split_idx  = (i-1)*J_test + jt;
            real pref_norm = sqrt(dot_self(eff_pref[global_idx]));
            for (k in 1:K) {
                real base_score = base_scores[global_idx, k];
                if (use_dawid_skene_noise == 1) {
                    real norm_score = base_score / pref_norm;
                    int latent_bin = C;
                    for (c in 1:C)
                        if (norm_score <= rating_thresholds_z[global_idx][c+1]) { latent_bin = c; break; }
                    for (c in 1:C)
                        test_posterior_rating_probs[split_idx, k][c] = confusion_matrix[latent_bin][c];
                } else {
                    real total_std = sqrt(pref_norm*pref_norm + sigma_measurement*sigma_measurement);
                    real z_std     = base_score / total_std;
                    real sigma_std = sigma_measurement / total_std;
                    for (c in 1:C) {
                        real p_upper = (rating_thresholds_z[global_idx][c+1] == positive_infinity()) ? 1.0
                                     : Phi((rating_thresholds_z[global_idx][c+1] - z_std) / sigma_std);
                        real p_lower = (rating_thresholds_z[global_idx][c] == negative_infinity()) ? 0.0
                                     : Phi((rating_thresholds_z[global_idx][c] - z_std) / sigma_std);
                        test_posterior_rating_probs[split_idx, k][c] = p_upper - p_lower;
                    }
                }
            }
        }
    }

    // ===== DEBUG =====
    if (DEBUG_PRINT == 1) {
        print("kappa=", kappa, "  T=", T, "  J_train=", J_train, "  J_val=", J_val, "  J_test=", J_test);
        print("alpha_jt[1]=", alpha_jt[1]);
        print("rating_probs[1]=", rating_probs[1]);
    }

    // ===== COMPUTE COUNTS =====
    num_train_observed_ratings = 0;
    num_val_observed_ratings   = 0;
    num_test_observed_ratings  = 0;
    for (i in 1:I) {
        for (j in 1:J_train) {
            int idx = (i-1)*J + j;
            for (k in 1:K)
                if (rating_observed[idx, k] == 1) num_train_observed_ratings += 1;
        }
        for (jv in 1:J_val) {
            int j = J_train + jv;
            int idx = (i-1)*J + j;
            for (k in 1:K)
                if (rating_observed[idx, k] == 1) num_val_observed_ratings += 1;
        }
        for (jt in 1:J_test) {
            int j = J_train + J_val + jt;
            int idx = (i-1)*J + j;
            for (k in 1:K)
                if (rating_observed[idx, k] == 1) num_test_observed_ratings += 1;
        }
    }
}

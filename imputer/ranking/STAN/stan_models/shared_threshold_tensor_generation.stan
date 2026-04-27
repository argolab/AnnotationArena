/*
 * Tensor model data generator with annotator-level shared thresholds.
 *
 * Generative model:
 *   z_ijk = sum_t alpha_jt * exp(u_i + v_t + u_it) . e_k
 *
 * where:
 *   u_i  in R^D: attribute embedding          ~ N(0, sigma_u)
 *   v_t  in R^D: prototype embedding (T total) ~ N(0, sigma_v)
 *   u_it in R^D: attribute-prototype interaction ~ N(0, sigma_uit)
 *   alpha_jt:    annotator j mixing weights over T prototypes ~ Dirichlet(1,...,1) # FLAG
 *   e_k  in R^D: item embedding               ~ N(0, 1)
 *
 * The effective preference vector for (i,j) is:
 *   eff_pref[ij] = sum_t alpha_jt[t] * exp(u_i + v_t + u_it)  (element-wise)
 * then z_ijk = dot(eff_pref[ij], e_k).
 *
 * Noise model (use_dawid_skene_noise):
 *   0 = continuous: noisy_score = z_ijk + N(0, sigma_measurement^2)
 *                   rating = ordinal_bin(noisy_score, thresholds)
 *   1 = discrete:   latent_bin = hard_bin(z_ijk, thresholds)
 *                   rating ~ Categorical(confusion_matrix[latent_bin])
 *
 * Threshold model:
 *   one Dirichlet bin-mass vector per annotator:
 *       p_j ~ Dirichlet(kappa/C, ...)
 *   thresholds tau_j are mixture quantiles of
 *       F_j(x) = I^-1 sum_i Phi(x / sqrt(||eff_pref[ij]||^2 + sigma_measurement^2))
 *   and are shared across all attributes i for annotator j.
 */

functions {

    real annotator_mixture_cdf_value(real x, vector scales) {
        real acc = 0.0;
        int I_local = num_elements(scales);
        for (i in 1:I_local)
            acc += Phi(x / scales[i]);
        return acc / I_local;
    }

    real annotator_mixture_quantile(real q, vector scales) {
        real lo = -1.0;
        real hi = 1.0;
        real mid;
        while (annotator_mixture_cdf_value(lo, scales) > q)
            lo *= 2.0;
        while (annotator_mixture_cdf_value(hi, scales) < q)
            hi *= 2.0;
        for (iter in 1:80) {
            mid = 0.5 * (lo + hi);
            if (annotator_mixture_cdf_value(mid, scales) < q)
                lo = mid;
            else
                hi = mid;
        }
        return 0.5 * (lo + hi);
    }

    vector ordinal_probs_from_thresholds(
        real base_score,
        real sigma_measurement,
        vector thresholds
    ) {
        int C_local = num_elements(thresholds) - 1;
        vector[C_local] probs;
        for (c in 1:C_local) {
            real upper_p = (thresholds[c+1] == positive_infinity())
                         ? 1.0
                         : Phi((thresholds[c+1] - base_score) / sigma_measurement);
            real lower_p = (thresholds[c] == negative_infinity())
                         ? 0.0
                         : Phi((thresholds[c] - base_score) / sigma_measurement);
            probs[c] = upper_p - lower_p;
        }
        return probs;
    }

}

data {
    int<lower=1> K_train;
    int<lower=1> K_test;
    int<lower=1> K_val;
    int<lower=1> I;               // number of attributes
    int<lower=1> J;               // number of annotators
    int<lower=1> D;               // embedding dimension
    int<lower=1> C;               // number of rating categories
    int<lower=1> T;               // number of annotator prototypes (default: 3)

    int<lower=0, upper=1> enable_pairwise_rankings;
    int<lower=0> pairwise_cap_per_item;

    // Embedding prior std
    real<lower=0> sigma_u;        // std for attribute embeddings u_i
    real<lower=0> sigma_v;        // std for prototype embeddings v_t
    real<lower=0> sigma_uit;      // std for attribute-prototype interactions u_it

    // Noise hyperparameters
    real<lower=0> sigma_measurement; // continuous noise std (only used if use_dawid_skene_noise=0)
    real<lower=0> kappa;             // Dirichlet concentration for rating_probs (threshold prior)
    real<lower=1> alpha_confusion;   // diagonal concentration for confusion matrix prior
    real<lower=0> temperature;       // Gumbel temperature for pairwise rankings

    int<lower=1> num_annotate_annotator;  // annotators selected per item

    int<lower=0, upper=1> use_dawid_skene_noise;           // 0=continuous, 1=discrete
    int<lower=0, upper=1> derive_thresholds_from_annotator; // 0=Dirichlet, 1=linear from alpha_j

    // Pass 1 when pairwise is disabled (avoids combinatorial array allocation).
    // Pass I*J*C*(K*(K-1)/2) when pairwise is enabled.
    int<lower=1> N_pairwise_max;
}

generated quantities {

    int DEBUG_PRINT = 1;

    int train_annotator_start = 1;
    int train_annotator_end   = J;
    int val_annotator_start   = 1;
    int val_annotator_end     = J;
    int test_annotator_start  = 1;
    int test_annotator_end    = J;

    // ===== TENSOR MODEL PARAMETERS =====
    matrix[I, D] u_attr;                  // attribute embeddings u_i
    matrix[T, D] v_proto;                 // prototype embeddings v_t
    array[I] matrix[T, D] u_inter;        // attribute-prototype interactions u_it
    array[J] simplex[T] alpha_jt;         // annotator mixing weights
    matrix[I*J, D] eff_pref;             // effective preference vectors for (i,j)

    // Rating threshold parameters.  The annotator_* arrays are the source
    // shared-threshold draws; rating_* arrays mirror them per (i,j) for
    // compatibility with existing generated-data consumers.
    array[J] simplex[C] annotator_rating_probs;
    array[J] vector[C] annotator_rating_cumprobs;
    array[J] vector[C+1] annotator_rating_thresholds;
    array[I*J] simplex[C] rating_probs;
    array[I*J] vector[C] rating_cumprobs;
    array[I*J] vector[C+1] rating_thresholds_z;
    matrix[C-1, T] threshold_transform_W;  // maps alpha_j (T-dim) -> (C-1) threshold logits
    matrix[I, C-1] threshold_attr_bias;    // per-attribute threshold bias

    // Confusion matrix (meaningful only when use_dawid_skene_noise=1)
    array[C] simplex[C] confusion_matrix;

    // ===== TRAIN / VAL / TEST INSTANCES =====
    matrix[K_train, D] train_embeddings;
    matrix[I*J, K_train] train_base_scores;
    array[I*J, K_train] int train_rating_values;
    array[I*J, K_train] int train_rating_observed;

    matrix[K_val, D] val_embeddings;
    matrix[I*J, K_val] val_base_scores;
    array[I*J, K_val] int val_rating_values;
    array[I*J, K_val] int val_rating_observed;

    matrix[K_test, D] test_embeddings;
    matrix[I*J, K_test] test_base_scores;
    array[I*J, K_test] int test_rating_values;
    array[I*J, K_test] int test_rating_observed;

    // ===== PAIRWISE RANKINGS =====
    array[N_pairwise_max, 2] int train_pairwise_items;
    array[N_pairwise_max] int train_pairwise_orders;
    array[N_pairwise_max] int train_pairwise_annotator;
    array[N_pairwise_max] int train_pairwise_attribute;
    array[N_pairwise_max] int train_pairwise_tied_rating;
    array[N_pairwise_max] int train_pairwise_observed;

    array[N_pairwise_max, 2] int val_pairwise_items;
    array[N_pairwise_max] int val_pairwise_orders;
    array[N_pairwise_max] int val_pairwise_annotator;
    array[N_pairwise_max] int val_pairwise_attribute;
    array[N_pairwise_max] int val_pairwise_tied_rating;
    array[N_pairwise_max] int val_pairwise_observed;

    array[N_pairwise_max, 2] int test_pairwise_items;
    array[N_pairwise_max] int test_pairwise_orders;
    array[N_pairwise_max] int test_pairwise_annotator;
    array[N_pairwise_max] int test_pairwise_attribute;
    array[N_pairwise_max] int test_pairwise_tied_rating;
    array[N_pairwise_max] int test_pairwise_observed;

    int num_train_pairwise_rankings;
    int num_val_pairwise_rankings;
    int num_test_pairwise_rankings;
    int num_train_observed_ratings;
    int num_val_observed_ratings;
    int num_test_observed_ratings;
    int num_train_missing_ratings;
    int num_val_missing_ratings;
    int num_test_missing_ratings;

    // Oracle posterior rating probabilities (ground-truth predictive distributions)
    array[I*J, K_train] vector[C] train_posterior_rating_probs;
    array[I*J, K_val]   vector[C] val_posterior_rating_probs;
    array[I*J, K_test]  vector[C] test_posterior_rating_probs;

    // ===== GENERATE TENSOR PARAMETERS =====
    {
        // u_i ~ N(0, sigma_u)
        for (i in 1:I)
            for (d in 1:D)
                u_attr[i, d] = normal_rng(0, sigma_u);

        // v_t ~ N(0, sigma_v)
        for (t in 1:T)
            for (d in 1:D)
                v_proto[t, d] = normal_rng(0, sigma_v);

        // u_it ~ N(0, sigma_uit)
        for (i in 1:I)
            for (t in 1:T)
                for (d in 1:D)
                    u_inter[i][t, d] = normal_rng(0, sigma_uit);

        // alpha_j ~ Dirichlet(1, ..., 1)  (uniform over prototypes)
        for (j in 1:J)
            alpha_jt[j] = dirichlet_rng(rep_vector(1.0, T));

        // eff_pref[ij] = sum_t alpha_jt[j][t] * exp(u_i + v_t + u_it)
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                row_vector[D] pref = rep_row_vector(0.0, D);
                for (t in 1:T) {
                    // log_gate[d] = u_attr[i,d] + v_proto[t,d] + u_inter[i][t,d]
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

    // ===== GENERATE ANNOTATOR-LEVEL SHARED RATING THRESHOLDS =====
    for (j in 1:J) {
        real threshold_noise = (use_dawid_skene_noise == 1) ? 0.05 : sigma_measurement;
        vector[I] annotator_scales;
        annotator_rating_probs[j] = dirichlet_rng(rep_vector(kappa / C, C));
        annotator_rating_cumprobs[j] = cumulative_sum(annotator_rating_probs[j]);
        for (i in 1:I) {
            int idx = (i-1)*J + j;
            annotator_scales[i] = sqrt(dot_self(eff_pref[idx])
                                      + threshold_noise * threshold_noise);
        }
        annotator_rating_thresholds[j][1] = negative_infinity();
        for (c in 2:C)
            annotator_rating_thresholds[j][c] = annotator_mixture_quantile(
                annotator_rating_cumprobs[j][c-1], annotator_scales
            );
        annotator_rating_thresholds[j][C+1] = positive_infinity();
    }

    // Mirror the shared annotator thresholds into the old per-pair containers.
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            rating_probs[idx] = annotator_rating_probs[j];
            rating_cumprobs[idx] = annotator_rating_cumprobs[j];
            rating_thresholds_z[idx] = annotator_rating_thresholds[j];
        }
    }

    // ===== GENERATE CONFUSION MATRIX =====
    // Always generated; only meaningful when use_dawid_skene_noise=1.
    // When noise=0 this is a uniform-prior Dirichlet draw and is unused.
    for (c in 1:C) {
        vector[C] alpha_dir = rep_vector(1.0, C);
        if (use_dawid_skene_noise == 1)
            alpha_dir[c] = alpha_confusion;
        confusion_matrix[c] = dirichlet_rng(alpha_dir);
    }

    // ===== GENERATE ITEM EMBEDDINGS AND BASE SCORES =====
    for (k in 1:K_train) {
        for (d in 1:D) train_embeddings[k, d] = normal_rng(0, 1);
        for (i in 1:I)
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                train_base_scores[idx, k] = dot_product(eff_pref[idx], train_embeddings[k]);
            }
    }
    for (k in 1:K_val) {
        for (d in 1:D) val_embeddings[k, d] = normal_rng(0, 1);
        for (i in 1:I)
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                val_base_scores[idx, k] = dot_product(eff_pref[idx], val_embeddings[k]);
            }
    }
    for (k in 1:K_test) {
        for (d in 1:D) test_embeddings[k, d] = normal_rng(0, 1);
        for (i in 1:I)
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                test_base_scores[idx, k] = dot_product(eff_pref[idx], test_embeddings[k]);
            }
    }

    // ===== GENERATE TRAINING RATINGS =====
    for (k in 1:K_train) {
        if ((train_annotator_end - train_annotator_start + 1) < num_annotate_annotator)
            reject("Too few training annotators for num_annotate_annotator");
        array[num_annotate_annotator] int selected_annotators;
        int num_selected = 0;
        while (num_selected < num_annotate_annotator) {
            real u = uniform_rng(0, 1);
            int candidate = train_annotator_start
                + to_int(floor(u * (train_annotator_end - train_annotator_start + 1)));
            if (candidate > train_annotator_end) candidate = train_annotator_end;
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
                train_rating_observed[idx, k] = 1;
                real base_score = train_base_scores[idx, k];
                if (use_dawid_skene_noise == 1) {
                    // Discrete: hard bin, then corrupt through confusion matrix
                    int latent_bin = C;
                    for (c in 1:C)
                        if (base_score <= rating_thresholds_z[idx][c+1]) { latent_bin = c; break; }
                    train_rating_values[idx, k] = categorical_rng(confusion_matrix[latent_bin]);
                } else {
                    // Continuous: add Gaussian noise, then bin
                    real noisy_score = base_score + normal_rng(0, sigma_measurement);
                    int rating = C;
                    for (c in 1:C)
                        if (noisy_score <= rating_thresholds_z[idx][c+1]) { rating = c; break; }
                    train_rating_values[idx, k] = rating;
                }
            }
        }
    }

    // ===== GENERATE VALIDATION RATINGS =====
    for (k in 1:K_val) {
        if ((val_annotator_end - val_annotator_start + 1) < num_annotate_annotator)
            reject("Too few validation annotators for num_annotate_annotator");
        array[num_annotate_annotator] int selected_annotators;
        int num_selected = 0;
        while (num_selected < num_annotate_annotator) {
            real u = uniform_rng(0, 1);
            int candidate = val_annotator_start
                + to_int(floor(u * (val_annotator_end - val_annotator_start + 1)));
            if (candidate > val_annotator_end) candidate = val_annotator_end;
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
                val_rating_observed[idx, k] = 1;
                real base_score = val_base_scores[idx, k];
                if (use_dawid_skene_noise == 1) {
                    int latent_bin = C;
                    for (c in 1:C)
                        if (base_score <= rating_thresholds_z[idx][c+1]) { latent_bin = c; break; }
                    val_rating_values[idx, k] = categorical_rng(confusion_matrix[latent_bin]);
                } else {
                    real noisy_score = base_score + normal_rng(0, sigma_measurement);
                    int rating = C;
                    for (c in 1:C)
                        if (noisy_score <= rating_thresholds_z[idx][c+1]) { rating = c; break; }
                    val_rating_values[idx, k] = rating;
                }
            }
        }
    }

    // ===== GENERATE TEST RATINGS =====
    for (k in 1:K_test) {
        if ((test_annotator_end - test_annotator_start + 1) < num_annotate_annotator)
            reject("Too few test annotators for num_annotate_annotator");
        array[num_annotate_annotator] int selected_annotators;
        int num_selected = 0;
        while (num_selected < num_annotate_annotator) {
            real u = uniform_rng(0, 1);
            int candidate = test_annotator_start
                + to_int(floor(u * (test_annotator_end - test_annotator_start + 1)));
            if (candidate > test_annotator_end) candidate = test_annotator_end;
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
                test_rating_observed[idx, k] = 1;
                real base_score = test_base_scores[idx, k];
                if (use_dawid_skene_noise == 1) {
                    int latent_bin = C;
                    for (c in 1:C)
                        if (base_score <= rating_thresholds_z[idx][c+1]) { latent_bin = c; break; }
                    test_rating_values[idx, k] = categorical_rng(confusion_matrix[latent_bin]);
                } else {
                    real noisy_score = base_score + normal_rng(0, sigma_measurement);
                    int rating = C;
                    for (c in 1:C)
                        if (noisy_score <= rating_thresholds_z[idx][c+1]) { rating = c; break; }
                    test_rating_values[idx, k] = rating;
                }
            }
        }
    }

    // ===== OBSERVATION COUNTS =====
    num_train_observed_ratings = K_train * I * num_annotate_annotator;
    num_val_observed_ratings   = K_val   * I * num_annotate_annotator;
    num_test_observed_ratings  = K_test  * I * num_annotate_annotator;
    num_train_missing_ratings  = K_train * I * (J - num_annotate_annotator);
    num_val_missing_ratings    = K_val   * I * (J - num_annotate_annotator);
    num_test_missing_ratings   = K_test  * I * (J - num_annotate_annotator);

    // ===== COMPUTE ORACLE POSTERIOR RATING PROBABILITIES =====
    // Continuous: ordinal probit probabilities given true sigma_measurement.
    // Discrete:   confusion matrix row at hard-binned latent bin.
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;

            for (k in 1:K_train) {
                real base_score = train_base_scores[idx, k];
                if (use_dawid_skene_noise == 1) {
                    int latent_bin = C;
                    for (c in 1:C)
                        if (base_score <= rating_thresholds_z[idx][c+1]) { latent_bin = c; break; }
                    for (c in 1:C)
                        train_posterior_rating_probs[idx, k][c] = confusion_matrix[latent_bin][c];
                } else {
                    train_posterior_rating_probs[idx, k] = ordinal_probs_from_thresholds(
                        base_score, sigma_measurement, rating_thresholds_z[idx]
                    );
                }
            }

            for (k in 1:K_val) {
                real base_score = val_base_scores[idx, k];
                if (use_dawid_skene_noise == 1) {
                    int latent_bin = C;
                    for (c in 1:C)
                        if (base_score <= rating_thresholds_z[idx][c+1]) { latent_bin = c; break; }
                    for (c in 1:C)
                        val_posterior_rating_probs[idx, k][c] = confusion_matrix[latent_bin][c];
                } else {
                    val_posterior_rating_probs[idx, k] = ordinal_probs_from_thresholds(
                        base_score, sigma_measurement, rating_thresholds_z[idx]
                    );
                }
            }

            for (k in 1:K_test) {
                real base_score = test_base_scores[idx, k];
                if (use_dawid_skene_noise == 1) {
                    int latent_bin = C;
                    for (c in 1:C)
                        if (base_score <= rating_thresholds_z[idx][c+1]) { latent_bin = c; break; }
                    for (c in 1:C)
                        test_posterior_rating_probs[idx, k][c] = confusion_matrix[latent_bin][c];
                } else {
                    test_posterior_rating_probs[idx, k] = ordinal_probs_from_thresholds(
                        base_score, sigma_measurement, rating_thresholds_z[idx]
                    );
                }
            }
        }
    }

    // ===== GENERATE PAIRWISE RANKINGS FROM TIED RATINGS =====
    num_train_pairwise_rankings = 0;
    num_val_pairwise_rankings   = 0;
    num_test_pairwise_rankings  = 0;

    if (enable_pairwise_rankings == 1) {

        // Training pairwise rankings
        for (i in 1:I) {
            for (j in train_annotator_start:train_annotator_end) {
                int ij_idx = (i-1)*J + j;
                for (rating_val in 1:C) {
                    array[K_train] int tied_items;
                    int num_tied = 0;
                    for (k in 1:K_train)
                        if (train_rating_values[ij_idx, k] == rating_val
                                && train_rating_observed[ij_idx, k] == 1) {
                            num_tied += 1;
                            tied_items[num_tied] = k;
                        }
                    if (num_tied >= 2 && pairwise_cap_per_item > 0) {
                        for (idx1 in 1:num_tied) {
                            int item1 = tied_items[idx1];
                            array[num_tied-1] int available_indices;
                            int num_available = 0;
                            for (idx in 1:num_tied)
                                if (idx != idx1) { num_available += 1; available_indices[num_available] = idx; }
                            int max_comparisons = min(pairwise_cap_per_item, num_available);
                            for (comp_idx in 1:max_comparisons) {
                                real u = uniform_rng(0, 1);
                                int sample_idx = 1 + to_int(floor(u * (num_available - comp_idx + 1)));
                                if (sample_idx > num_available - comp_idx + 1)
                                    sample_idx = num_available - comp_idx + 1;
                                int idx2  = available_indices[sample_idx];
                                int item2 = tied_items[idx2];
                                real u1 = uniform_rng(0, 1);
                                real g1 = -log(-log(u1));
                                real u2 = uniform_rng(0, 1);
                                real g2 = -log(-log(u2));
                                real util1 = train_base_scores[ij_idx, item1] / temperature + g1;
                                real util2 = train_base_scores[ij_idx, item2] / temperature + g2;
                                int order = (util1 > util2) ? 1 : 2;
                                num_train_pairwise_rankings += 1;
                                train_pairwise_items[num_train_pairwise_rankings, 1]  = item1;
                                train_pairwise_items[num_train_pairwise_rankings, 2]  = item2;
                                train_pairwise_orders[num_train_pairwise_rankings]    = order;
                                train_pairwise_annotator[num_train_pairwise_rankings] = j;
                                train_pairwise_attribute[num_train_pairwise_rankings] = i;
                                train_pairwise_tied_rating[num_train_pairwise_rankings] = rating_val;
                                train_pairwise_observed[num_train_pairwise_rankings]  = 1;
                                available_indices[sample_idx] = available_indices[num_available - comp_idx + 1];
                            }
                        }
                    }
                }
            }
        }

        // Validation pairwise rankings
        for (i in 1:I) {
            for (j in val_annotator_start:val_annotator_end) {
                int ij_idx = (i-1)*J + j;
                for (rating_val in 1:C) {
                    array[K_val] int tied_items;
                    int num_tied = 0;
                    for (k in 1:K_val)
                        if (val_rating_values[ij_idx, k] == rating_val
                                && val_rating_observed[ij_idx, k] == 1) {
                            num_tied += 1;
                            tied_items[num_tied] = k;
                        }
                    if (num_tied >= 2 && pairwise_cap_per_item > 0) {
                        for (idx1 in 1:num_tied) {
                            int item1 = tied_items[idx1];
                            array[num_tied-1] int available_indices;
                            int num_available = 0;
                            for (idx in 1:num_tied)
                                if (idx != idx1) { num_available += 1; available_indices[num_available] = idx; }
                            int max_comparisons = min(pairwise_cap_per_item, num_available);
                            for (comp_idx in 1:max_comparisons) {
                                real u = uniform_rng(0, 1);
                                int sample_idx = 1 + to_int(floor(u * (num_available - comp_idx + 1)));
                                if (sample_idx > num_available - comp_idx + 1)
                                    sample_idx = num_available - comp_idx + 1;
                                int idx2  = available_indices[sample_idx];
                                int item2 = tied_items[idx2];
                                real u1 = uniform_rng(0, 1);
                                real g1 = -log(-log(u1));
                                real u2 = uniform_rng(0, 1);
                                real g2 = -log(-log(u2));
                                real util1 = val_base_scores[ij_idx, item1] / temperature + g1;
                                real util2 = val_base_scores[ij_idx, item2] / temperature + g2;
                                int order = (util1 > util2) ? 1 : 2;
                                num_val_pairwise_rankings += 1;
                                val_pairwise_items[num_val_pairwise_rankings, 1]  = item1;
                                val_pairwise_items[num_val_pairwise_rankings, 2]  = item2;
                                val_pairwise_orders[num_val_pairwise_rankings]    = order;
                                val_pairwise_annotator[num_val_pairwise_rankings] = j;
                                val_pairwise_attribute[num_val_pairwise_rankings] = i;
                                val_pairwise_tied_rating[num_val_pairwise_rankings] = rating_val;
                                val_pairwise_observed[num_val_pairwise_rankings]  = 1;
                                available_indices[sample_idx] = available_indices[num_available - comp_idx + 1];
                            }
                        }
                    }
                }
            }
        }

        // Test pairwise rankings
        for (i in 1:I) {
            for (j in test_annotator_start:test_annotator_end) {
                int ij_idx = (i-1)*J + j;
                for (rating_val in 1:C) {
                    array[K_test] int tied_items;
                    int num_tied = 0;
                    for (k in 1:K_test)
                        if (test_rating_values[ij_idx, k] == rating_val
                                && test_rating_observed[ij_idx, k] == 1) {
                            num_tied += 1;
                            tied_items[num_tied] = k;
                        }
                    if (num_tied >= 2 && pairwise_cap_per_item > 0) {
                        for (idx1 in 1:num_tied) {
                            int item1 = tied_items[idx1];
                            array[num_tied-1] int available_indices;
                            int num_available = 0;
                            for (idx in 1:num_tied)
                                if (idx != idx1) { num_available += 1; available_indices[num_available] = idx; }
                            int max_comparisons = min(pairwise_cap_per_item, num_available);
                            for (comp_idx in 1:max_comparisons) {
                                real u = uniform_rng(0, 1);
                                int sample_idx = 1 + to_int(floor(u * (num_available - comp_idx + 1)));
                                if (sample_idx > num_available - comp_idx + 1)
                                    sample_idx = num_available - comp_idx + 1;
                                int idx2  = available_indices[sample_idx];
                                int item2 = tied_items[idx2];
                                real u1 = uniform_rng(0, 1);
                                real g1 = -log(-log(u1));
                                real u2 = uniform_rng(0, 1);
                                real g2 = -log(-log(u2));
                                real util1 = test_base_scores[ij_idx, item1] / temperature + g1;
                                real util2 = test_base_scores[ij_idx, item2] / temperature + g2;
                                int order = (util1 > util2) ? 1 : 2;
                                num_test_pairwise_rankings += 1;
                                test_pairwise_items[num_test_pairwise_rankings, 1]  = item1;
                                test_pairwise_items[num_test_pairwise_rankings, 2]  = item2;
                                test_pairwise_orders[num_test_pairwise_rankings]    = order;
                                test_pairwise_annotator[num_test_pairwise_rankings] = j;
                                test_pairwise_attribute[num_test_pairwise_rankings] = i;
                                test_pairwise_tied_rating[num_test_pairwise_rankings] = rating_val;
                                test_pairwise_observed[num_test_pairwise_rankings]  = 1;
                                available_indices[sample_idx] = available_indices[num_available - comp_idx + 1];
                            }
                        }
                    }
                }
            }
        }

    } // end if enable_pairwise_rankings

}

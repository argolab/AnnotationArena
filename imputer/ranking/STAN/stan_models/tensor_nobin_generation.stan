/*
 * Tensor model data generator (no rating binning).
 *
 * This variant uses a direct annotator-specific factorization (no prototype
 * mixture): t=j and alpha_jt is implicitly one-hot.
 *
 * Generative model:
 *   z_ijk = dot(eff_pref_ij, e_k)
 *   y_ijk = z_ijk + Normal(0, sigma_measurement)
 *
 * Observations:
 *   - train/val/test keep binary observed masks
 *   - all entries store scalar scores (observed and missing)
 *   - observed mask controls supervision visibility only
 *
 * Pairwise outputs are intentionally disabled in this no-binning generator.
 */

data {
    int<lower=1> K_train;
    int<lower=1> K_test;
    int<lower=1> K_val;
    int<lower=1> I;
    int<lower=1> J;
    int<lower=1> D;
    int<lower=1> C;               // unused, kept for data compatibility

    int<lower=0, upper=1> enable_pairwise_rankings; // ignored
    int<lower=0> pairwise_cap_per_item;             // ignored

    real<lower=0> sigma_u;
    real<lower=0> sigma_v;
    real<lower=0> sigma_uit;
    real<lower=0> sigma_measurement;
    real<lower=0> kappa;             // unused, kept for compatibility
    real<lower=1> alpha_confusion;   // unused, kept for compatibility
    real<lower=0> temperature;       // unused, kept for compatibility

    int<lower=1> num_annotate_annotator;

    int<lower=0, upper=1> use_dawid_skene_noise;            // unused
    int<lower=0, upper=1> derive_thresholds_from_annotator; // unused

    int<lower=1> N_pairwise_max; // unused, kept for compatibility
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
    matrix[I, D] u_attr;
    matrix[J, D] v_annot;
    array[I] matrix[J, D] u_inter;
    matrix[I*J, D] eff_pref;

    // ===== TRAIN / VAL / TEST INSTANCES =====
    matrix[K_train, D] train_embeddings;
    matrix[I*J, K_train] train_base_scores;
    matrix[I*J, K_train] train_rating_scores;
    array[I*J, K_train] int train_rating_observed;

    matrix[K_val, D] val_embeddings;
    matrix[I*J, K_val] val_base_scores;
    matrix[I*J, K_val] val_rating_scores;
    array[I*J, K_val] int val_rating_observed;

    matrix[K_test, D] test_embeddings;
    matrix[I*J, K_test] test_base_scores;
    matrix[I*J, K_test] test_rating_scores;
    array[I*J, K_test] int test_rating_observed;

    int num_train_observed_ratings;
    int num_val_observed_ratings;
    int num_test_observed_ratings;
    int num_train_missing_ratings;
    int num_val_missing_ratings;
    int num_test_missing_ratings;

    // ===== GENERATE TENSOR PARAMETERS =====
    {
        int disable_u_inter = sigma_uit <= 1e-12;

        for (i in 1:I)
            for (d in 1:D)
                u_attr[i, d] = normal_rng(0, sigma_u);

        for (j in 1:J)
            for (d in 1:D)
                v_annot[j, d] = normal_rng(0, sigma_v);

        for (i in 1:I)
            for (j in 1:J)
                for (d in 1:D)
                    u_inter[i][j, d] = disable_u_inter == 1 ? 1.0 : normal_rng(0, sigma_uit);

        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                row_vector[D] log_gate;
                if (disable_u_inter == 1) {
                    // u_it disabled: use exp(u_i .* v_j) directly.
                    log_gate = u_attr[i] .* v_annot[j];
                } else {
                    log_gate = u_attr[i] .* v_annot[j] .* u_inter[i][j];
                }
                eff_pref[idx] = exp(log_gate);
            }
        }
    }

    // ===== GENERATE ITEM EMBEDDINGS AND BASE SCORES =====
    for (k in 1:K_train) {
        for (d in 1:D) train_embeddings[k, d] = normal_rng(0, 0.1);
        for (i in 1:I)
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                train_base_scores[idx, k] = dot_product(eff_pref[idx], train_embeddings[k]);
                train_rating_scores[idx, k] = 0;
                train_rating_observed[idx, k] = 0;
            }
    }
    for (k in 1:K_val) {
        for (d in 1:D) val_embeddings[k, d] = normal_rng(0, 0.1);
        for (i in 1:I)
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                val_base_scores[idx, k] = dot_product(eff_pref[idx], val_embeddings[k]);
                val_rating_scores[idx, k] = 0;
                val_rating_observed[idx, k] = 0;
            }
    }
    for (k in 1:K_test) {
        for (d in 1:D) test_embeddings[k, d] = normal_rng(0, 0.1);
        for (i in 1:I)
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                test_base_scores[idx, k] = dot_product(eff_pref[idx], test_embeddings[k]);
                test_rating_scores[idx, k] = 0;
                test_rating_observed[idx, k] = 0;
            }
    }

    // ===== GENERATE TRAINING SCORES =====
    for (k in 1:K_train) {
        if ((train_annotator_end - train_annotator_start + 1) < num_annotate_annotator)
            reject("Too few training annotators for num_annotate_annotator");
        array[num_annotate_annotator] int selected_annotators;
        array[J] int is_selected;
        int num_selected = 0;
        for (j in 1:J) is_selected[j] = 0;
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
                is_selected[candidate] = 1;
            }
        }
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                train_rating_scores[idx, k] = train_base_scores[idx, k] + normal_rng(0, sigma_measurement);
                train_rating_observed[idx, k] = is_selected[j];
            }
        }
    }

    // ===== GENERATE VALIDATION SCORES =====
    for (k in 1:K_val) {
        if ((val_annotator_end - val_annotator_start + 1) < num_annotate_annotator)
            reject("Too few validation annotators for num_annotate_annotator");
        array[num_annotate_annotator] int selected_annotators;
        array[J] int is_selected;
        int num_selected = 0;
        for (j in 1:J) is_selected[j] = 0;
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
                is_selected[candidate] = 1;
            }
        }
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                val_rating_scores[idx, k] = val_base_scores[idx, k] + normal_rng(0, sigma_measurement);
                val_rating_observed[idx, k] = is_selected[j];
            }
        }
    }

    // ===== GENERATE TEST SCORES =====
    for (k in 1:K_test) {
        if ((test_annotator_end - test_annotator_start + 1) < num_annotate_annotator)
            reject("Too few test annotators for num_annotate_annotator");
        array[num_annotate_annotator] int selected_annotators;
        array[J] int is_selected;
        int num_selected = 0;
        for (j in 1:J) is_selected[j] = 0;
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
                is_selected[candidate] = 1;
            }
        }
        for (i in 1:I) {
            for (j in 1:J) {
                int idx = (i-1)*J + j;
                test_rating_scores[idx, k] = test_base_scores[idx, k] + normal_rng(0, sigma_measurement);
                test_rating_observed[idx, k] = is_selected[j];
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
}

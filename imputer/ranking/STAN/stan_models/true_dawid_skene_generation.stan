/*
 * True Dawid-Skene synthetic data generator.
 *
 * This generator matches the structure of true_dawid_skene_model.stan:
 * - one latent true class per (attribute, item) cell
 * - one confusion matrix per annotator, shared across attributes
 * - no embeddings, no ordinal scores, no pairwise rankings
 *
 * The supplied alpha_* values are generation hyperparameters controlling the
 * Dirichlet draws for:
 * - pi[i]                  : attribute-specific class prevalence
 * - confusion[j][c_true]   : annotator confusion row for latent class c_true
 */

data {
    int<lower=1> K_train;
    int<lower=1> K_test;
    int<lower=1> K_val;
    int<lower=1> I;
    int<lower=1> J;
    int<lower=1> C;

    int<lower=0, upper=1> enable_pairwise_rankings;
    int<lower=0> pairwise_cap_per_item;
    int<lower=1> num_annotate_annotator;
    int<lower=1> N_pairwise_max;

    real<lower=0> alpha_pi;
    real<lower=0> alpha_confusion_diag;
    real<lower=0> alpha_confusion_offdiag;
}

generated quantities {
    int DEBUG_PRINT = 0;

    array[I] simplex[C] pi;
    array[J, C] simplex[C] confusion;

    array[I, K_train] int<lower=1, upper=C> train_true_classes;
    array[I, K_val] int<lower=1, upper=C> val_true_classes;
    array[I, K_test] int<lower=1, upper=C> test_true_classes;

    array[I * J, K_train] int<lower=0, upper=C> train_rating_values;
    array[I * J, K_train] int<lower=0, upper=1> train_rating_observed;
    array[I * J, K_val] int<lower=0, upper=C> val_rating_values;
    array[I * J, K_val] int<lower=0, upper=1> val_rating_observed;
    array[I * J, K_test] int<lower=0, upper=C> test_rating_values;
    array[I * J, K_test] int<lower=0, upper=1> test_rating_observed;

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

    int num_train_pairwise_rankings = 0;
    int num_val_pairwise_rankings = 0;
    int num_test_pairwise_rankings = 0;
    int num_train_observed_ratings = 0;
    int num_val_observed_ratings = 0;
    int num_test_observed_ratings = 0;
    int num_train_missing_ratings = 0;
    int num_val_missing_ratings = 0;
    int num_test_missing_ratings = 0;

    array[I * J, K_train] vector[C] train_posterior_rating_probs;
    array[I * J, K_val] vector[C] val_posterior_rating_probs;
    array[I * J, K_test] vector[C] test_posterior_rating_probs;

    for (i in 1:I) {
        pi[i] = dirichlet_rng(rep_vector(alpha_pi, C));
    }

    for (j in 1:J) {
        for (c_true in 1:C) {
            vector[C] alpha_dir = rep_vector(alpha_confusion_offdiag, C);
            alpha_dir[c_true] = alpha_confusion_diag;
            confusion[j, c_true] = dirichlet_rng(alpha_dir);
        }
    }

    for (n in 1:N_pairwise_max) {
        train_pairwise_items[n, 1] = 1;
        train_pairwise_items[n, 2] = 1;
        train_pairwise_orders[n] = 1;
        train_pairwise_annotator[n] = 1;
        train_pairwise_attribute[n] = 1;
        train_pairwise_tied_rating[n] = 1;
        train_pairwise_observed[n] = 0;

        val_pairwise_items[n, 1] = 1;
        val_pairwise_items[n, 2] = 1;
        val_pairwise_orders[n] = 1;
        val_pairwise_annotator[n] = 1;
        val_pairwise_attribute[n] = 1;
        val_pairwise_tied_rating[n] = 1;
        val_pairwise_observed[n] = 0;

        test_pairwise_items[n, 1] = 1;
        test_pairwise_items[n, 2] = 1;
        test_pairwise_orders[n] = 1;
        test_pairwise_annotator[n] = 1;
        test_pairwise_attribute[n] = 1;
        test_pairwise_tied_rating[n] = 1;
        test_pairwise_observed[n] = 0;
    }

    for (ij in 1:(I * J)) {
        for (k in 1:K_train) {
            train_rating_values[ij, k] = 0;
            train_rating_observed[ij, k] = 0;
        }
        for (k in 1:K_val) {
            val_rating_values[ij, k] = 0;
            val_rating_observed[ij, k] = 0;
        }
        for (k in 1:K_test) {
            test_rating_values[ij, k] = 0;
            test_rating_observed[ij, k] = 0;
        }
    }

    for (k in 1:K_train) {
        array[num_annotate_annotator] int selected_annotators;
        int num_selected = 0;

        while (num_selected < num_annotate_annotator) {
            real u = uniform_rng(0, 1);
            int candidate = 1 + to_int(floor(u * J));
            if (candidate > J) candidate = J;

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
            int c_true = categorical_rng(pi[i]);
            train_true_classes[i, k] = c_true;

            for (j in 1:J) {
                int ij_idx = (i - 1) * J + j;
                train_posterior_rating_probs[ij_idx, k] = to_vector(confusion[j, c_true]);
            }

            for (s in 1:num_annotate_annotator) {
                int j = selected_annotators[s];
                int ij_idx = (i - 1) * J + j;
                train_rating_values[ij_idx, k] = categorical_rng(confusion[j, c_true]);
                train_rating_observed[ij_idx, k] = 1;
                num_train_observed_ratings += 1;
            }
            for (j in 1:J) {
                int observed = 0;
                for (s in 1:num_annotate_annotator) {
                    if (selected_annotators[s] == j) {
                        observed = 1;
                        break;
                    }
                }
                if (observed == 0) {
                    num_train_missing_ratings += 1;
                }
            }
        }
    }

    for (k in 1:K_val) {
        array[num_annotate_annotator] int selected_annotators;
        int num_selected = 0;

        while (num_selected < num_annotate_annotator) {
            real u = uniform_rng(0, 1);
            int candidate = 1 + to_int(floor(u * J));
            if (candidate > J) candidate = J;

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
            int c_true = categorical_rng(pi[i]);
            val_true_classes[i, k] = c_true;

            for (j in 1:J) {
                int ij_idx = (i - 1) * J + j;
                val_posterior_rating_probs[ij_idx, k] = to_vector(confusion[j, c_true]);
            }

            for (s in 1:num_annotate_annotator) {
                int j = selected_annotators[s];
                int ij_idx = (i - 1) * J + j;
                val_rating_values[ij_idx, k] = categorical_rng(confusion[j, c_true]);
                val_rating_observed[ij_idx, k] = 1;
                num_val_observed_ratings += 1;
            }
            for (j in 1:J) {
                int observed = 0;
                for (s in 1:num_annotate_annotator) {
                    if (selected_annotators[s] == j) {
                        observed = 1;
                        break;
                    }
                }
                if (observed == 0) {
                    num_val_missing_ratings += 1;
                }
            }
        }
    }

    for (k in 1:K_test) {
        array[num_annotate_annotator] int selected_annotators;
        int num_selected = 0;

        while (num_selected < num_annotate_annotator) {
            real u = uniform_rng(0, 1);
            int candidate = 1 + to_int(floor(u * J));
            if (candidate > J) candidate = J;

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
            int c_true = categorical_rng(pi[i]);
            test_true_classes[i, k] = c_true;

            for (j in 1:J) {
                int ij_idx = (i - 1) * J + j;
                test_posterior_rating_probs[ij_idx, k] = to_vector(confusion[j, c_true]);
            }

            for (s in 1:num_annotate_annotator) {
                int j = selected_annotators[s];
                int ij_idx = (i - 1) * J + j;
                test_rating_values[ij_idx, k] = categorical_rng(confusion[j, c_true]);
                test_rating_observed[ij_idx, k] = 1;
                num_test_observed_ratings += 1;
            }
            for (j in 1:J) {
                int observed = 0;
                for (s in 1:num_annotate_annotator) {
                    if (selected_annotators[s] == j) {
                        observed = 1;
                        break;
                    }
                }
                if (observed == 0) {
                    num_test_missing_ratings += 1;
                }
            }
        }
    }
}

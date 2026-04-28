/*
 * True Dawid-Skene annotation model.
 *
 * This is a multi-attribute extension of Dawid & Skene (1979). It is structurally
 * distinct from all dot-product models in this codebase: there is no latent
 * continuous score, no item embedding, no ordinal probit, and no annotator
 * preference vector. Items are characterized purely by their discrete true class.
 *
 * ── Generative process ──────────────────────────────────────────────────────
 *
 *   For each (attribute i, item k):
 *       T_ik ~ Categorical(pi[i])
 *   where pi[i] in Delta^{C-1} is the class prevalence for attribute i.
 *   T_ik is the latent "true" rating for this (attribute, item) pair.
 *
 *   For each annotator j who rates item k on attribute i:
 *       r_ijk ~ Categorical(confusion[j][ T_ik ])
 *   where confusion[j] is a per-annotator C×C row-stochastic matrix.
 *   confusion[j][c][v] = P(annotator j reports v | true class is c).
 *
 * ── Key modelling assumptions ────────────────────────────────────────────────
 *
 *   - T_ik is per (attribute, item): different attributes of the same item
 *     have independent true classes (no cross-attribute correlation).
 *   - Confusion is per annotator, not per (annotator, attribute): annotator j
 *     has the same confusion matrix across all 9 attributes.
 *   - All annotators observing the same (i, k) pair are coupled through the
 *     shared T_ik — this is what makes it correct DS, not a product of
 *     independent per-rating mixtures.
 *
 * ── Inference ────────────────────────────────────────────────────────────────
 *
 *   T_ik is a discrete latent variable. HMC requires marginalizing it out.
 *   Ratings are grouped into cells — one per (attribute, item) pair — so the
 *   coupling across annotators is captured:
 *
 *   log P({r_ijk}_j | pi[i], {confusion[j]}) =
 *       log sum_c  pi[i][c] * prod_j confusion[j][c][r_ijk]
 *
 * ── Hyperparameters ───────────────────────────────────────────────────────────
 *
 *   alpha_pi             — Dirichlet concentration for pi[i]; symmetric across
 *                          all C classes.
 *   alpha_confusion_diag — diagonal of Dirichlet prior for each confusion row.
 *   alpha_confusion_offdiag — off-diagonal entries of the Dirichlet prior.
 *
 *   This model supports two modes:
 *   1. Fixed-anchor mode (legacy): keep the three concentrations effectively
 *      fixed to the supplied anchor values.
 *   2. Learned-hyperparameter mode: sample the three concentrations with broad,
 *      proper priors, analogous to the real-data "flat-like" treatment used in
 *      other models in this codebase.
 *
 * ── Generated quantities interface (matches predictives.py) ─────────────────
 *
 *   missing_rating_predictions[N_missing_ratings] — posterior predictive draws
 *   missing_rating_probs[N_missing_ratings, C]    — predictive distributions
 *   log_lik_ratings_obs                           — total observed log-likelihood
 *   log_lik_pairwise_obs                          — stub (always 0)
 *   total_log_lik                                 — same as log_lik_ratings_obs
 *
 * ── CSV pattern for evaluate_predictions.py ─────────────────────────────────
 *
 *   --csv-pattern "true_dawid_skene_model-*.csv"
 */

data {

    // ── Dimensions ──────────────────────────────────────────────────────────
    int<lower=1> K;    // total items (K_train + K_val + K_test)
    int<lower=1> I;    // number of attributes
    int<lower=1> J;    // number of annotators
    int<lower=1> C;    // number of rating classes

    // ── Hyperparameter controls ──────────────────────────────────────────────
    real<lower=0> alpha_pi_anchor;                // legacy/fixed-mode anchor for alpha_pi
    real<lower=0> alpha_confusion_diag_anchor;    // legacy/fixed-mode anchor for confusion diagonal
    real<lower=0> alpha_confusion_offdiag_anchor; // legacy/fixed-mode anchor for confusion off-diagonal
    int<lower=0, upper=1> use_flat_priors;        // 1 = learn hyperparameters with broad proper priors

    // ── Cell structure ───────────────────────────────────────────────────────
    // Each cell = one unique (attribute, item) pair.
    // N_cells covers all cells that appear in observed OR missing ratings.
    // Cells with only missing ratings (no observed) have cell_obs_count = 0;
    // their likelihood contribution is 0, and prediction falls back to the prior.
    int<lower=0> N_cells;
    array[N_cells] int<lower=1, upper=I> cell_attr;  // attribute i for each cell
    array[N_cells] int<lower=1, upper=K> cell_item;  // item k for each cell
    array[N_cells] int<lower=1>          cell_obs_start; // start index in obs arrays (1-indexed)
    array[N_cells] int<lower=0>          cell_obs_count; // number of observed ratings in this cell

    // ── Observed ratings (sorted by cell) ────────────────────────────────────
    // Cells with count=0 contribute no entries. The total length equals
    // sum(cell_obs_count).
    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=J> obs_annotator;
    array[N_ratings] int<lower=1, upper=C> obs_value;

    // ── Missing ratings to predict ────────────────────────────────────────────
    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=N_cells> missing_cell;     // which cell
    array[N_missing_ratings] int<lower=1, upper=J>       missing_annotator; // annotator j

}

parameters {

    // Learned prior concentrations. In fixed-anchor mode these are tightly
    // concentrated around the supplied anchor values.
    real<lower=0> alpha_pi_param;
    real<lower=0> alpha_confusion_diag_param;
    real<lower=0> alpha_confusion_offdiag_param;

    // Per-attribute class prevalence.
    // pi[i][c] = P(T_ik = c) for any item k under attribute i.
    array[I] simplex[C] pi;

    // Per-annotator confusion matrix.
    // confusion[j][c] is a C-simplex representing P(observed class | true class = c).
    // confusion[j][c][v] = P(r = v | T = c, annotator j).
    array[J, C] simplex[C] confusion;

}

transformed parameters {

    real<lower=0> alpha_pi = alpha_pi_param;
    real<lower=0> alpha_confusion_diag = alpha_confusion_diag_param;
    real<lower=0> alpha_confusion_offdiag = alpha_confusion_offdiag_param;

}

model {

    // ── Priors ──────────────────────────────────────────────────────────────

    if (use_flat_priors == 1) {
        // Broad but proper priors for real-data fitting. This mirrors the
        // "flat-like" treatment used elsewhere in the repo: let the model
        // choose these concentrations instead of pinning them.
        alpha_pi_param                ~ gamma(2, 2);        // mean 1.0
        alpha_confusion_diag_param    ~ gamma(2, 2.0 / 7.0); // mean 7.0
        alpha_confusion_offdiag_param ~ gamma(2, 2);        // mean 1.0
    } else {
        // Legacy behavior: effectively fix these to supplied anchors while
        // keeping the model proper.
        alpha_pi_param                ~ lognormal(log(fmax(alpha_pi_anchor, 1e-8)), 0.02);
        alpha_confusion_diag_param    ~ lognormal(log(fmax(alpha_confusion_diag_anchor, 1e-8)), 0.02);
        alpha_confusion_offdiag_param ~ lognormal(log(fmax(alpha_confusion_offdiag_anchor, 1e-8)), 0.02);
    }

    // Symmetric Dirichlet prior on class prevalences.
    for (i in 1:I) {
        pi[i] ~ dirichlet(rep_vector(alpha_pi, C));
    }

    // Informative Dirichlet prior on each confusion row.
    // Diagonal gets alpha_confusion_diag; off-diagonals get alpha_confusion_offdiag.
    // This pushes confusion matrices toward the identity (accurate annotators)
    // while allowing the data to learn actual confusion patterns.
    for (j in 1:J) {
        for (c_true in 1:C) {
            vector[C] alpha_dir = rep_vector(alpha_confusion_offdiag, C);
            alpha_dir[c_true] = alpha_confusion_diag;
            confusion[j][c_true] ~ dirichlet(alpha_dir);
        }
    }

    // ── Likelihood: marginalize over T_ik ────────────────────────────────────
    //
    // For each cell (i,k), the joint likelihood of all observed ratings is:
    //   P({r_ijk}_j | pi[i], {confusion[j]}) =
    //       sum_{c=1}^{C}  pi[i][c]  *  prod_{j: obs} confusion[j][c][r_ijk]
    //
    // Computed in log-space via log_sum_exp for numerical stability.
    for (n in 1:N_cells) {
        if (cell_obs_count[n] == 0) {
            // No observations for this cell: zero likelihood contribution.
            continue;
        }
        int i = cell_attr[n];
        vector[C] log_mix;
        for (c in 1:C) {
            log_mix[c] = log(pi[i][c]);
        }
        for (r in cell_obs_start[n]:(cell_obs_start[n] + cell_obs_count[n] - 1)) {
            int j = obs_annotator[r];
            int v = obs_value[r];
            for (c in 1:C) {
                log_mix[c] += log(confusion[j][c][v]);
            }
        }
        target += log_sum_exp(log_mix);
    }

}

generated quantities {

    // ── Observed log-likelihood ──────────────────────────────────────────────
    real log_lik_ratings_obs = 0.0;
    real log_lik_pairwise_obs = 0.0;  // stub — no pairwise data
    real total_log_lik;

    for (n in 1:N_cells) {
        if (cell_obs_count[n] == 0) {
            continue;
        }
        int i = cell_attr[n];
        vector[C] log_mix;
        for (c in 1:C) {
            log_mix[c] = log(pi[i][c]);
        }
        for (r in cell_obs_start[n]:(cell_obs_start[n] + cell_obs_count[n] - 1)) {
            int j = obs_annotator[r];
            int v = obs_value[r];
            for (c in 1:C) {
                log_mix[c] += log(confusion[j][c][v]);
            }
        }
        log_lik_ratings_obs += log_sum_exp(log_mix);
    }

    total_log_lik = log_lik_ratings_obs;

    // ── Posterior predictive for missing ratings ─────────────────────────────
    //
    // For each missing rating (i, j, k):
    //   1. Compute the posterior over T_ik using all observed ratings of that cell:
    //          post[c] ∝ pi[i][c] * prod_{j': observed} confusion[j'][c][r_ij'k]
    //   2. Mix through annotator j's confusion row to get the predictive:
    //          P(r = v | observed) = sum_c post[c] * confusion[j][c][v]
    //
    // When cell_obs_count = 0 (no observed ratings for this cell), post = pi[i].

    array[N_missing_ratings] int<lower=1, upper=C> missing_rating_predictions;
    array[N_missing_ratings] vector[C]             missing_rating_probs;

    for (n in 1:N_missing_ratings) {
        int cell_n = missing_cell[n];
        int i      = cell_attr[cell_n];
        int j      = missing_annotator[n];

        // Step 1: posterior over true class T_ik
        vector[C] log_post;
        for (c in 1:C) {
            log_post[c] = log(pi[i][c]);
        }
        if (cell_obs_count[cell_n] > 0) {
            for (r in cell_obs_start[cell_n]:(cell_obs_start[cell_n] + cell_obs_count[cell_n] - 1)) {
                int j_obs = obs_annotator[r];
                int v_obs = obs_value[r];
                for (c in 1:C) {
                    log_post[c] += log(confusion[j_obs][c][v_obs]);
                }
            }
        }
        vector[C] post = softmax(log_post);

        // Step 2: predictive distribution P(r = v | T_ik posterior, annotator j)
        vector[C] pred = rep_vector(0.0, C);
        for (c in 1:C) {
            for (v in 1:C) {
                pred[v] += post[c] * confusion[j][c][v];
            }
        }

        missing_rating_probs[n]       = pred;
        missing_rating_predictions[n] = categorical_rng(pred);
    }

}

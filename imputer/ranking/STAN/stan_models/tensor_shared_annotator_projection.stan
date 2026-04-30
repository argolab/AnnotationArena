/*
 * Tensor model HMC inference with shared annotator projection weights.
 *
 * Misspecification:
 *   All annotators share one prototype mixture alpha_shared.
 *   Scores depend on (i, k) only through eff_pref_attr[i] . e_k.
 *   Thresholds remain per-(i,j) as in tensor_model.stan.
 */

functions {

    real ordinal_bin_prob(
        real base_score,
        real total_std,
        real sigma_measurement,
        real lower_threshold,
        real upper_threshold
    ) {
        real mean_std = base_score / total_std;
        real cond_std = sigma_measurement / total_std;
        real upper_p  = (upper_threshold == positive_infinity())
                         ? 1.0
                         : Phi((upper_threshold - mean_std) / cond_std);
        real lower_p  = (lower_threshold == negative_infinity())
                         ? 0.0
                         : Phi((lower_threshold - mean_std) / cond_std);
        return upper_p - lower_p;
    }

}

data {
    int<lower=1> K;
    int<lower=1> I;
    int<lower=1> J;
    int<lower=1> D;
    int<lower=1> C;
    int<lower=1> T;

    int<lower=0, upper=1> use_dawid_skene_noise;
    int<lower=0, upper=1> derive_thresholds_from_annotator;

    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=I> rating_attributes;
    array[N_ratings] int<lower=1, upper=J> rating_annotators;
    array[N_ratings] int<lower=1, upper=K> rating_items;
    array[N_ratings] int<lower=1, upper=C> rating_values;

    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=I> missing_rating_attributes;
    array[N_missing_ratings] int<lower=1, upper=J> missing_rating_annotators;
    array[N_missing_ratings] int<lower=1, upper=K> missing_rating_items;

    real<lower=0> alpha_dirichlet_jt;
    real<lower=0> alpha_confusion;
}

parameters {
    matrix[K, D] embeddings;

    matrix[I, D] u_attr;
    matrix[T, D] v_proto;
    array[I] matrix[T, D] u_inter;
    real<lower=0> sigma_u;
    real<lower=0> sigma_v;
    real<lower=0> sigma_uit;
    real<lower=0> sigma_measurement;
    real<lower=0> kappa;

    // Shared mixture over prototypes for all annotators.
    simplex[T] alpha_shared;

    array[I*J] simplex[C] rating_probs;
    matrix[C-1, T] threshold_transform_W;
    matrix[I, C-1] threshold_attr_bias;
    array[C] simplex[C] confusion_matrix;
}

transformed parameters {
    // Shared preference vectors by attribute.
    matrix[I, D] eff_pref_attr;
    for (i in 1:I) {
        row_vector[D] pref = rep_row_vector(0.0, D);
        for (t in 1:T) {
            row_vector[D] log_gate = u_attr[i] + v_proto[t] + u_inter[i][t];
            pref = pref + alpha_shared[t] * exp(log_gate);
        }
        eff_pref_attr[i] = pref;
    }

    matrix[I, K] base_scores_attr;
    for (i in 1:I)
        for (k in 1:K)
            base_scores_attr[i, k] = dot_product(eff_pref_attr[i], embeddings[k]);

    array[I*J] vector[C+1] rating_thresholds;
    for (ij in 1:(I*J)) {
        rating_thresholds[ij][1] = negative_infinity();
        for (c in 2:C)
            rating_thresholds[ij][c] = inv_Phi(sum(rating_probs[ij][1:(c-1)]));
        rating_thresholds[ij][C+1] = positive_infinity();
    }

    real bin_smoothing = 0.05;
    array[I] real total_std_attr;
    for (i in 1:I) {
        real noise_sq = (use_dawid_skene_noise == 1)
                         ? bin_smoothing * bin_smoothing
                         : sigma_measurement * sigma_measurement;
        total_std_attr[i] = sqrt(dot_self(eff_pref_attr[i]) + noise_sq);
    }

    real sigma_noise = (use_dawid_skene_noise == 1) ? bin_smoothing : sigma_measurement;
}

model {
    for (k in 1:K)
        embeddings[k] ~ normal(0, 1);

    sigma_u           ~ gamma(2, 2.5);
    sigma_v           ~ gamma(2, 0.25);
    sigma_uit         ~ gamma(2, 2.5);
    sigma_measurement ~ gamma(2, 20);
    kappa             ~ gamma(2, 2.0 / 15.0);

    to_vector(u_attr)  ~ normal(0, sigma_u);
    to_vector(v_proto) ~ normal(0, sigma_v);
    for (i in 1:I)
        to_vector(u_inter[i]) ~ normal(0, sigma_uit);

    alpha_shared ~ dirichlet(rep_vector(alpha_dirichlet_jt, T));

    for (ij in 1:(I*J))
        rating_probs[ij] ~ dirichlet(rep_vector(kappa / C, C));

    to_vector(threshold_transform_W) ~ normal(0, 1.0 / sqrt(T));
    to_vector(threshold_attr_bias)   ~ normal(0, 0.1);

    for (c in 1:C) {
        vector[C] alpha_dir = rep_vector(1.0, C);
        alpha_dir[c] = alpha_confusion;
        confusion_matrix[c] ~ dirichlet(alpha_dir);
    }

    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;
        int c_obs  = rating_values[n];

        if (use_dawid_skene_noise == 1) {
            real mix = 0.0;
            for (c_lat in 1:C) {
                mix += ordinal_bin_prob(
                    base_scores_attr[i, k], total_std_attr[i], sigma_noise,
                    rating_thresholds[ij_idx][c_lat], rating_thresholds[ij_idx][c_lat+1]
                ) * confusion_matrix[c_lat][c_obs];
            }
            target += log(mix + 1e-10);
        } else {
            target += log(ordinal_bin_prob(
                base_scores_attr[i, k], total_std_attr[i], sigma_noise,
                rating_thresholds[ij_idx][c_obs], rating_thresholds[ij_idx][c_obs+1]
            ) + 1e-10);
        }
    }
}

generated quantities {
    real log_lik_ratings_obs = 0;
    real log_lik_pairwise_obs = 0;
    real total_log_lik;

    array[N_missing_ratings] int<lower=1, upper=C>  missing_rating_predictions;
    array[N_missing_ratings] vector[C]               missing_rating_probs;

    for (n in 1:N_ratings) {
        int i      = rating_attributes[n];
        int j      = rating_annotators[n];
        int k      = rating_items[n];
        int ij_idx = (i-1)*J + j;
        int c_obs  = rating_values[n];

        if (use_dawid_skene_noise == 1) {
            real mix = 0.0;
            for (c_lat in 1:C)
                mix += ordinal_bin_prob(
                    base_scores_attr[i, k], total_std_attr[i], sigma_noise,
                    rating_thresholds[ij_idx][c_lat], rating_thresholds[ij_idx][c_lat+1]
                ) * confusion_matrix[c_lat][c_obs];
            log_lik_ratings_obs += log(mix + 1e-10);
        } else {
            log_lik_ratings_obs += log(ordinal_bin_prob(
                base_scores_attr[i, k], total_std_attr[i], sigma_noise,
                rating_thresholds[ij_idx][c_obs], rating_thresholds[ij_idx][c_obs+1]
            ) + 1e-10);
        }
    }

    total_log_lik = log_lik_ratings_obs;

    for (n in 1:N_missing_ratings) {
        int i      = missing_rating_attributes[n];
        int j      = missing_rating_annotators[n];
        int k      = missing_rating_items[n];
        int ij_idx = (i-1)*J + j;

        real base_score = base_scores_attr[i, k];
        vector[C] bin_probs;
        for (c in 1:C)
            bin_probs[c] = ordinal_bin_prob(
                base_score, total_std_attr[i], sigma_noise,
                rating_thresholds[ij_idx][c], rating_thresholds[ij_idx][c+1]
            );

        if (use_dawid_skene_noise == 1) {
            for (c_obs in 1:C) {
                real p = 0.0;
                for (c_lat in 1:C)
                    p += bin_probs[c_lat] * confusion_matrix[c_lat][c_obs];
                missing_rating_probs[n][c_obs] = p;
            }
            {
                int sampled_latent = categorical_rng(bin_probs);
                missing_rating_predictions[n] = categorical_rng(confusion_matrix[sampled_latent]);
            }
        } else {
            missing_rating_probs[n] = bin_probs;
            missing_rating_predictions[n] = categorical_rng(bin_probs);
        }
    }
}

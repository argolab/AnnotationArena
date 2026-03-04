/*
 * Version A domain model: prototype–style rubric with additive μ_{i,m,s}
 * and full mixture ordinal probit over prototypes and styles.
 *
 * This model is designed to pair with `discrete_type_data_generation.stan`.
 * It replaces the embedding dot-product story with a rubric table:
 *
 *   μ_{i,m,s} = a_i + u_m + v_s + δ_{i,m,s}
 *
 *   And the score comes from binning a noised version:
 *   rating = Bin(μ_{i,m,s} + rubric_fuzz + measurement_noise)
 *
 * The likelihood uses a full mixture over prototypes m and styles s:
 *   P(y=c | i,j,k) = Σ_m Σ_s w_k[m] * w_j[s] * P(y=c | μ_{i,m,s}, σ_tot)
 *   where σ_tot^2 = σ_rubric_fuzz^2 + σ_measurement^2
 *
 * Uses soft prototype/style assignments for items/annotators so the
 * model remains fully continuous (Stan does not support discrete params).
 */

data {
    // Dimensions
    int<lower=1> K;  // number of items in this instance
    int<lower=1> I;  // number of criteria (attributes)
    int<lower=1> J;  // number of annotators  
    int<lower=1> D;  // embedding dimension (kept for compatibility; unused)
    int<lower=1> C;  // number of rating categories

    // Prototype / style dimensions (must match generator)
    int<lower=1> M;  // number of item prototypes
    int<lower=1> S;  // number of annotator styles
    
    // Observed ratings
    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=I> rating_attributes;
    array[N_ratings] int<lower=1, upper=J> rating_annotators;
    array[N_ratings] int<lower=1, upper=K> rating_items;
    array[N_ratings] int<lower=1, upper=C> rating_values;
    
    // Observed pairwise rankings (Bradley-Terry model)
    int<lower=0> N_pairwise_rankings;
    array[N_pairwise_rankings] int<lower=1, upper=I> pairwise_ranking_attributes;
    array[N_pairwise_rankings] int<lower=1, upper=J> pairwise_ranking_annotators;
    array[N_pairwise_rankings, 2] int<lower=1, upper=K> pairwise_ranking_items;  // [item1, item2]
    array[N_pairwise_rankings] int<lower=1, upper=2> pairwise_ranking_orders;    // 1 if item1 > item2, 2 if item2 > item1
    
    // Missing variables to predict (optional)
    int<lower=0> N_missing_ratings;
    array[N_missing_ratings] int<lower=1, upper=I> missing_rating_attributes;
    array[N_missing_ratings] int<lower=1, upper=J> missing_rating_annotators;
    array[N_missing_ratings] int<lower=1, upper=K> missing_rating_items;
    
    int<lower=0> N_missing_pairwise_rankings;
    array[N_missing_pairwise_rankings] int<lower=1, upper=I> missing_pairwise_ranking_attributes;
    array[N_missing_pairwise_rankings] int<lower=1, upper=J> missing_pairwise_ranking_annotators;
    array[N_missing_pairwise_rankings, 2] int<lower=1, upper=K> missing_pairwise_ranking_items;
    
    // Hyperparameters
    real<lower=0> sigma_annotator;   // kept for compatibility; not used directly
    real<lower=0> sigma_measurement;
    real<lower=0> kappa;
    real<lower=0> temperature;
}

transformed data {
    // Debug printing flag (0 = off, 1 = on). Toggle here.
    int DEBUG_PRINT;
    DEBUG_PRINT = 0;
}

parameters {
    // Criterion biases: a_i
    vector[I] a_attr;

    // Prototype qualities: u_m
    vector[M] u_proto;

    // Style biases: v_s
    vector[S] v_style;

    // Residual interactions: δ_{i,m,s}, flattened over (i,m,s)
    vector[I * M * S] delta_ims;

    // Soft prototype assignments for items: each item has a simplex over M
    // (continuous relaxation of discrete z_k)
    array[K] simplex[M] proto_weights;

    // Soft style assignments for annotators: each annotator has a simplex over S
    // (continuous relaxation of discrete s_j)
    array[J] simplex[S] style_weights;

    // Rating probabilities: p_ij ~ Dir(α/C, ..., α/C)
    array[I*J] simplex[C] rating_probs;
    
    // Rubric fuzz: small noise added at rubric lookup time
    // This is separate from measurement noise and captures uncertainty
    // in the rubric table lookup itself.
    real<lower=0> sigma_rubric_fuzz;
}

transformed parameters {
    // Rubric means μ_{i,m,s} = a_i + u_m + v_s + δ_{i,m,s}
    vector[I * M * S] mu_ims;

    // Base utility scores: z_ijk derived from μ and soft assignments
    matrix[I*J, K] base_scores;

    // Rating thresholds in z-space
    array[I*J] vector[C+1] rating_thresholds;

    // Build μ_{i,m,s}
    {
        int idx;
        for (i in 1:I) {
            for (m in 1:M) {
                for (s in 1:S) {
                    idx = ((i-1) * M + (m-1)) * S + s;
                    mu_ims[idx] = a_attr[i] + u_proto[m] + v_style[s] + delta_ims[idx];
                }
            }
        }
    }

    // Compute base_scores[i,j,k] as expectation over (m,s)
    for (i in 1:I) {
        for (j in 1:J) {
            int ij_idx = (i-1) * J + j;
            for (k in 1:K) {
                real z = 0;
                for (m in 1:M) {
                    for (s in 1:S) {
                        int idx_ims = ((i-1) * M + (m-1)) * S + s;
                        z += mu_ims[idx_ims] * proto_weights[k][m] * style_weights[j][s];
                    }
                }
                base_scores[ij_idx, k] = z;
            }
        }
    }

    // Convert rating probabilities to thresholds via inverse normal CDF
    for (ij in 1:(I*J)) {
        rating_thresholds[ij][1] = negative_infinity();
        for (c in 2:C) {
            real cum_prob = sum(rating_probs[ij][1:(c-1)]);
            cum_prob = fmin(fmax(cum_prob, 1e-6), 1.0 - 1e-6);
            rating_thresholds[ij][c] = inv_Phi(cum_prob);
        }
        rating_thresholds[ij][C+1] = positive_infinity();
    }

    // ===== DEBUG: optional summaries per draw =====
    if (DEBUG_PRINT == 1) {
        int max_i = (I < 2) ? I : 2;
        int max_j = (J < 2) ? J : 2;
        for (i in 1:max_i) {
            for (j in 1:max_j) {
                int ij_idx = (i-1) * J + j;
                real min_bs = positive_infinity();
                real max_bs = negative_infinity();
                real sum_bs = 0;
                for (k in 1:K) {
                    real v = base_scores[ij_idx, k];
                    sum_bs += v;
                    if (v < min_bs) min_bs = v;
                    if (v > max_bs) max_bs = v;
                }
                print("[DEBUG SUMMARY] ij=", ij_idx,
                      " base_scores mean=", sum_bs / K,
                      " min=", min_bs, " max=", max_bs);
            }
        }
    }
}

model {
    // ===== PRIORS =====

    // Criterion biases
    a_attr ~ normal(0, 1);

    // Prototype and style strengths (Version A knobs)
    // Using Cauchy priors for flatter, weakly informative priors
    u_proto ~ cauchy(0, 2.0);
    v_style ~ cauchy(0, 1.5);

    // Residual interactions
    delta_ims ~ cauchy(0, 0.5);

    // Soft prototype/style assignments:
    for (k in 1:K) {
        proto_weights[k] ~ dirichlet(rep_vector(0.8, M));   // encourages near-sparse mixtures
    }
    for (j in 1:J) {
        style_weights[j] ~ dirichlet(rep_vector(0.8, S));
    }

    // Rating probabilities: p_ij ~ Dir(α/C, ..., α/C)
    for (ij in 1:(I*J)) {
        rating_probs[ij] ~ dirichlet(rep_vector(kappa / C, C));
    }
    
    // Rubric fuzz prior: relatively flat, weakly informative prior
    // Using half-Cauchy which is standard for scale parameters
    sigma_rubric_fuzz ~ cauchy(0, 1);

    // ===== LIKELIHOODS =====

    // 1. Rating likelihood (full mixture over prototypes and styles)
    //
    // For each rating y_{ijk}, we treat the latent score as a mixture:
    //   score | (i,j,k,m,s) ~ Normal(μ_{i,m,s}, sigma_tot^2)
    //   where sigma_tot^2 = sigma_rubric_fuzz^2 + sigma_measurement^2
    // with mixing weights proto_weights[k][m] * style_weights[j][s].
    // The probability of category c is the mixture of Normal bin probabilities
    // between rating_thresholds[ij][c] and rating_thresholds[ij][c+1].
    {
        real sigma_tot = sqrt(sigma_rubric_fuzz * sigma_rubric_fuzz +
                              sigma_measurement * sigma_measurement);
        
        for (n in 1:N_ratings) {
            int i = rating_attributes[n];
            int j = rating_annotators[n];
            int k = rating_items[n];
            int c = rating_values[n];
            int ij_idx = (i-1) * J + j;
            
            real lower_th = rating_thresholds[ij_idx][c];
            real upper_th = rating_thresholds[ij_idx][c+1];

            // Mixture log probability over (m,s)
            real log_mixture = negative_infinity();
            for (m in 1:M) {
                for (s in 1:S) {
                    int idx_ims = ((i-1) * M + (m-1)) * S + s;
                    real mu = mu_ims[idx_ims];

                    real upper_prob;
                    real lower_prob;

                    if (upper_th == positive_infinity()) {
                        upper_prob = 1.0;
                    } else {
                        upper_prob = Phi((upper_th - mu) / sigma_tot);
                    }
                    if (lower_th == negative_infinity()) {
                        lower_prob = 0.0;
                    } else {
                        lower_prob = Phi((lower_th - mu) / sigma_tot);
                    }

                    real bin_prob = fmax(upper_prob - lower_prob, 1e-12);
                    real log_weight = log(proto_weights[k][m]) + log(style_weights[j][s]);

                    log_mixture = log_sum_exp(log_mixture, log_weight + log(bin_prob));
                }
            }

            target += log_mixture;
        }
    }

    // 2. Pairwise ranking likelihood (Bradley–Terry with Gumbel story)
    for (n in 1:N_pairwise_rankings) {
        int i = pairwise_ranking_attributes[n];
        int j = pairwise_ranking_annotators[n];
        int ij_idx = (i-1) * J + j;

        int item1 = pairwise_ranking_items[n, 1];
        int item2 = pairwise_ranking_items[n, 2];
        int order = pairwise_ranking_orders[n];

        real score1 = base_scores[ij_idx, item1] / temperature;
        real score2 = base_scores[ij_idx, item2] / temperature;

        if (order == 1) {
            target += log_inv_logit(score1 - score2);
        } else if (order == 2) {
            target += log_inv_logit(score2 - score1);
        } else {
            reject("Error: pairwise_ranking_orders[n] must be 1 or 2.");
        }
    }
}

generated quantities {
    // Log-likelihood components
    real log_lik_ratings_obs = 0;
    real log_lik_pairwise_obs = 0;
    real total_log_lik = 0;
    
    // Posterior predictive samples for missing variables
    array[N_missing_ratings] int<lower=1, upper=C> missing_rating_predictions;
    array[N_missing_pairwise_rankings] int<lower=1, upper=2> missing_pairwise_ranking_predictions;
    
    // Predicted distributions for missing variables
    array[N_missing_ratings] vector[C] missing_rating_probs;
    array[N_missing_pairwise_rankings] real missing_pairwise_logits;

    // Combined standard deviation for mixture components
    real sigma_tot = sqrt(sigma_rubric_fuzz * sigma_rubric_fuzz +
                          sigma_measurement * sigma_measurement);

    // ===== Observed rating log-likelihood (using mixture) =====
    for (n in 1:N_ratings) {
        int i = rating_attributes[n];
        int j = rating_annotators[n];
        int k = rating_items[n];
        int c = rating_values[n];
        int ij_idx = (i-1) * J + j;
        
        real lower_th = rating_thresholds[ij_idx][c];
        real upper_th = rating_thresholds[ij_idx][c+1];

        // Mixture log probability over (m,s) - same as model block
        real log_mixture = negative_infinity();
        for (m in 1:M) {
            for (s in 1:S) {
                int idx_ims = ((i-1) * M + (m-1)) * S + s;
                real mu = mu_ims[idx_ims];

                real upper_prob;
                real lower_prob;

                if (upper_th == positive_infinity()) {
                    upper_prob = 1.0;
                } else {
                    upper_prob = Phi((upper_th - mu) / sigma_tot);
                }
                if (lower_th == negative_infinity()) {
                    lower_prob = 0.0;
                } else {
                    lower_prob = Phi((lower_th - mu) / sigma_tot);
                }

                real bin_prob = fmax(upper_prob - lower_prob, 1e-12);
                real log_weight = log(proto_weights[k][m]) + log(style_weights[j][s]);

                log_mixture = log_sum_exp(log_mixture, log_weight + log(bin_prob));
            }
        }

        log_lik_ratings_obs += log_mixture;

        if (DEBUG_PRINT == 1 && n <= 10) {
            print("[DEBUG RATING] n=", n,
                  " i=", i, " j=", j, " k=", k, " ij_idx=", ij_idx,
                  " base_score=", base_scores[ij_idx, k],
                  " lower_th=", lower_th, " upper_th=", upper_th,
                  " value=", c);
        }
    }

    // ===== Observed pairwise log-likelihood =====
    for (n in 1:N_pairwise_rankings) {
        int i = pairwise_ranking_attributes[n];
        int j = pairwise_ranking_annotators[n];
        int ij_idx = (i-1) * J + j;

        int item1 = pairwise_ranking_items[n, 1];
        int item2 = pairwise_ranking_items[n, 2];
        int order = pairwise_ranking_orders[n];

        real score1 = base_scores[ij_idx, item1] / temperature;
        real score2 = base_scores[ij_idx, item2] / temperature;

        if (order == 1) {
            log_lik_pairwise_obs += log_inv_logit(score1 - score2);
        } else {
            log_lik_pairwise_obs += log_inv_logit(score2 - score1);
        }

        if (DEBUG_PRINT == 1 && n <= 10) {
            real logit12 = score1 - score2;
            real p12 = inv_logit(logit12);
            print("[DEBUG PAIR] n=", n,
                  " i=", i, " j=", j, " ij_idx=", ij_idx,
                  " item1=", item1, " item2=", item2, " order=", order,
                  " score1_T=", score1, " score2_T=", score2,
                  " logit=", logit12, " p12=", p12);
        }
    }

    total_log_lik = log_lik_ratings_obs + log_lik_pairwise_obs;

    // ===== Posterior predictive for missing ratings =====
    // For missing ratings, we compute the mixture probability over (m,s)
    // and sample from the predictive distribution.
    for (n in 1:N_missing_ratings) {
        int i = missing_rating_attributes[n];
        int j = missing_rating_annotators[n];
        int k = missing_rating_items[n];
        int ij_idx = (i-1) * J + j;

        // Compute mixture probability for each category c
        for (c in 1:C) {
            real lower_th = rating_thresholds[ij_idx][c];
            real upper_th = rating_thresholds[ij_idx][c+1];

            real log_mixture = negative_infinity();
            for (m in 1:M) {
                for (s in 1:S) {
                    int idx_ims = ((i-1) * M + (m-1)) * S + s;
                    real mu = mu_ims[idx_ims];

                    real upper_prob;
                    real lower_prob;

                    if (upper_th == positive_infinity()) {
                        upper_prob = 1.0;
                    } else {
                        upper_prob = Phi((upper_th - mu) / sigma_tot);
                    }
                    if (lower_th == negative_infinity()) {
                        lower_prob = 0.0;
                    } else {
                        lower_prob = Phi((lower_th - mu) / sigma_tot);
                    }

                    real bin_prob = fmax(upper_prob - lower_prob, 1e-12);
                    real log_weight = log(proto_weights[k][m]) + log(style_weights[j][s]);

                    log_mixture = log_sum_exp(log_mixture, log_weight + log(bin_prob));
                }
            }

            missing_rating_probs[n][c] = exp(log_mixture);
        }

        // Sample a rating from the predictive distribution
        // Use base_scores (expectation over mixture) directly
        real base_score = base_scores[ij_idx, k];
        real noisy_score = base_score + normal_rng(0, sigma_tot);
        
        int rating = 1;
        for (c in 1:C) {
            if (noisy_score <= rating_thresholds[ij_idx][c+1]) {
                rating = c;
                break;
            }
        }
        missing_rating_predictions[n] = rating;

        if (DEBUG_PRINT == 1 && n <= 10) {
            print("[DEBUG MRATING] n=", n,
                  " i=", i, " j=", j, " k=", k, " ij_idx=", ij_idx,
                  " base_score=", base_score, " noisy_score=", noisy_score,
                  " thresholds=", rating_thresholds[ij_idx],
                  " probs=", missing_rating_probs[n],
                  " sampled=", rating);
        }
    }

    // ===== Posterior predictive for missing pairwise rankings =====
    for (n in 1:N_missing_pairwise_rankings) {
        int i = missing_pairwise_ranking_attributes[n];
        int j = missing_pairwise_ranking_annotators[n];
        int ij_idx = (i-1) * J + j;

        int item1 = missing_pairwise_ranking_items[n, 1];
        int item2 = missing_pairwise_ranking_items[n, 2];

        real score1 = base_scores[ij_idx, item1] / temperature;
        real score2 = base_scores[ij_idx, item2] / temperature;

        // Predicted logit for P(item1 > item2)
        missing_pairwise_logits[n] = score1 - score2;

        // Sample ranking using Gumbel noise
        real gumbel1 = -log(-log(uniform_rng(0, 1)));
        real gumbel2 = -log(-log(uniform_rng(0, 1)));

        real utility1 = score1 + gumbel1;
        real utility2 = score2 + gumbel2;

        missing_pairwise_ranking_predictions[n] = (utility1 > utility2) ? 1 : 2;

        if (DEBUG_PRINT == 1 && n <= 10) {
            real logit12 = score1 - score2;
            real p12 = inv_logit(logit12);
            print("[DEBUG MPAIR] n=", n,
                  " i=", i, " j=", j, " ij_idx=", ij_idx,
                  " item1=", item1, " item2=", item2,
                  " score1_T=", score1, " score2_T=", score2,
                  " logit=", logit12, " p12=", p12,
                  " sampled_order=", missing_pairwise_ranking_predictions[n]);
        }
    }
}


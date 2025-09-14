/*
 * Domain model for mixed annotation inference using MCMC
 * Learns embeddings and preferences from ratings, comparisons, and rankings
 */

data {
    // Dimensions
    int<lower=1> K;  // number of items
    int<lower=1> I;  // number of attributes
    int<lower=1> J;  // number of annotators  
    int<lower=1> D;  // embedding dimension
    int<lower=1> C;  // number of rating categories
    int<lower=2> ranking_size;  // size of ranking sets
    
    // Observed ratings
    int<lower=0> N_ratings;
    array[N_ratings] int<lower=1, upper=I> rating_attributes;
    array[N_ratings] int<lower=1, upper=J> rating_annotators;
    array[N_ratings] int<lower=1, upper=K> rating_items;
    array[N_ratings] int<lower=1, upper=C> rating_values;
    
    
    // Observed rankings
    int<lower=0> N_rankings;
    array[N_rankings] int<lower=1, upper=I> ranking_attributes;
    array[N_rankings] int<lower=1, upper=J> ranking_annotators;
    array[N_rankings, ranking_size] int<lower=1, upper=K> ranking_items;
    array[N_rankings, ranking_size] int<lower=1, upper=ranking_size> ranking_orders;
    
    // Hyperparameters
    real<lower=0> sigma_annotator;
    real<lower=0> sigma_measurement;
    real<lower=0> alpha_dirichlet;
    real<lower=0> temperature;
    
    // Prior scales
    // Use this as a paramters?
    real<lower=0> sigma_embedding_prior;
    real<lower=0> sigma_preference_prior;
}

parameters {
    // Latent embeddings (unit norm constraint for identification)
    matrix[K, D] embeddings_raw;
    
    // Mean preferences per attribute
    matrix[I, D] mean_preferences;
    
    // Annotator-specific preferences
    matrix[I*J, D] annotator_preferences;
    
    // Rating thresholds - first threshold fixed at 0 for identification
    array[I*J, C-2] real rating_thresholds_increments;
}

transformed parameters {
    // Unit-normalized embeddings for identification
    matrix[K, D] embeddings;
    
    // Base utility scores  
    matrix[I*J, K] base_scores;
    
    // Rating thresholds with -inf and +inf boundaries
    array[I*J] vector[C+1] rating_thresholds;
    
    // Normalize embeddings to unit norm for identification
    for (k in 1:K) {
        real norm = sqrt(dot_self(embeddings_raw[k]));
        if (norm > 1e-10) {
            embeddings[k] = embeddings_raw[k] / norm;
        } else {
            embeddings[k] = embeddings_raw[k];
        }
    }
    
    // Compute base scores: z_ij_k = v_ij · e_k  
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], embeddings[k]);
            }
        }
    }
    
    // Construct ordered thresholds with identification constraints
    for (ij in 1:(I*J)) {
        rating_thresholds[ij][1] = negative_infinity();  // -∞ for category 1
        
        // First threshold FIXED at 0 for identification
        rating_thresholds[ij][2] = 0.0;
        
        // Subsequent thresholds using positive increments
        for (c in 3:C) {
            rating_thresholds[ij][c] = rating_thresholds[ij][c-1] + abs(rating_thresholds_increments[ij, c-2]);
        }
        
        rating_thresholds[ij][C+1] = positive_infinity();  // +∞ for category C
    }
}

model {
    // ===== PRIORS =====
    
    // Raw embeddings: e_k ~ N(0, σ_e²I) (will be normalized)
    for (k in 1:K) {
        embeddings_raw[k] ~ normal(0, sigma_embedding_prior);
    }
    
    // Mean preferences: v_i ~ N(0, σ_v²I) 
    for (i in 1:I) {
        mean_preferences[i] ~ normal(0, sigma_preference_prior);
    }
    
    // Annotator preferences: v_ij ~ N(v_i, σ_a²I)
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            annotator_preferences[idx] ~ normal(mean_preferences[i], sigma_annotator);
        }
    }
    
    // Rating threshold increments (positive spacings)
    for (ij in 1:(I*J)) {
        for (c in 1:(C-2)) {
            rating_thresholds_increments[ij, c] ~ normal(0, 0.5);  // Moderate spacing
        }
    }
    
    // ===== LIKELIHOODS =====
    
    // 1. RATING LIKELIHOOD
    for (n in 1:N_ratings) {
        int i = rating_attributes[n];
        int j = rating_annotators[n]; 
        int k = rating_items[n];
        int c = rating_values[n];
        int ij_idx = (i-1)*J + j;
        
        // Base score: z_ijk = v_ij · e_k
        real base_score = base_scores[ij_idx, k];
        
        // Rating likelihood: P(rating = c) = Φ((Q_c - z)/σ_m) - Φ((Q_{c-1} - z)/σ_m)
        // where Q_c are the thresholds and ε ~ N(0, σ_m²)
        real upper_threshold = rating_thresholds[ij_idx][c+1];
        real lower_threshold = rating_thresholds[ij_idx][c];
        
        real upper_prob, lower_prob;
        
        if (upper_threshold == positive_infinity()) {
            upper_prob = 1.0;
        } else {
            upper_prob = Phi((upper_threshold - base_score) / sigma_measurement);
        }
        
        if (lower_threshold == negative_infinity()) {
            lower_prob = 0.0;
        } else {
            lower_prob = Phi((lower_threshold - base_score) / sigma_measurement);
        }
        
        real bin_prob = upper_prob - lower_prob;
        
        // Numerical stability
        if (bin_prob > 1e-8) {
            target += log(bin_prob);
        } else {
            target += log(1e-8);
        }
    }
    
    // 2. PAIRWISE RANKING LIKELIHOOD (Simplified for binary preferences)
    for (n in 1:N_rankings) {
        int i = ranking_attributes[n];
        int j = ranking_annotators[n];
        int ij_idx = (i-1)*J + j;
        
        // For pairwise: ranking_items[n] = [item1, item2], ranking_orders[n] = [1, 2] or [2, 1]
        int item1 = ranking_items[n, 1];
        int item2 = ranking_items[n, 2];
        real score1 = base_scores[ij_idx, item1] / temperature;
        real score2 = base_scores[ij_idx, item2] / temperature;
        
        // If order = [1, 2], item1 > item2, so P(item1 > item2) = sigmoid(score1 - score2)
        // If order = [2, 1], item2 > item1, so P(item2 > item1) = sigmoid(score2 - score1)
        if (ranking_orders[n, 1] == 1) {  // item1 ranks first
            target += log_inv_logit(score1 - score2);
        } else {  // item2 ranks first
            target += log_inv_logit(score2 - score1);
        }
    }
}

generated quantities {
    // Log-likelihood components for evaluation
    real log_lik_ratings = 0;
    real log_lik_rankings = 0;
    real total_log_lik = 0;
    
    // Compute log-likelihoods (same as in model block)
    for (n in 1:N_ratings) {
        int i = rating_attributes[n];
        int j = rating_annotators[n];
        int k = rating_items[n];
        int c = rating_values[n];
        int ij_idx = (i-1)*J + j;
        
        real base_score = base_scores[ij_idx, k];
        real upper_threshold = rating_thresholds[ij_idx][c+1];
        real lower_threshold = rating_thresholds[ij_idx][c];
        
        real upper_prob, lower_prob;
        
        if (upper_threshold == positive_infinity()) {
            upper_prob = 1.0;
        } else {
            upper_prob = Phi((upper_threshold - base_score) / sigma_measurement);
        }
        
        if (lower_threshold == negative_infinity()) {
            lower_prob = 0.0;
        } else {
            lower_prob = Phi((lower_threshold - base_score) / sigma_measurement);
        }
        
        real bin_prob = upper_prob - lower_prob;
        log_lik_ratings += log(bin_prob + 1e-10);
    }
    
    for (n in 1:N_rankings) {
        int i = ranking_attributes[n];
        int j = ranking_annotators[n];
        int ij_idx = (i-1)*J + j;
        
        // Same simplified pairwise logic as in model block
        int item1 = ranking_items[n, 1];
        int item2 = ranking_items[n, 2];
        real score1 = base_scores[ij_idx, item1] / temperature;
        real score2 = base_scores[ij_idx, item2] / temperature;
        
        if (ranking_orders[n, 1] == 1) {  // item1 ranks first
            log_lik_rankings += log_inv_logit(score1 - score2);
        } else {  // item2 ranks first
            log_lik_rankings += log_inv_logit(score2 - score1);
        }
    }
    
    total_log_lik = log_lik_ratings + log_lik_rankings;
}
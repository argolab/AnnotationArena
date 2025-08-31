/*
 * Basic data generation model for ranking system
 * Implements the core mathematical framework for generating synthetic annotations
 */

data {
    // Dimensions
    int<lower=1> K;  // number of items
    int<lower=1> I;  // number of attributes
    int<lower=1> J;  // number of annotators  
    int<lower=1> D;  // embedding dimension
    int<lower=1> C;  // number of rating categories
    
    // Query generation parameters
    int<lower=1> N_ratings;      // number of rating queries to generate
    int<lower=1> N_comparisons;  // number of pairwise comparison queries
    int<lower=1> N_rankings;     // number of ranking queries
    int<lower=3> ranking_size;   // size of sets to rank (e.g., 5 items per ranking)
    
    // Hyperparameters
    real<lower=0> sigma_annotator;     // σ² - annotator preference variance
    real<lower=0> sigma_measurement;   // σ²ⱼ - measurement error variance (same for all j for now)
    real<lower=0> alpha_dirichlet;     // α - concentration for rating bins
    real<lower=0> temperature;         // temperature for Plackett-Luce rankings
}

generated quantities {
    // True model components
    matrix[K, D] embeddings;           // eₖ ~ N(0,I)
    matrix[I, D] mean_preferences;     // vᵢ ~ N(0,I)
    matrix[I*J, D] annotator_preferences;  // vᵢⱼ ~ N(vᵢ, σ²I)
    
    // Rating thresholds per annotator-attribute pair
    array[I*J] simplex[C] rating_probs;     // pᵢⱼ ~ Dir(α/C, ..., α/C)
    array[I*J] vector[C] rating_thresholds; // qᵢⱼ = cumsum(pᵢⱼ)
    
    // Base scores for all annotator-attribute-item combinations
    matrix[I*J, K] base_scores;        // zᵢⱼₖ = vᵢⱼ · eₖ
    
    // UNARY RATINGS: Individual item ratings
    array[N_ratings] int<lower=1, upper=I> rating_attributes;   // which attribute
    array[N_ratings] int<lower=1, upper=J> rating_annotators;   // which annotator  
    array[N_ratings] int<lower=1, upper=K> rating_items;        // which item
    array[N_ratings] int<lower=1, upper=C> rating_values;       // rating result
    
    // PAIRWISE COMPARISONS: Item A vs Item B
    array[N_comparisons] int<lower=1, upper=I> comparison_attributes;
    array[N_comparisons] int<lower=1, upper=J> comparison_annotators;
    array[N_comparisons] int<lower=1, upper=K> comparison_item_a;
    array[N_comparisons] int<lower=1, upper=K> comparison_item_b;
    array[N_comparisons] int<lower=0, upper=1> comparison_results;  // 1 if A > B, 0 if B > A
    
    // LISTWISE RANKINGS: Full rankings of item sets
    array[N_rankings] int<lower=1, upper=I> ranking_attributes;
    array[N_rankings] int<lower=1, upper=J> ranking_annotators;
    array[N_rankings, ranking_size] int<lower=1, upper=K> ranking_item_sets;  // which items to rank
    array[N_rankings, ranking_size] int<lower=1, upper=ranking_size> ranking_orders;  // ranking result
    
    // ===== STEP 1: Generate true model components =====
    
    // Generate embeddings: eₖ ~ N(0,I)
    for (k in 1:K) {
        for (d in 1:D) {
            embeddings[k, d] = normal_rng(0, 1);
        }
    }
    
    // Generate mean preferences: vᵢ ~ N(0,I) 
    for (i in 1:I) {
        for (d in 1:D) {
            mean_preferences[i, d] = normal_rng(0, 1);
        }
    }
    
    // Generate annotator-specific preferences: vᵢⱼ ~ N(vᵢ, σ²I)
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;  // Convert to linear index
            for (d in 1:D) {
                annotator_preferences[idx, d] = normal_rng(mean_preferences[i, d], sigma_annotator);
            }
        }
    }
    
    // Generate rating thresholds: pᵢⱼ ~ Dir(α/C, ..., α/C)
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            rating_probs[idx] = dirichlet_rng(rep_vector(alpha_dirichlet/C, C));
            rating_thresholds[idx] = cumulative_sum(rating_probs[idx]);
        }
    }
    
    // Compute base scores: zᵢⱼₖ = vᵢⱼ · eₖ
    for (i in 1:I) {
        for (j in 1:J) {
            int idx = (i-1)*J + j;
            for (k in 1:K) {
                base_scores[idx, k] = dot_product(annotator_preferences[idx], embeddings[k]);
            }
        }
    }
    
    // ===== STEP 2: Generate unary ratings =====
    for (n in 1:N_ratings) {
        // Randomly select attribute, annotator, item
        rating_attributes[n] = categorical_rng(rep_vector(1.0/I, I));
        rating_annotators[n] = categorical_rng(rep_vector(1.0/J, J));
        rating_items[n] = categorical_rng(rep_vector(1.0/K, K));
        
        // Get indices
        int i = rating_attributes[n];
        int j = rating_annotators[n]; 
        int k = rating_items[n];
        int idx = (i-1)*J + j;
        
        // Add measurement noise and bin
        real noisy_score = base_scores[idx, k] + normal_rng(0, sigma_measurement);
        real pref_norm = sqrt(dot_self(annotator_preferences[idx]));
        real total_std = sqrt(pref_norm^2 + sigma_measurement^2);
        real standardized_score = noisy_score / total_std;
        real cdf_val = Phi(standardized_score);
        
        // Bin into categories
        rating_values[n] = 1;
        for (c in 1:C) {
            if (cdf_val <= rating_thresholds[idx][c]) {
                rating_values[n] = c;
                break;
            }
        }
    }
    
    // ===== STEP 3: Generate pairwise comparisons =====
    for (n in 1:N_comparisons) {
        // Randomly select attribute, annotator, and two different items
        comparison_attributes[n] = categorical_rng(rep_vector(1.0/I, I));
        comparison_annotators[n] = categorical_rng(rep_vector(1.0/J, J));
        comparison_item_a[n] = categorical_rng(rep_vector(1.0/K, K));
        
        // Ensure item B is different from item A
        comparison_item_b[n] = comparison_item_a[n];
        while (comparison_item_b[n] == comparison_item_a[n]) {
            comparison_item_b[n] = categorical_rng(rep_vector(1.0/K, K));
        }
        
        // Get scores with noise
        int i = comparison_attributes[n];
        int j = comparison_annotators[n];
        int k_a = comparison_item_a[n];
        int k_b = comparison_item_b[n];
        int idx = (i-1)*J + j;
        
        real score_a = base_scores[idx, k_a] + normal_rng(0, sigma_measurement);
        real score_b = base_scores[idx, k_b] + normal_rng(0, sigma_measurement);
        
        // Compare: 1 if A > B, 0 if B > A
        comparison_results[n] = (score_a > score_b) ? 1 : 0;
    }
    
    // ===== STEP 4: Generate listwise rankings =====  
    for (n in 1:N_rankings) {
        // Randomly select attribute and annotator
        ranking_attributes[n] = categorical_rng(rep_vector(1.0/I, I));
        ranking_annotators[n] = categorical_rng(rep_vector(1.0/J, J));
        
        int i = ranking_attributes[n];
        int j = ranking_annotators[n];
        int idx = (i-1)*J + j;
        
        // Randomly select items to rank (without replacement)
        array[K] int available_items;
        for (k in 1:K) available_items[k] = k;
        
        for (r in 1:ranking_size) {
            int remaining = K - r + 1;
            int selected_idx = categorical_rng(rep_vector(1.0/remaining, remaining));
            ranking_item_sets[n, r] = available_items[selected_idx];
            
            // Remove selected item by swapping with last
            available_items[selected_idx] = available_items[remaining];
        }
        
        // Generate Plackett-Luce ranking using Gumbel trick
        array[ranking_size] real gumbel_scores;
        for (r in 1:ranking_size) {
            int k = ranking_item_sets[n, r];
            real base_score = base_scores[idx, k];
            // Add Gumbel noise: -log(-log(uniform(0,1)))
            real u = uniform_rng(0, 1);
            real gumbel = -log(-log(u));
            gumbel_scores[r] = base_score / temperature + gumbel;
        }
        
        // Sort to get ranking order (higher scores get lower rank numbers)
        array[ranking_size] int sorted_indices;
        for (r in 1:ranking_size) sorted_indices[r] = r;
        
        // Simple bubble sort to get ranking
        for (i_sort in 1:(ranking_size-1)) {
            for (j_sort in 1:(ranking_size-i_sort)) {
                if (gumbel_scores[sorted_indices[j_sort]] < gumbel_scores[sorted_indices[j_sort+1]]) {
                    int temp = sorted_indices[j_sort];
                    sorted_indices[j_sort] = sorted_indices[j_sort+1];
                    sorted_indices[j_sort+1] = temp;
                }
            }
        }
        
        // Store ranking order
        for (r in 1:ranking_size) {
            ranking_orders[n, r] = sorted_indices[r];
        }
    }
}
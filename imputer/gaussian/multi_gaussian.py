import numpy as np
import pandas as pd
import json
import random
from collections import Counter
from tqdm import tqdm
from itertools import combinations

def gibbs_mvnormal(mean, cov, n_samples, burn_in=100):
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    Λ = np.linalg.inv(cov)
    d = len(mean)
    x = mean.copy() 
    samples = np.zeros((n_samples, d))
    
    for t in range(n_samples + burn_in):
        for i in range(d):
            cond_var = 1.0 / Λ[i, i]
            sum_except_i = np.dot(Λ[i, :], x - mean) - Λ[i, i] * (x[i] - mean[i])
            cond_mean = mean[i] - cond_var * sum_except_i
            x[i] = np.random.normal(cond_mean, np.sqrt(cond_var))
            
        if t >= burn_in:
            samples[t - burn_in] = x
            
    return samples

def get_category(value, boundaries):
    if value <= boundaries[0]:
        return 1
    elif value <= boundaries[1]:
        return 2
    elif value <= boundaries[2]:
        return 3
    elif value <= boundaries[3]:
        return 4
    else:
        return 5

def generate_subset_observation_patterns(known_indices):
    """Generate all possible observation patterns that are subsets of known_indices"""
    patterns = []
    for r in range(len(known_indices) + 1):  # From 0 to len(known_indices) observed variables
        patterns.extend(list(combinations(known_indices, r)))
    return patterns

def compute_marginal_distribution(sample, observed_indices, boundaries_list, pool_samples, mean, cov):
    """Compute marginal distribution of unobserved variables given observed ones"""
    masked_indices = [i for i in range(10) if i not in observed_indices]
    
    if not masked_indices:  # All variables observed
        return {}
    
    if not observed_indices:  # No variables observed - return uniform
        return {i: [0.2, 0.2, 0.2, 0.2, 0.2] for i in masked_indices}
    
    true_categories = [sample[i] for i in observed_indices]
    valid_samples = []
    category_counts = {i: [0, 0, 0, 0, 0] for i in masked_indices}
    
    shuffled_pool = pool_samples.copy()
    np.random.shuffle(shuffled_pool)
    
    for candidate in shuffled_pool:
        all_match = True
        for idx, i in enumerate(observed_indices):
            category = get_category(candidate[i], boundaries_list[i])
            if category != true_categories[idx]:
                all_match = False
                break
        
        if all_match:
            valid_samples.append(candidate)
            for i in masked_indices:
                category = get_category(candidate[i], boundaries_list[i])
                category_counts[i][category-1] += 1
            
            if len(valid_samples) >= 1000:
                break
    
    if len(valid_samples) < 1000:
        attempts = 0
        max_attempts = 10000
        
        while len(valid_samples) < 1000 and attempts < max_attempts:
            candidate = gibbs_mvnormal(mean, cov, n_samples=1)[0]
            
            all_match = True
            for idx, i in enumerate(observed_indices):
                category = get_category(candidate[i], boundaries_list[i])
                if category != true_categories[idx]:
                    all_match = False
                    break
            
            if all_match:
                valid_samples.append(candidate)
                for i in masked_indices:
                    category = get_category(candidate[i], boundaries_list[i])
                    category_counts[i][category-1] += 1
            
            attempts += 1
    
    marginal_distributions = {}
    for i in masked_indices:
        total = sum(category_counts[i])
        if total > 0:
            marginal_distributions[i] = [count / total for count in category_counts[i]]
        else:
            marginal_distributions[i] = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform fallback
    
    return marginal_distributions

mean = np.array([0.0, 0.5, 1.0, 0.5, 2.0, -0.5, 1.5, -1.0, 0.8, 1.2])
cov = np.array([
    [2.0, 0.5, 0.3, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
    [0.5, 1.5, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
    [0.3, 0.2, 1.0, 0.4, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.4, 1.2, 0.3, 0.0, 0.0, 0.0, 0.4, 0.0],
    [0.0, 0.0, 0.0, 0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.3],
    [0.2, 0.0, 0.0, 0.0, 0.0, 1.1, 0.4, 0.0, 0.0, 0.0],
    [0.0, 0.3, 0.0, 0.0, 0.0, 0.4, 1.3, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.9, 0.3, 0.0],
    [0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.3, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.7]
])

pool_samples = gibbs_mvnormal(mean, cov, n_samples=1000000, burn_in=2000)
print("Sample pool shape:", pool_samples.shape)
print("Empirical covariance:\n", np.cov(pool_samples, rowvar=False))
samples = gibbs_mvnormal(mean, cov, n_samples=1000, burn_in=2000)

df = pd.DataFrame(samples, columns=[f'X{i+1}' for i in range(10)])

categorized = pd.DataFrame()
boundaries_list = []

for col in df.columns:
    col_min = df[col].min()
    col_max = df[col].max()
    boundaries = np.sort(np.random.uniform(col_min, col_max, 4))
    boundaries_list.append(boundaries)
    categorized[col] = pd.cut(
        df[col], 
        bins=[-np.inf] + boundaries.tolist() + [np.inf],
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

categorized_list = categorized.values.tolist()
with open("raw_data.json", "w") as f:
    json.dump(categorized_list, f, indent=2)
print("✅ Saved categorized data to 'raw_data.json'")
with open("raw_data.json", 'r') as file:
    data = json.load(file)

train_data_list = []
dev_data_list = []

for index, sample in tqdm(enumerate(data[:900]), desc="Generating training data"):
    if index % 100 == 0:
        print(f"Processing training sample {index}...")
    
    mask = [random.random() < 0.5 for _ in range(10)]
    if all(mask):
        mask[random.randint(0, 9)] = False
    
    known_indices = [i for i, m in enumerate(mask) if not m]
    masked_indices = [i for i, m in enumerate(mask) if m]
    
    entry = {
        "known_questions": [0 if m else 1 for m in mask],
        "annotators": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "questions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "input": [],
        "answers": []
    }
    for i in range(10):
        answer_vec = [0.0] * 5
        answer_vec[sample[i] - 1] = 1.0
        entry["answers"].append(answer_vec)
        
        if i in known_indices:
            entry["input"].append([0.0] + answer_vec)
        else:
            entry["input"].append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    if known_indices and masked_indices:
        marginal_dist = compute_marginal_distribution(
            sample, known_indices, boundaries_list, pool_samples, mean, cov
        )
        for i in masked_indices:
            if i in marginal_dist:
                entry["answers"][i] = marginal_dist[i]
    elif not known_indices: 
        for i in masked_indices:
            entry["answers"][i] = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    train_data_list.append(entry)

print(f"Generated {len(train_data_list)} training entries")
for index, sample in tqdm(enumerate(data[900:]), desc="Generating dev data"):
    if index % 100 == 0:
        print(f"Processing dev sample {index}...")
    
    # Use a single random masking pattern (ensuring at least one is observable)
    mask = [random.random() < 0.5 for _ in range(10)]
    # Ensure at least one variable is observable
    if all(mask):
        mask[random.randint(0, 9)] = False
    
    known_indices = [i for i, m in enumerate(mask) if not m]
    masked_indices = [i for i, m in enumerate(mask) if m]
    
    entry = {
        "known_questions": [0 if m else 1 for m in mask],
        "annotators": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "questions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "input": [],
        "answers": []
    }
    
    # Prepare the answers from original categorized data
    for i in range(10):
        # Build answer as a one-hot vector of length 5 (using index category-1)
        answer_vec = [0.0] * 5
        answer_vec[sample[i] - 1] = 1.0
        entry["answers"].append(answer_vec)
        if i in known_indices:
            entry["input"].append([0.0] + answer_vec)
        else:
            entry["input"].append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # If we have known variables, compute marginal distribution for masked variables
    if known_indices and masked_indices:
        marginal_dist = compute_marginal_distribution(
            sample, known_indices, boundaries_list, pool_samples, mean, cov
        )
        
        # Update answers for masked variables with marginal probabilities
        for i in masked_indices:
            if i in marginal_dist:
                entry["answers"][i] = marginal_dist[i]
    elif not known_indices:  # No observed variables
        for i in masked_indices:
            entry["answers"][i] = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform
    
    dev_data_list.append(entry)

print(f"Generated {len(dev_data_list)} dev entries")

# Save the datasets
with open("gaussian_train_10.json", "w") as file:
    json.dump(train_data_list, file, indent=4)

with open("gaussian_dev_10.json", "w") as file:
    json.dump(dev_data_list, file, indent=4)

print("✅ Saved to 'gaussian_train.json' and 'gaussian_dev.json'")
print(f"Training data: {len(train_data_list)} entries")
print(f"Dev data: {len(dev_data_list)} entries")
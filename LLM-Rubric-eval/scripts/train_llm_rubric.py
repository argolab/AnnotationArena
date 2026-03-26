from pathlib import Path
import random
import json

import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats 
import argparse
import torch
import sys
sys.path.insert(0, "/home/stone/AnnotationArena/LLM-Rubric")
from llm_rubric import pd_utils
from llm_rubric.model.torch_impl import (
    Hyperparameters, PersonalizedCalibrationNetwork, pretrain_loop, finetune_loop
)


def load_json_data_to_dataframe(json_path: Path) -> pd.DataFrame:
    """
    Load JSON data and convert to dataframe format.

    The JSON contains an 'all_ratings' list of records with fields:
      attribute (1-9), annotator (1-24 human, 25 LLM), item, value, rating_dist

    Returns one row per (item, human_annotator) pair, with LLM (annotator 25)
    ratings as input features and human ratings as output labels.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    ratings = data['all_ratings']

    # Index ratings by item
    from collections import defaultdict
    llm_ratings = defaultdict(dict)       # item -> {attribute -> rating_dist}
    human_ratings = defaultdict(lambda: defaultdict(dict))  # item -> annotator -> {attribute -> rating_dist}

    for r in ratings:
        if r['instance'] != 'train':
            continue
        item = r['item']
        attr = r['attribute'] - 1  # convert 1-9 to 0-indexed 0-8
        if r['annotator'] == 25:
            llm_ratings[item][attr] = r['rating_dist']
        else:
            human_ratings[item][r['annotator']][attr] = r['rating_dist']

    rows = []
    for item, annotators_dict in human_ratings.items():
        text_id = f"text_{item}"
        llm_data = llm_ratings.get(item, {})

        for annotator, questions_dict in annotators_dict.items():
            row = {
                'text_id': text_id,
                'annotator': annotator,
            }

            # Add LLM probability distributions as input features (Q0-Q8, 4 probs each)
            for q in range(9):
                if q in llm_data:
                    for ans_idx, prob in enumerate(llm_data[q], start=1):
                        row[f'Q{q}_{ans_idx}_prob'] = prob
                else:
                    for ans_idx in range(1, 5):
                        row[f'Q{q}_{ans_idx}_prob'] = 0.0

            # Add human answers as output labels (Q0-Q8)
            all_known = True
            for q in range(9):
                if q in questions_dict:
                    answer_probs = questions_dict[q]
                    label = int(np.argmax(answer_probs) + 1)
                    row[f'Q{q}'] = label
                else:
                    row[f'Q{q}'] = -1
                    all_known = False

            row['all_known'] = all_known
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main(
    train_json_path: str,
    model_output_dir: str,
    judge_map_path: str,
    num_questions: int = 9,
    num_answers: int = 4,
    random_seed: int = 43,
    all_data_size: int = None,
    layer1_size: int = None,
    layer2_size: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    pretraining_epochs: int = None,
    finetuning_epochs: int = None,
): 
    # Load training data from JSON
    print(f"Loading training data from {train_json_path}")
    df = load_json_data_to_dataframe(Path(train_json_path))
    
    # For training, we only use rows where we have all human labels
    df = df[df['all_known'] == True].copy()
    
    print(f"Loaded {len(df)} training samples")
    print(f"Unique texts: {df['text_id'].nunique()}")
    print(f"Unique annotators: {df['annotator'].nunique()}")
    
    # Create annotator name column
    df['annotator_name'] = df['annotator'].apply(lambda x: f'annotator_{x}')
    
    # Add judge IDs
    judge_id_map = pd_utils.add_judge_ids(df, 'annotator_name', 'judge_id')
    num_judges = len(judge_id_map)
    print(f"Number of judges: {num_judges}")
    
    # Define input and output criteria
    input_criteria = [f'Q{i}' for i in range(num_questions)]
    output_criteria = [f'Q{i}' for i in range(num_questions)]
    
    print(f"Input size: {len(input_criteria) * num_answers}")
    print(f"Output size: {len(output_criteria)}")
    
    input_size = len(input_criteria) * num_answers
    output_size = len(output_criteria)

    # Set up hyperparameters
    hp_args = {}
    if all_data_size is not None:
        hp_args["all_data_size"] = all_data_size
    else:
        hp_args["all_data_size"] = len(df)
        
    if layer1_size is not None:
        hp_args["layer1_size"] = layer1_size
    if layer2_size is not None:
        hp_args["layer2_size"] = layer2_size
    if batch_size is not None:
        hp_args["batch_size"] = batch_size
    if learning_rate is not None:
        hp_args["learning_rate"] = learning_rate
    if pretraining_epochs is not None:
        hp_args["pretraining_epochs"] = pretraining_epochs
    if finetuning_epochs is not None:
        hp_args["finetuning_epochs"] = finetuning_epochs

    hp = Hyperparameters(
        input_size=input_size,
        output_size=output_size,
        num_judges=num_judges,
        **hp_args,
    )
    
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {hp.learning_rate}")
    print(f"  Batch size: {hp.batch_size}")
    print(f"  Pretraining epochs: {hp.pretraining_epochs}")
    print(f"  Finetuning epochs: {hp.finetuning_epochs}")
    print(f"  Layer 1 size: {hp.layer1_size}")
    print(f"  Layer 2 size: {hp.layer2_size}")
    
    # Create dataset
    ds_train = pd_utils.make_dataset(df, input_criteria, 'judge_id', output_criteria)
    # Create output directory
    output_dir = Path(model_output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save judge mapping (same for all models)
    Path(judge_map_path).parent.mkdir(exist_ok=True, parents=True)
    with open(judge_map_path, "w") as fh:
        json.dump(judge_id_map, fh, indent=2)
    print(f"Judge map saved to {judge_map_path}")
    
    # ==========================================
    # PRETRAIN ONCE for all questions
    # ==========================================
    print(f"\n{'='*60}")
    print(f"Pretraining (shared for all questions)")
    print(f"{'='*60}")
    
    pretrained_model = PersonalizedCalibrationNetwork(hp)
    optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=hp.learning_rate)
    pretrain_loop(pretrained_model, ds_train, optimizer, hp)
    
    # Save pretrained weights to reuse
    pretrained_state = pretrained_model.state_dict()
    pretrained_path = output_dir / "pretrained_model.pt"
    torch.save(pretrained_state, pretrained_path)
    print(f"Pretrained model saved to {pretrained_path}")
    
    # ==========================================
    # FINETUNE separately for each question
    # ==========================================
    for q_idx in range(num_questions):
        print(f"\n{'='*60}")
        print(f"Finetuning model for Q{q_idx}")
        print(f"{'='*60}")
        
        # Initialize model and load pretrained weights
        model = PersonalizedCalibrationNetwork(hp)
        model.load_state_dict(pretrained_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
        
        # Finetuning on specific question
        print(f"Finetuning for Q{q_idx}...")
        # Update hyperparameter to finetune on this specific question
        hp_q = Hyperparameters(
            input_size=input_size,
            output_size=output_size,
            num_judges=num_judges,
            finetune_output=q_idx,  # Set to current question index
            **hp_args,
        )
        finetune_loop(model, ds_train, optimizer, hp_q)
        
        # Evaluation on the finetuned question
        print(f"Evaluating Q{q_idx}...")
        train_loss = model.loss(ds_train.X, ds_train.A, ds_train.Y, I=[q_idx]).detach().numpy()
        print(f"Train Loss (FT) for Q{q_idx}: {train_loss}")
        
        yhat = model.decode(ds_train.X, ds_train.A, I=[q_idx])[:, 0].detach().numpy()
        y = ds_train.Y[:, q_idx].detach().numpy()
        
        p_corr, _ = stats.pearsonr(y, yhat)
        s_corr, _ = stats.spearmanr(y, yhat)
        t_corr, _ = stats.kendalltau(y, yhat)
        
        print(f"Train pearsonr for Q{q_idx}: {p_corr}")
        print(f"Train spearmanr for Q{q_idx}: {s_corr}")
        print(f"Train kendallt for Q{q_idx}: {t_corr}")
        
        # Save finetuned model for this question
        model_path = output_dir / f"model_Q{q_idx}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model for Q{q_idx} saved to {model_path}")
    
    print(f"\n{'='*60}")
    print(f"All {num_questions} models saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train personalized calibration models')
    
    # Required arguments
    parser.add_argument('--train-json', type=str, required=True,
                        help='Path to training JSON file')
    parser.add_argument('--model-output-dir', type=str, required=True,
                        help='Directory to save trained models')
    parser.add_argument('--judge-map', type=str, required=True,
                        help='Path to save judge ID mapping JSON')
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate for training (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--pretraining-epochs', type=int, default=50,
                        help='Number of pretraining epochs (default: 50)')
    parser.add_argument('--finetuning-epochs', type=int, default=50,
                        help='Number of finetuning epochs per question (default: 50)')
    
    # Model architecture
    parser.add_argument('--layer1-size', type=int, default=100,
                        help='Size of first hidden layer (default: 100)')
    parser.add_argument('--layer2-size', type=int, default=100,
                        help='Size of second hidden layer (default: 100)')
    
    # Other parameters
    parser.add_argument('--num-questions', type=int, default=9,
                        help='Number of questions (default: 9)')
    parser.add_argument('--num-answers', type=int, default=4,
                        help='Number of answer choices per question (default: 4)')
    parser.add_argument('--random-seed', type=int, default=43,
                        help='Random seed for reproducibility (default: 43)')
    parser.add_argument('--all-data-size', type=int, default=None,
                        help='Override dataset size for training (default: None, uses actual size)')
    
    args = parser.parse_args()
    
    main(
        train_json_path=args.train_json,
        model_output_dir=args.model_output_dir,
        judge_map_path=args.judge_map,
        num_questions=args.num_questions,
        num_answers=args.num_answers,
        random_seed=args.random_seed,
        all_data_size=args.all_data_size,
        layer1_size=args.layer1_size,
        layer2_size=args.layer2_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pretraining_epochs=args.pretraining_epochs,
        finetuning_epochs=args.finetuning_epochs,
    )

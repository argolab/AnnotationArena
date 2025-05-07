import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from variables import *
from typing import List, Optional
import sys
sys.path.insert(0, "..")
from joint_dataset import JointDataset
from Imputer import Imputer

def train():
    # Load datasets (modify paths as needed)
    dataset = JointDataset("../gaussian_train_200.json")
    eval_dataset = JointDataset("../gaussian_dev.json")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize the Imputer model
    model = Imputer(
        total_embedding_dimension=30,
        num_heads=6,
        num_layers=5,
        ff_dim=1024,
        dropout=0.1
    )

    model.load_state_dict(torch.load("imputer_model_20.pth"))
    
    num_classes = 5  # Assuming 5 classes as in the original code
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set KL divergence loss for all variables
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    
    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training parameters
    log_interval = 100
    max_epochs = 160
    max_no_improvement = 10

    
    # Training tracking
    prev_dev_loss = float('inf')
    best_dev_loss = float('inf')
    no_improvement_count = 0
    epoch = 0
    
    while True:
        epoch += 1
        model.train()
        running_loss = 0.0
        with torch.autograd.set_detect_anomaly(True):
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
                known_questions = batch[0]
                inputs = batch[1]
                labels = batch[2]
                questions = batch[4]  # Getting the question indices
                
                # Move tensors to device
                inputs, labels, questions = inputs.to(device), labels.to(device), questions.to(device)
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass through the model
                outputs = model(inputs, questions)
                
                # Calculate loss using KL divergence
                loss = torch.tensor(0.0, device=device)
                batch_size = inputs.shape[0]
                var_num = inputs.shape[1]
                
                for i in range(batch_size):
                    for j in range(var_num):
                        if inputs[i, j, 0] == 0:  # Skip observed variables
                            continue
                        
                        # Get variable and dimension
                        var_name = model.question_num_to_question_name[questions[i, j].item()]
                        var = model.variables[var_name]
                        var_dim = var.param_dim()
                        
                        # Get model output for this variable
                        output_logits = outputs[i, j, -var_dim:]
                        log_probs = F.log_softmax(output_logits, dim=-1)
                        
                        # Get target distribution
                        target_probs = labels[i, j, :]
                        
                        # Calculate KL divergence loss
                        var_loss = kl_criterion(log_probs.unsqueeze(0), target_probs.unsqueeze(0))
                        loss += var_loss
                
                # Normalize loss by number of variables
                loss = loss / (batch_size * var_num)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Batch {batch_idx+1}/{len(dataloader)}, KL Divergence: {loss.item():.4f}")
        
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} - Training KL Divergence: {avg_loss:.4f}")
        
        # Evaluate on dev set
        dev_loss = evaluate(model, eval_dataset, kl_criterion, device)
        print(f"Epoch {epoch} - Dev KL Divergence: {dev_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(dev_loss)
        
        # Save the best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), "imputer_model_best.pth")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Early stopping
        if no_improvement_count >= max_no_improvement or epoch >= 20:
            print(f"Stopping training after {epoch} epochs. No improvement for {no_improvement_count} epochs.")
            break
    
    # Save final model
    torch.save(model.state_dict(), "imputer_model_final.pth")
    print("Training complete. Model saved.")

def evaluate(model, dataset, criterion, device):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating on dev"):
            known_questions, inputs, labels, _, questions = batch
            inputs, labels, questions = inputs.to(device), labels.to(device), questions.to(device)
            
            # Forward pass
            outputs = model(inputs, questions)
            
            # Calculate loss
            loss = torch.tensor(0.0, device=device)
            batch_size = inputs.shape[0]
            var_num = inputs.shape[1]
            
            for i in range(batch_size):
                for j in range(var_num):
                    if inputs[i, j, 0] == 0:  # Skip observed variables
                        continue
                    
                    # Get variable and dimension
                    var_name = model.question_num_to_question_name[questions[i, j].item()]
                    var = model.variables[var_name]
                    var_dim = var.param_dim()
                    
                    # Get model output for this variable
                    output_logits = outputs[i, j, -var_dim:]
                    log_probs = F.log_softmax(output_logits, dim=-1)
                    
                    # Get target distribution
                    target_probs = labels[i, j, :]
                    
                    # Calculate KL divergence loss
                    var_loss = criterion(log_probs.unsqueeze(0), target_probs.unsqueeze(0))
                    loss += var_loss
                    
                    count += 1
            
            total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / count if count > 0 else 0
    
    print(f"Evaluation - Avg KL Divergence: {avg_loss:.4f}")
    
    return avg_loss

if __name__ == "__main__":
    train()
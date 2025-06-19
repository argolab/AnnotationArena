@echo off
setlocal enabledelayedexpansion


python src/post_training.py --model_path outputs/models/enhanced_gradient_voi_all_questions.pth --results_path outputs/results_enhanced_hanna/enhanced_gradient_voi_all_questions_new_loss_random_mask.json --dataset hanna --epochs 20 --batch_size 8 --lr 1e-4 --plot_trends --use_embedding --training_type random_mask




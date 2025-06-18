@echo off
setlocal enabledelayedexpansion

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment gradient_voi_q0_human --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options random_mask

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment gradient_voi_q0_both --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options random_mask

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment gradient_voi_q0_all_questions --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options random_mask

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment variable_gradient_comparison --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options random_mask

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment gradient_voi_q0_human --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options old

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment gradient_voi_q0_both --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options old

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment gradient_voi_q0_all_questions --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options old

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment variable_gradient_comparison --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options old

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment gradient_voi_q0_human --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options bd

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment gradient_voi_q0_both --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options bd

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment gradient_voi_q0_all_questions --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options bd

python src/activeLearner.py --examples_per_cycle 50 --features_per_example 5 --experiment variable_gradient_comparison --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 20 --cold_start True --use_embedding True --epochs_per_cycle 10 --training_options bd


@echo off
setlocal enabledelayedexpansion

REM List of experiments
set experiments=gradient_fast_voi_cold_start gradient_voi_cold_start gradient_sequential_cold_start gradient_random_cold_start

REM Loop through each experiment and run the Python script
for %%e in (%experiments%) do (
    echo Running experiment: %%e
    python src/activeLearner.py --examples_per_cycle 60 --features_per_example 3 --experiment %%e --loss_type l2 --resample_validation --dataset hanna --runner haojun --cycles 10 --cold_start True --batch_size 8
)

echo All experiments completed.
pause
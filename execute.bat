@echo off
setlocal enabledelayedexpansion

REM List of experiments
set experiments=gradient_fast_voi gradient_fast_voi_top_only

REM Loop through each experiment and run the Python script
for %%e in (%experiments%) do (
    echo Running experiment: %%e
    python src/activeLearner.py --examples_per_cycle 30 --features_per_example 3 --experiment %%e --loss_type l2 --resample_validation --dataset llm_rubric --runner haojun --run_until_exhausted
)

echo All experiments completed.
pause
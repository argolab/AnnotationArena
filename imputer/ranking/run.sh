#!/bin/bash

#SBATCH -A jeisner1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --job-name=stan_inference
#SBATCH --output=inferencce%j.txt
#SBATCH --cpus-per-task=64

#conda activate ui-tars
#export CXX=$(which g++)
export PYTHONPATH=.
bash scripts/stan/run_inference.sh 

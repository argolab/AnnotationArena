#!/bin/bash

#SBATCH -A jeisner1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --job-name=datagen
#SBATCH --output=datagen%j.txt
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

#conda activate ui-tars
#export CXX=$(which g++)
export PYTHONPATH=.

#bash scripts/stan/generate_data_itemtest.sh
bash scripts/stan/generate_data_annotatortest.sh
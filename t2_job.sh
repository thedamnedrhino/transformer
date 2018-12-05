#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mem=4000
#SBATCH --time=0-00:11
source startup.sh
python t2.py -o the_model -l 100


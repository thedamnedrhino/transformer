#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mem=4000
#SBATCH --time=0-03:11
source startup.sh
python t2.py -o the_model -s 10000


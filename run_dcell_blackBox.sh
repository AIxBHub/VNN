#!/bin/bash
#
#SBATCH -n 1                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --job-name=NxN_costanzo09_10k
#SBATCH --output=NxN_costanzo09/NxN_costanzo09_10k
#SBATCH --time=146:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dariusmb@ad.unc.edu
#
#SBATCH --mem-per-cpu=500000

~/bin/miniconda2/envs/AIxB/bin/python VNN/dcell_blackBox_NeuralNet.py --inputFiles input_files_costanzo09 --filename NxN_lowestPval_weightedPval_alleleListFiltered_100percent.csv --directory NxN_costanzo09 --epochs 300 --layers 10 --percent 100 --label Genetic_interaction_score --batch 10000

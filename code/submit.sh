#!/bin/bash
#SBATCH --job-name=ss_pd_eta0.1
#SBATCH --output=slurm_output/results_%j.log
#SBATCH --error=slurm_output/errors_%j.log
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

julia --project=. -p $SLURM_NTASKS ss_photodet.jl

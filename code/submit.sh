#!/bin/bash
#SBATCH --job-name=pd_p_eta0.4_alpha1.0
#SBATCH --output=slurm_output/results_%j.log
#SBATCH --error=slurm_output/errors_%j.log
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

julia --project=. -p $SLURM_NTASKS photodet.jl

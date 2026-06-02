#!/bin/bash
#SBATCH --job-name=pd_p_1.0_0.4
#SBATCH --output=slurm_output/results_%j.log
#SBATCH --error=slurm_output/errors_%j.log
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G

julia --project=. -p $SLURM_NTASKS photodet.jl p 1.0 0.4

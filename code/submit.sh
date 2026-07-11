#!/bin/bash
#SBATCH --job-name=hod0_p_eta1.0_alpha0.4
#SBATCH --output=slurm_output/results_%j.log
#SBATCH --error=slurm_output/errors_%j.log
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

julia --project=. -p $SLURM_NTASKS data_analysis.jl hod0

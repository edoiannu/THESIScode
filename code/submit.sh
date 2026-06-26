#!/bin/bash
#SBATCH --job-name=sme_solve
#SBATCH --output=slurm_output/results_%j.log
#SBATCH --error=slurm_output/errors_%j.log
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

julia --project=. -p $SLURM_NTASKS photodet.jl

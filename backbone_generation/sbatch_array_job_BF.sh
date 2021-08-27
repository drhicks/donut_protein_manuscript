#!/bin/bash 
#SBATCH -p backfill
#SBATCH -n 1 
#SBATCH --mem=4g
#SBATCH -o bf.log
sed -n ${SLURM_ARRAY_TASK_ID}p jobs.file | bash

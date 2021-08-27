sbatch -a 1-$(cat jobs.file | wc -l) sbatch_array_job_BF.sh

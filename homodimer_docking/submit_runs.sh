for i in $(find . -type f -name "*.run") ; do sbatch $i ; done

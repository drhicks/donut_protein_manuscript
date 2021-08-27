input_full="$1"
input=`echo ${input_full##*/} |cut -d'_' -f1` 
sym=${2}

if [[ ! -e output/ ]]; then mkdir output/; fi
if [[ ! -e output/${input}/ ]]; then mkdir output/${input}/; fi
if [[ ! -e output/${input}/C${sym}/ ]]; then mkdir output/${input}/C${sym}/; fi

OMP_NUM_THREADS=1 \
/home/sheffler/rosetta/scheme_hier/source/cmake/build_omp/sicdock \
	@dock.flags \
	-s ${input_full} \
	-o output/${input}/C${sym}/ > output/${input}/C${sym}/${input}_${sym}.log

rm motif*

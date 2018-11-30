#!/bin/bash
# Launch HPL on Megware cluster

export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

# HT off
#export OMP_NUM_THREADS=36

# HT on
export OMP_NUM_THREADS=72

# Launch Linpack
/cluster/intel/compilers_and_libraries_2018.1.163/linux/mkl/benchmarks/linpack/xlinpack_xeon64 < ${PWD}/linpack_input

#!/bin/bash -x
# Launch IMB Ping Pong on Megware cluster

logDir=~/Results_PS_OPA/

if [ ! -d ${logDir} ]; then
	mkdir -p ${logDir}
fi

module load mpi/intelmpi/2018.1
pushd /cluster/intel/impi/2018.1.163/intel64/bin

# Check the node mapping output with -print-rank map (1 MPI/node)

# PS MPI
export LD_LIBRARY_PATH=/opt/parastation/mpi/lib:/opt/parastation/lib64
mpiexec -u 144 -H ibp05,ibp06 -ppn=1 -np 2 ./IMB-MPI1 pingpong -iter 10000,off -msglog 25 -time 30000000 > ${logDir}/log_$$ 2>&1

# Intel MPI
# mpirun -np 2 -ppn 1 -host ibp05,ibp06 -print-rank-map hostname
# Those are used for IMPI_OFA_CACHE results
#export I_MPI_FABRICS=shm:ofa
#export I_MPI_OFA_TRANSLATION_CACHE=1
#mpirun -np 2 -ppn 1 -host ibp05,ibp06 -print-rank-map ./IMB-MPI1 pingpong -iter 10000,off -msglog 25 -time 30000000 > ~/Results_IMPI_OFA_CACHE_IB/log_$$ 2>&1

# openMPI
#mpirun -np 2 --map-by ppr:1:node -host ibp05,ibp06 -print-rank-map ./IMB-MPI1 pingpong -iter 10000,off -msglog 8 

popd

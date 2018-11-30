#!/bin/bash


#SBATCH --job-name=stream
#SBATCH --exclusive
#SBATCH --time=2:00:00

#SBATCH --output=stdout
#SBATCH --error=stderr
#SBATCH --export=all

#SBATCH --ntasks=1
##SBATCH --nodes=1  #new @LM
#SBATCH --cpus-per-task=24

#SBATCH --partition=sdv

module purge
module load impi/5.1

export OMP_NUM_THREADS=24
export KMP_AFFINITY=scatter

srun -n 1 ./runme_xeon64

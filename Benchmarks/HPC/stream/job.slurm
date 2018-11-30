#!/bin/bash

#SBATCH --job-name=stream_benchmarks
#SBATCH --exclusive
#SBATCH --time=2:00:00

#SBATCH --output=stdout
#SBATCH --error=stderr
#SBATCH --export=all

##SBATCH --ntasks=1
##SBATCH --nodes=1  #new @LM
#SBATCH --cpus-per-task=24

#SBATCH --partition=sdv

export OMP_NUM_THREADS=24
export KMP_AFFINITY=scatter

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

numactl -l ./stream

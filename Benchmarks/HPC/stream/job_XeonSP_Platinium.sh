#!/bin/bash

export OMP_NUM_THREADS=112
export KMP_AFFINITY=scatter

numactl -l ./stream

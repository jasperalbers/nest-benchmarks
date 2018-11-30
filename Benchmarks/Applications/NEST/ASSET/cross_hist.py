import numpy as np
import quantities as pq

import neo
import elephant.statistics as estats
import elephant.spike_train_generation as stg
import elephant.spike_train_correlation as stc
import elephant.conversion as conv

#try:
#    from mpi4py import MPI
#    mpi_accelerated = True
#except:
mpi_accelerated = False

# ===========================================================================
# Parameters definition
# ===========================================================================
# Data parameters
N = 1000                 # Number of spike trains
rate = 15 * pq.Hz       # Firing rate
T = 1 * pq.s            # Length of data 
binsize = 5 * pq.ms

# =======================================================================
# Data generation
# =======================================================================

# Generate the data
sts = []
for i in range(N):
    np.random.seed(i)
    sts.append(conv.BinnedSpikeTrain(stg.homogeneous_poisson_process(rate, t_stop=T), binsize=binsize)) 

for i in sts:
    for j in sts:
        stc.cch(i, j, window='full', border_correction=False, binary=False,
                kernel=None, method='speed', cross_corr_coef=False)

print('Done')

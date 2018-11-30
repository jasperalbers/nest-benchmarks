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
N = 500                 # Number of spike trains
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
    sts.append(stg.homogeneous_poisson_process(rate, t_stop=T)) 

cc = stc.corrcoef(conv.BinnedSpikeTrain(sts, binsize=binsize))
print(cc)

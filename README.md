# NEST benchmarks

To run MAM:
download nested_dict: pip install nested_dict --user
download dicthash: pip install dicthash --user

To run 4x4: (must use Py2)
PIZ-DAINT: module load cray-python/2.7.15.1
           module load PyExtensions/2.7.15.1-CrayGNU-18.08
           module load h5py/2.8.0-CrayGNU-18.08-python2-parallel

download NeuroTools: pip install NeuroTools
download h5py

STALLO: download mpi4py

run python setup.py install --user

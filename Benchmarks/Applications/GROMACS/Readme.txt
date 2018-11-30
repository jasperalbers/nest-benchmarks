Mirco benchmark with GROMACS 2018 on one node.

Plese run jube in the GROMACS directory of the Benchmark repo due to the replative paths used in the implementation.

compile:

jube run gmx-2018-compile-sdv.xml

run benchmark with different steps varing number MPI procs. and number of OpenMP threads per MPI proc:

jube run gmx-2018-magainin-sdv-1N.xml


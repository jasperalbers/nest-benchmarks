# Task 1.1 Benchmarks for DEEP-EST

### Authors: Hans Ekkehard Plesser <hans.ekkehard.plesser@nmbu.no>, Susanne Kunkel <susanne.kunkel@nmbu.no>

# TABLE of CONTENTS

- NEST Benchmarks
- Arbor Benchmarks
- Elephant Benchmarks

# NEST Benchmarks for DEEP-EST

This directory contains benchmarks for the NEST simulator. For
background information, see DEEP-EST deliverable D1.2, Sec 2.

## HPC Benchmark (Micro Benchmark)

The HPC Benchmark is a slightly modified version of a benchmark used in NEST development since 2017. It requires only NEST with a minimum of external libraries. In particular, it does not require Python or GSL.

The benchmark is provided with a complete set of JUBE scripts for benchmark setup and execution.

The Benchmark supports both the most recent NEST release version (2.14) and the future 5th-generation NEST kernel which has a different strategy for connectivity representation and MPI communication to permit scaling beyond 100.000 MPI ranks and 1.000.000 virtual processes. For details on the 5th generation kernel see Jordan J, Ippen T, Helias M, Kitayama I, Sato M, Igarashi J, Diesmann M and Kunkel S (2018) Extremely Scalable Spiking Neuronal Network Simulation Code: From Laptops to Exascale Computers. Front. Neuroinform. 12:2. doi: 10.3389/fninf.2018.00002.


### Prerequisites

#### Directory Structure

The benchmark suite is set up such that benchmark configuration files (JUBE XML files) are kept in the DEEP-EST Gitlab Repository `Benchmark` under `Applications/NEST`, while all other files (simulator source code, build and installation directories, JUBE output) is kept in a separate directory `BenchWork`. Directories can be adjusted in `jube_config/dir_config.xml`.

#### Software installation

To run the full set of benchmarks, the following software is required

- jemalloc (strongly recommended)
- NEST 2.14
- NEST 5G (optional)

In order for the JUBE build scripts to work, source code needs to be manually installed as follows:

##### Jemalloc

```
cd <PATH>/BenchWork
mkdir jemalloc
cd jemalloc
wget https://github.com/jemalloc/jemalloc/releases/download/5.0.1/jemalloc-5.0.1.tar.bz2
tar xvf jemalloc-5.0.1.tar.bz2
```

##### NEST 2.14

```
cd <PATH>/BenchWork
mkdir NEST-2.14
cd NEST-2.14
git clone ssh://git@gitlab.version.fz-juelich.de:10022/DEEP-EST/NEST-2.14.git src
```

Note: clone into `src`

##### NEST 5G

```
cd <PATH>/BenchWork
mkdir NEST-5G
cd NEST-5G
git clone ssh://git@gitlab.version.fz-juelich.de:10022/DEEP-EST/NEST-5G.git src
```

##### Building the software

Once the source code is in place, build and install via JUBE:

Adjust the `bench_home` and `bench_work` folders defined in `Benchmarks/Applications/NEST/jube_config/dir_config.xml`, then run

```
jube run Benchmarks/Applications/NEST/jube_build/build_jemalloc_sdv.xml
jube run Benchmarks/Applications/NEST/jube_build/build_nest_2_14_sdv_strict.xml 
jube run Benchmarks/Applications/NEST/jube_build/build_nest_5g_sdv_strict.xml 
```

Software will be installed in
- `BenchWork/jemalloc-sdv/install`
- `BenchWork/NEST-2.14-sdv-base_O3_strict/install`
- `BenchWork/NEST-5G-sdv-base_O3_strict/install`

##### Compilation configuration

- Compiler flags are defined in `jube_config/packages.xml`
- There is one overall parameterset per package (NEST 2.14, NEST 5G, Arbor, Jemalloc), and then further down in the file parametersets for different compiler flag configurations.
- Create a new configuration by copying an existing parameterset and changing `nest_cmake_defs`
- **NOTE**: You must give the new configuration a new `nest_config` name, otherwise build and install directories will clash (building removes existing directories first).
- Modify `nest_cmake_defs` to change the way NEST is built
- Then create a new XML file in `jube_build` by copying an existing one and replacing `nest_base_O3` in
```
<use from="packages.xml">nest_2_14,nest_base_O3</use>
```
by the name of your configuration.

##### IEEE 754 Conformance required

- NEST must be built with compiler options ensuring strict compliance with IEEE 754 arithmetic.
- For the Intel compiler, this requires `-fp-model strict`.
- Build scripts without `strict` in their name will build NEST without IEEE 754 compliance; you can use that code for exploring potential gains. If there sshould be signficant gains, please consult with us about how to proceed!

##### Building with Python

A script to build NEST with Python interface is provided, but it currently still fails when linking the PyNEST component.

#### Testsuite

You should run the testsuite

Before you can run the NEST testsuite, you must create file `.nestrc` in your home directory. The file must contain at least the following lines:

```
/mpirun
[/integertype /stringtype /stringtype]
[/numproc     /executable /scriptfile]
{
 () [
  (srun -n ) numproc cvs ( ) executable ( ) scriptfile
 ] {join} Fold
} Function def
```

If NEST has created `.nestrc` for you, you most likely need to replace `(mpirun -np )`
by `(srun -n )`; note the space after `-n`.

You can then run the NEST testsuite with
```
jube run Benchmarks/Applications/NEST/jube_build/installcheck_nest_2_14_sdv_strict.xml 
jube run Benchmarks/Applications/NEST/jube_build/installcheck_nest_5g_sdv_strict.xml 
```
followed by
```
jube report -a ...
```
The report should show no failed tests, e.g.,
```
NEST Installcheck:
nest_version | system_name |     nest_config | n_tests | n_passed | n_skipped | n_failed
-------------+-------------+-----------------+---------+----------+-----------+---------
   NEST-2.14 |         sdv |  base_O3_strict |     358 |      337 |        21 |        0
```
   

### The benchmark script

- The benchmark script is `hpc_scripts/hpc_benchmark.sli`.
- It is written in an extended version of Postscript.
- See the detailed comment at the beginning of the script.
- The benchmark can be run with any combination of number of MPI ranks and threads per rank.
- The product of the number of MPI ranks and threads per rank is the number of *virtual processes*.
- Running the benchmark with the same number of virtual processes split differently between MPI ranks and threads shall yield identical values for `N_spks_sum`.


### Running the benchmark

- JUBE benchmark configurations are in `jube_hpc`
- All benchmark configuration files have names beginning with `hpc_benchmark_`
- The name of the configuration file indicates
  - the version of the NEST kernel
  - the system and compiler options (-fp-model strict)
  - the scale of the system (s20: scale 20)
  - a flag indicating the delay policy (see below)
  - possibly a flag indicating synaptic plasticity
  - the parallel configuration
    - _mpi: 24 MPI ranks, 1 thread each
    - _trd: 1 MPI rank, 24 threads
	- _mix: 6 MPI ranks, 4 threads each
- Only some combinations are provided by default to focus benchmarking efforts.

#### Delay policy

Note: Jube scripts for all variants except "fd_plastic" are in
directory `jube_hpc_experimental`.

##### fd_plastic: fixed delay with plastic synapses

This is closest to the classic hpc_benchmark. In this case, all delays are exactly 1.5 ms. This means
- ring buffers are exactly 30 elements (doubles) each
- with 225.000 neurons on 24 processes, each process has 9375 neurons; each has two spike input ring buffers, so the total ring buffer memory per process is 9375 x 2 x 30 x 8B = 4.3 MB
- MPI processes/threads synchronize every 15 time steps
- 64% of synapses are plastic, requiring considerably more effort during simulation

##### fd: fixed delay 

As fd_plastic, but without plastic synapses.

##### sd: short delays

- delays are uniformly distributed between 0.1 and 1.5 ms
- ring buffer size is 16 elements each
- total ring buffer size per process is 2.3 MB
- MPI processes and threads synchronize after every time step

##### ld: long delays

- this regime is closest to the one for the multiarea benchmark
- delays are uniformly distributed between 0.1 and 50. ms
- ring buffer size is 501 elements each
- total ring buffer size per process is 269 MB
- MPI processes and threads synchronize after every time step


#### Runtime and memory requirements

- At scale 20, the benchmark (2.14)
  - requires about 100 GB memory
  - takes between 250s (24 mpi ranks) and 360s (24 threads)

- At scale 20, the network has
    - 225.000 neurons
    - 2.5 billion connections
	- Expected rate between 6 and 8 spikes/s (depends on delay regime)
	- Expected N_spks_sum around 1.6 million

#### Benchmark output

- For information about output generated by the benchmark script, see the script
- JUBE analysis and report will provide the following table
```
NUMBER_OF_NODES | TASKS_PER_NODE | THREADS_PER_TASK | SCALE | PLASTIC | T_nrns | T_conns_min | T_conns_max | T_ini_min | T_ini_max | T_equ | T_sim | VSize_sum | N_spks_sum | Rate_sum | N_nrns | N_conns_sum | d_min | d_max
----------------+----------------+------------------+-------+---------+--------+-------------+-------------+-----------+-----------+-------+-------+-----------+------------+----------+--------+-------------+-------+------
              1 |              6 |                4 |    20 |    true |   0.26 |        68.2 |       70.85 |      1.28 |      3.93 | 10.95 | 172.7 | 103783128 |    1582628 |    6.952 | 225000 |  2531476000 |   1.5 |   1.5
```
- `T_nrns`: time to instantiate neurons
- `T_conns`: time to instantiate connections
- `T_ini`: time to simulate first time step
- `T_equ`: time to simulate first 100 ms (equilibration phase)
- `T_sim`: time to simulate next 1000 ms
- `VSize`: virtual memory size at end of simulation, sum across ranks
- `N_spks`: number of spikes fired, sum across ranks
- `Rate`: average firing rate
- `N_nrns`: number of neurons in network, all ranks
- `N_conns`: number of connections, sum across ranks
- `d_min`: minimal delay
- `d_max`: maximal delay

##### Notes
- Due to synchronization points in the simulation, the following should hold (first rank to complete connections needs to wait longest to simulate initial step):
```
T_conns_min + T_ini_max == T_conns_max + T_ini_min
```
- Average rate across entire network is obtained by summing across ranks since each rank only knows the spikes it fires, but normalizes with full neuron number
- Rate will be different from N_spks divided by neuron number and time, because N_spks includes all spikes during entire simulation, while Rate only includes spikes from NUM_REC neurons during simtime.

#### Adjusting the benchmark

To modify the benchmark setup, copy one of the files in `jube_hpc` and change one or more of the following parameters:
```
<parameterset name="scale_check">
  <parameter name="SCALE" type="int">20</parameter>
  <parameter name="SIMTIME" type="float">1000</parameter>
  <parameter name="NUMBER_OF_NODES" type="int">1</parameter>
  <parameter name="TASKS_PER_NODE" type="int">24</parameter>
  <parameter name="THREADS_PER_TASK" type="int">1</parameter>
  <parameter name="NUM_REC" type="int">1000</parameter>
  <parameter name="PLASTIC" type="string">false</parameter>
  <parameter name="D_MIN" type="float">0.1</parameter>
  <parameter name="D_MAX" type="float">50.0</parameter>
</parameterset>
```
- `SCALE` determines network size; can be reduced to 1 for testing, but >= 8 gives most relevant dynamics
- `SIMTIME` can be reduced to zero to reduce runtime, although results then may be dominated by transient dynamics
- `NUMBER_OF_NODES` should be 1 for micro benchmark, but can be as large as desired
- `TASKS_PER_NODE` is number of MPI ranks per node; experience indicates that oversubscribing HW cores yields small gains
- `THREADS_PER_TASK` is number of threads per MPI rank
- `NUM_REC` number of neurons to record from; can be set to 0, does not affect benchmark
- `PLASTIC` is `true` or `false`; if true, 64% of synapses (E->E) will be plastic, i.e., change over time
- `D_MIN` is minimal delay in ms, multiple of 0.1
- `D_MAX` is maximal delay in ms, must be `>= D_MIN` and multiple of 0.1




## Simplified multi-area model Benchmark

This benchmark is very similar to the HPC Benchmark with the same
prerequisites. This section therefore describes only those aspects
that differ from the HPC Benchmark.

### The benchmark script

- The benchmark script is `hpc_scripts/hpc_mam_benchmark.sli`.


### Running the benchmark

- JUBE benchmark configurations are in `jube_hpc`
- All benchmark configuration files have names beginning with `hpc_mam_benchmark_`
- The name of the configuration file indicates
  - the version of the NEST kernel
  - the system and compiler options (-fp-model strict)
  - the scale of the system (s20: scale 20)
  - the parallel configuration
    - _mpi: 24 MPI ranks, 1 thread each
    - _trd: 1 MPI rank, 24 threads
    - _mix: 6 MPI ranks, 4 threads each
- Only some combinations are provided by default to focus benchmarking efforts.

#### Delay policy

Delays are randomized uniformly between 0.1 ms and 50.0 ms.

#### Runtime and memory requirements

- At scale 20, the benchmark (2.14)
  - requires about 40 GB memory
  - takes between 130s (24 mpi ranks) and 190s (24 threads)

- At scale 20, the network has
    - 225.000 neurons
    - 1.3 billion connections
	- Expected rate around 13 spikes/s 
	- Expected N_spks_sum around 2.9 million

##### Full-scale benchmark on 8 SDV nodes

We also provide configurations for full scale benchmarks, these are marked `s367` (scale 367). In this case,

- the network
    - has approximately the same number of neurons (4.13 million)
    - and connections (23.2 billion) as the multi-area model
    - fires 54 million spikes (rate 13 sp/s)
- runs on
    - 8 SDV nodes
    - 6 MPI ranks per node, total 48 MPI ranks
    - 4 threads per rank
- requires (NEST 2.14)
    - 760 GB total memory (95 GB per node)
    - 390s total run time (T_sim 271 s)


#### Benchmark output

- For information about output generated by the benchmark script, see the script
- JUBE analysis and report will provide the same table as for the HPC Benchmark


#### Adjusting the benchmark

See HPC Benchmark, except that `PLASTIC`, `D_MIN` and `D_MAX` cannot
be set.


## Potjans-Diesmann Model

to be added

## Multi-area Model

to be added

# Arbor Benchmarks

### Arbor expert: Alex Peyser <a.peyser@fz-juelich.de>

## About the benchmark

Arbor is under heavy development. We currently provide a single benchmark which runs as a stand-alone application independent of NEST. We will extend the set of Arbor benchmarks in coming weeks.


### Prerequisites

#### Directory Structure

The overall directory structure is the same as for the NEST benchmarks.

#### Software installation

In order for the JUBE build scripts to work, source code for Arbor needs to be manually installed as follows:

```
cd <PATH>/BenchWork
mkdir Arbor
cd Arbor
git clone https://github.com/eth-cscs/arbor.git src
```

##### Building the software

Build the software using

```
jube run Benchmarks/Applications/NEST/jube_build/build_arbor_sdv_plain.xml 
jube run Benchmarks/Applications/NEST/jube_build/build_arbor_sdv_avx2.xml 
jube run Benchmarks/Applications/NEST/jube_build/build_arbor_sdv_avx512.xml 
```

Software will be installed in
- `BenchWork/Arbor-sdv-plain/install`
- `BenchWork/Arbor-sdv-avx2/install`
- `BenchWork/Arbor-sdv-avx512/install`


##### Compilation configuration

- Compiler flags are defined in `jube_config/packages.xml`
- See information on NEST build configuration for more information on the setup
- Arbor also provides optimization for KNL and CUDA, but those have not been tested on DEEP-ER systems
  - To activate CUDA, support, `-DARB_WITH_CUDA=ON` is required
- Arbor can also use Intel TBB instead of C++11 threads
  - This is activated using `-DARB_THREADING_MODEL=tbb`
  - This will download and build TBB during the Arbor build
  - This has not been tested in DEEP-ER systems

### Running the benchmark

- Benchmark setups are provided in directory `jube_arbor`
- The benchmarks can be run using the avx2 or avx512 builds
    - benchmark configuration scripts are labeled respectively
    - avx512 configurations have not been not tested, since we have no access to suitable hardware at present

Results are reported as

```
ARBOR THREE SEGMENT AVX2:
NUMBER_OF_NODES | TASKS_PER_NODE | THREADS_PER_TASK | SCALE | NUM_SYNAPSES | NUM_COMPARTMENTS | SIMTIME | T_setup | T_init | T_simulate | n_spikes
----------------+----------------+------------------+-------+--------------+------------------+---------+---------+--------+------------+---------
              1 |              1 |               48 |   2.0 |         2000 |              400 |   200.0 |     0.0 |  9.809 |    414.002 |   202501
```

- The first entries are the same as for NEST benchmarks, but Arbor benchmarks allow fractional SCALE.
- NUM_SYN is the number of synapses per neuron.
- NUM_COMP is the number of compartments per segment. To obtain a total of approximately 1000 compartments per neuron, comparable with relevant cases, this should be around 7 for the pyramidal cell cases and much larger for the three-segment benchmark.
- SIMTIME is the time simulated in milliseconds.
- T_setup is setup time.
- T_init is initialization time.
- T_simulate is simulation time.
- n_spikes is the number of spikes fired during SIMTIME, including the one injected spike.

In addition, Arbor writes a detailed report indicating the time spent on parts of the code to `stdout`, but we do not currently extract this information in our JUBE scripts.


#### Standard three-segment-dendrite benchmark

This benchmark can be run using

```
jube run Benchmarks/Applications/NEST/jube_arbor/benchmark_three_segment_sdv_avx2.xml
jube run Benchmarks/Applications/NEST/jube_arbor/benchmark_three_segment_sdv_avx512.xml
```

#### Pyramidal cell non-spiking benchmark

```
jube run Benchmarks/Applications/NEST/jube_arbor/benchmark_pyr_nonspiking_sdv_avx2.xml  
jube run Benchmarks/Applications/NEST/jube_arbor/benchmark_pyr_nonspiking_sdv_avx512.xml  
```


#### Pyramidal cell spiking benchmark

To be written.


# Elephant Benchmarks

To be written.

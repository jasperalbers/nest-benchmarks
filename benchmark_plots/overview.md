### Disse benchmarkene kjører jeg:

- **Vanlig hpc**: hpc_benchmark_conn_sort_daint_strict.xml
- **hpc med statisk synapse eller forskjellig delay**: hpc_benchmark_conn_sort_daint_strict.xml med endrede parametere
- **hpc split into many Connect calls**: hpc_benchmark_conn_sort_daint_strict_split_new.xml
    - enten med ```<parameter name="NBLOCKS" type="int" mode="python">1000</parameter>```
    - eller med `<parameter name="NBLOCKS" type="int" mode="python">1000*$NUMBER_OF_NODES</parameter>`
- **Population model**: population_conn_sort_daint_strict.xml
- **Multi-Area Model**: multi-area-model_conn_sort_daint_strict.xml
- **4x4 mesocircuit**: 4x4_mesocircuit_conn_sort_daint_strict.xml
- **hpc med forskjellig regel**: hpc_benchmark_conn_sort_daint_strict_rule.xml

For å kjøre hpc med fixed VP, different threads kjører jeg hpc_benchmark_conn_sort_daint_strict.xml med NUMBER_OF_NODES=1 eller 36, og forskjellige tråder.



### NEST Varianter

Jeg kjører NEST **med** Boost (må kopiere ut ting i sort.h) NEST git hash: `51abb7c7`, fra denne branchen: https://github.com/stinebuu/nest-simulator/tree/connection_updade_pluss_boost_sorting Tror begge er inne i master nå, men for å få det helt konsistent burde du kanskje bruke denne?

Du trenger 2 varianter av NEST:

- build_nest_conn_sort_daint_strict.xml for *hpc benchmarkene, population, 4x4*
- build_nest_conn_sort_py_daint_strict.xml for *MAM* (den lager for Py-3).



### For å kjøre MAM:

- download nested_dict: `pip install nested_dict --user`
- download dicthash: `pip install dicthash --user`

Jeg tror jeg bare bruker vanlig python fra system description, så her trenger du ikke gjøre så mye.

Vær obs på at MAM kan være litt sta, den går ikke alltid i gjennom. Noen ganger klager den på manglende mappe, noen ganger klager den på JSON, noen ganger bruker den rett og slett for lang tid.

For 32 noder hender det den trenger 2.5 timer i stedet for 1.5 timer som jeg vanligvis lar dem kjøre på. *hpc_benchmark med fixed_outdegree (out)* trenger også av og til 2.5 timer.

Du må ha installert NEST med python (build_nest_conn_sort_py_daint_strict.xml) for å kjøre MAM.



### For å kjøre 4x4:

For å installere:

> > > > > > > > > >```
> > > > > > > > > >
> > > > > > > > > >```
> > > > > > > > > >
> > > > > > > > > >> > > > > > > > > module load cray-python/2.7.15.1
> > > > > > > > > >> > > > > > > > >
> > > > > > > > > >> > > > > > > > > module load PyExtensions/2.7.15.1-CrayGNU-18.08
> > > > > > > > > >> > > > > > > > >
> > > > > > > > > >> > > > > > > > > module load h5py/2.8.0-CrayGNU-18.08-python2-parallel

pip install NeuroTools

```bash

```

Gå så til mesocirucuit mappen og kjør `run python setup.py install --user`

4x4 bruker et sli script for å kjøre selve benchmarket, men genererer parameteret gjennom en python fil, så derfor må man installere med python, men når man først har gjort det, er det rett frem.



### Benchmark analyse

Notebooken og alle figurene/latex-tabellene ligger i mappen `benchmark_plots`. 
# Building Arbor on Megware

This describes manual compilation attempts, not via JUBE scripts.

### Basic setup
```
ssh megware
ssh frontend

# best to work on compute node, only there is all SW 
srun --exclusive -p CPU_8260 --nodes=1 -t 1:00:00 --pty bash

module load bench/jube-2.2.0
module load comp/cmake/3.11.1
```

See https://arbor.readthedocs.io/en/latest/install.html for
installation instructions.

## Minimal build

- Up-to-date Arbor sources (2ff590e)
- No MPI
- Spack gcc 8.1.0 package

```
module load spack/comp/gcc/8.1.0

mkdir bld_2ff590e_minimal
cd bld_2ff590e_minimal
export CC=gcc
export CXX=g++

cmake ../src -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install -DARB_WITH_MPI=OFF

make -j48
```

Fails with:
```
[ 76%] Linking CXX executable ../../bin/validate
`_ZNSt17_Function_handlerIFbN3arb16cell_member_typeEEZNS0_L9one_probeES1_EUlS1_E_E9_M_invokeERKSt9_Any_dataOS1_' referenced in section `.rodata.cst8' of CMakeFiles/validate.dir/validate_kinetic.cpp.o: defined in discarded section `.text._ZNSt17_Function_handlerIFbN3arb16cell_member_typeEEZNS0_L9one_probeES1_EUlS1_E_E9_M_invokeERKSt9_Any_dataOS1_[_ZN3arb23convergence_test_runnerIfE3runERNS_10simulationEffffRKSt6vectorIfSaIfEE]' of CMakeFiles/validate.dir/validate_kinetic.cpp.o
`_ZNSt17_Function_handlerIFbN3arb16cell_member_typeEEZNS0_L9one_probeES1_EUlS1_E_E9_M_invokeERKSt9_Any_dataOS1_' referenced in section `.rodata.cst8' of CMakeFiles/validate.dir/validate_synapses.cpp.o: defined in discarded section `.text._ZNSt17_Function_handlerIFbN3arb16cell_member_typeEEZNS0_L9one_probeES1_EUlS1_E_E9_M_invokeERKSt9_Any_dataOS1_[_ZN3arb23convergence_test_runnerIiE3runERNS_10simulationEifffRKSt6vectorIfSaIfEE]' of CMakeFiles/validate.dir/validate_synapses.cpp.o
collect2: error: ld returned 1 exit status
```

But all else goes through.

`make install` fails since `validate` cannot be built, but the
following binaries are provided in `bin`:
```
bench
brunel-miniapp
event-gen
lmorpho
miniapp
modcc
ring
unit
unit-local
unit-modcc
```

## Minimal build with vectorization

```
cmake ../src -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install \
      -DARB_WITH_MPI=OFF \
      -DARB_VECTORIZE=ON \
	  -DARB_ARCH=native
```

In this case even `validate` builds.


## Build with Intel compiler

```
module load intel-studio-2018
module load spack/comp/gcc/8.1.0   # for STL

export CC=icc
export CXX=icpc
```


Fails with

```
[ 12%] Building CXX object modcc/CMakeFiles/libmodcc.dir/printer/printerutil.cpp.o
[ 12%] Built target generate_version_hpp
[ 12%] Built target default_catalogue_cpp_target
/cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/gcc-8.1.0-tiy4ahj2zumrmrmgiks4fww5hkxbwmoc/include/c++/8.1.0/bits/stl_tree.h(700): error: identifier "_Node_allocator" is undefined
  	    _GLIBCXX_NOEXCEPT_IF(
  	    ^
          detected during instantiation of "std::_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_Rb_tree_impl<_Key_compare, <unnamed>>::_Rb_tree_impl() [with _Key=tok, _Val=std::pair<const tok, int>, _KeyOfValue=std::_Select1st<std::pair<const tok, int>>, _Compare=std::less<tok>, _Alloc=std::allocator<std::pair<const tok, int>>, _Key_compare=std::less<tok>, <unnamed>=true]" at line 350 of "/home/hplesser/BenchWork/Arbor/src/modcc/lexer.cpp"

/cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/gcc-8.1.0-tiy4ahj2zumrmrmgiks4fww5hkxbwmoc/include/c++/8.1.0/type_traits(921): error: class "std::__is_default_constructible_atom<<error-type>>" has no member class "type"
      : public __is_default_constructible_atom<_Tp>::type
                                                     ^
          detected during:
            instantiation of class "std::__is_default_constructible_safe<_Tp, false> [with _Tp=<error-type>]" at line 927
            instantiation of class "std::is_default_constructible<_Tp> [with _Tp=<error-type>]" at line 144
            instantiation of class "std::__and_<_B1, _B2> [with _B1=std::is_default_constructible<<error-type>>, _B2=std::__is_nt_default_constructible_impl<<error-type>, false>]" at line 995
            instantiation of class "std::is_nothrow_default_constructible<_Tp> [with _Tp=<error-type>]" at line 700 of "/cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/gcc-8.1.0-tiy4ahj2zumrmrmgiks4fww5hkxbwmoc/include/c++/8.1.0/bits/stl_tree.h"
            instantiation of "std::_Rb_tree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::_Rb_tree_impl<_Key_compare, <unnamed>>::_Rb_tree_impl() [with _Key=tok, _Val=std::pair<const tok, int>, _KeyOfValue=std::_Select1st<std::pair<const tok, int>>, _Compare=std::less<tok>, _Alloc=std::allocator<std::pair<const tok, int>>, _Key_compare=std::less<tok>, <unnamed>=true]" at line 350 of "/home/hplesser/BenchWork/Arbor/src/modcc/lexer.cpp"
```


## Building with GCC and MPI

```
module load spack/comp/gcc/8.1.0
module load spack/mpi/openmpi-1.10.7-gcc-7.3.0-cuda-mxm

export CC=`which mpicc`
export CXX=`which mpicxx`

cmake ../src -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install -DARB_WITH_MPI=ON -DARB_VECTORIZE=ON -DARB_ARCH=native
```

fails with

```
[hplesser@ibp27 bld_2ff590e_mpi]$ cmake ../src -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install -DARB_WITH_MPI=ON -DARB_VECTORIZE=ON -DARB_ARCH=native
-- The C compiler identification is GNU 7.3.0
-- The CXX compiler identification is GNU 7.3.0
-- Check for working C compiler: /cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-1.10.7-ki54b7xwboly5glte3dnn2dploggnezc/bin/mpicc
-- Check for working C compiler: /cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-1.10.7-ki54b7xwboly5glte3dnn2dploggnezc/bin/mpicc -- broken
CMake Error at /cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/cmake-3.11.1-tp5nebmorutou4mve7bscq7cajrwupe5/share/cmake-3.11/Modules/CMakeTestCCompiler.cmake:52 (message):
  The C compiler

    "/cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-1.10.7-ki54b7xwboly5glte3dnn2dploggnezc/bin/mpicc"

  is not able to compile a simple test program.

  It fails with the following output:

    Change Dir: /home/hplesser/BenchWork/Arbor/bld_2ff590e_mpi/CMakeFiles/CMakeTmp
    
    Run Build Command:"/bin/gmake" "cmTC_c5b13/fast"
    /bin/gmake -f CMakeFiles/cmTC_c5b13.dir/build.make CMakeFiles/cmTC_c5b13.dir/build
    gmake[1]: Entering directory `/home/hplesser/BenchWork/Arbor/bld_2ff590e_mpi/CMakeFiles/CMakeTmp'
    Building C object CMakeFiles/cmTC_c5b13.dir/testCCompiler.c.o
    /cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-1.10.7-ki54b7xwboly5glte3dnn2dploggnezc/bin/mpicc    -o CMakeFiles/cmTC_c5b13.dir/testCCompiler.c.o   -c /home/hplesser/BenchWork/Arbor/bld_2ff590e_mpi/CMakeFiles/CMakeTmp/testCCompiler.c
    Linking C executable cmTC_c5b13
    /cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/cmake-3.11.1-tp5nebmorutou4mve7bscq7cajrwupe5/bin/cmake -E cmake_link_script CMakeFiles/cmTC_c5b13.dir/link.txt --verbose=1
    /cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-1.10.7-ki54b7xwboly5glte3dnn2dploggnezc/bin/mpicc      CMakeFiles/cmTC_c5b13.dir/testCCompiler.c.o  -o cmTC_c5b13 
    /bin/ld: warning: libmxm.so.2, needed by /cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-1.10.7-ki54b7xwboly5glte3dnn2dploggnezc/lib/libmpi.so, not found (try using -rpath or -rpath-link)
    /cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-1.10.7-ki54b7xwboly5glte3dnn2dploggnezc/lib/libmpi.so: undefined reference to `mxm_ep_get_address'
    /cluster/tools/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-1.10.7-ki54b7xwboly5glte3dnn2dploggnezc/lib/libmpi.so: undefined reference to `mxm_error_string'
```


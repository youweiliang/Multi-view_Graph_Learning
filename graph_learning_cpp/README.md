### Compile the Consistent Graph Learning algorithm in MATLAB to a mex file or C++ application
For folks without MATLAB and for industrial usage, the Consistent Graph Learning algorithm is coded in C++ (or C with slight modification). A MATLAB implementation should be simple by following the algorithm in the paper and maybe faster than a C++ implementation _without_ MKL, since the bottleneck of the Consistent Graph Learning algorithm lies in large matrix multiplication and MATLAB internally uses MKL for matrix multiplication. Therefore, if using C++ implementation, it is highly recommended to install MKL (it's free). 

Both MinGW64 Compiler and Visual Studio Compiler would work (use `mex -setup C++` in MATLAB to choose the compiler). Also, see [installing MinGW64](https://www.mathworks.com/matlabcentral/fileexchange/52848-matlab-support-for-mingw-w64-c-c-compiler).

* If using MinGW64 for compilation, just remove `COMPFLAGS="$COMPFLAGS /openmp" LINKFLAGS='$LINKFLAGS /nodefaultlib:vcomp;vcompd'`
* If using MKL, modify the mkl library location `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl` and `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\compiler\lib\intel64_win` in the commands depending on where the mkl is installed
* If using MKL, comment out the line `#define NOBLAS` in file `use_blas.h`

#### Compilation commands
* NO BLAS and NO MKL  
`mex -v -largeArrayDims COMPFLAGS="$COMPFLAGS /openmp" LINKFLAGS='$LINKFLAGS /nodefaultlib:vcomp;vcompd' consistent_graph_dca.cpp DCA.cpp helper.cpp qp_simplex_fixed2.cpp linear_algebra.cpp`

* With MKL, use openmp for multithreading, static link  
`mex -v -largeArrayDims COMPFLAGS="$COMPFLAGS /openmp" LINKFLAGS='$LINKFLAGS /nodefaultlib:vcomp;vcompd' -I'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\include' consistent_graph_dca.cpp DCA.cpp helper.cpp qp_simplex_fixed2.cpp linear_algebra.cpp -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\lib\intel64_win' -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\compiler\lib\intel64_win' mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib`

* With MKL, use openmp for multithreading, dynamic link  
`mex -v -largeArrayDims COMPFLAGS="$COMPFLAGS /openmp" LINKFLAGS='$LINKFLAGS /nodefaultlib:vcomp;vcompd' -I'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\include' consistent_graph_dca.cpp DCA.cpp helper.cpp qp_simplex_fixed2.cpp linear_algebra.cpp -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\lib\intel64_win' -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\compiler\lib\intel64_win' mkl_intel_lp64_dll.lib mkl_intel_thread_dll.lib mkl_core_dll.lib libiomp5md.lib`

* With MKL, use tbb for multithreading, dynamic link  
`mex -v -largeArrayDims -I'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\include' consistent_graph_dca.cpp DCA.cpp helper.cpp qp_simplex_fixed2.cpp linear_algebra.cpp -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\lib\intel64_win' -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\tbb\lib\intel64_win\vc_mt' mkl_intel_lp64_dll.lib mkl_tbb_thread_dll.lib mkl_core_dll.lib tbb.lib`

### Compile the Consistent Graph Learning algorithm in MATLAB to a mex file or C++ application
The Consistent Graph Learning algorithm (_not_ **SGF** or **DGF**) is coded in C++ (or C with slight modification). You can complile the code to a mex file to work with MATLAB or use the code in your C++ or C project. If your dataset size is larger than 20,000, it is highly recommended to install MKL (it's free). 

#### Compiler and compilation options
Both MinGW64 Compiler (g++) and Microsoft Visual Studio Compiler would work (use `mex -setup C++` in MATLAB to choose the compiler). But compiling the code with Microsoft Visual Studio using OpenMP may improve the performance. Also, see [installing MinGW64](https://www.mathworks.com/matlabcentral/fileexchange/52848-matlab-support-for-mingw-w64-c-c-compiler).

* If using MinGW64 for compilation, just remove `COMPFLAGS="$COMPFLAGS /openmp" LINKFLAGS='$LINKFLAGS /nodefaultlib:vcomp;vcompd'`
* If using MKL, __modify the mkl library location__ `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl` and `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\compiler\lib\intel64_win` in the commands depending on where the mkl is installed
* If using MKL, comment out the line `#define NOBLAS` in file `use_blas.h`

#### Compilation commands
##### MATLAB
* NO BLAS and NO MKL  
`mex -v -largeArrayDims COMPFLAGS="$COMPFLAGS /openmp" LINKFLAGS='$LINKFLAGS /nodefaultlib:vcomp;vcompd' consistent_graph_dca.cpp DCA.cpp helper.cpp qp_simplex_fixed2.cpp linear_algebra.cpp`

* With MKL, using openmp for multithreading, static link  
`mex -v -largeArrayDims COMPFLAGS="$COMPFLAGS /openmp" LINKFLAGS='$LINKFLAGS /nodefaultlib:vcomp;vcompd' -I'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\include' consistent_graph_dca.cpp DCA.cpp helper.cpp qp_simplex_fixed2.cpp linear_algebra.cpp -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\lib\intel64_win' -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\compiler\lib\intel64_win' mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib`

* With MKL, using openmp for multithreading, dynamic link  
`mex -v -largeArrayDims COMPFLAGS="$COMPFLAGS /openmp" LINKFLAGS='$LINKFLAGS /nodefaultlib:vcomp;vcompd' -I'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\include' consistent_graph_dca.cpp DCA.cpp helper.cpp qp_simplex_fixed2.cpp linear_algebra.cpp -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\lib\intel64_win' -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\compiler\lib\intel64_win' mkl_intel_lp64_dll.lib mkl_intel_thread_dll.lib mkl_core_dll.lib libiomp5md.lib`

* With MKL, using tbb for multithreading, dynamic link  
`mex -v -largeArrayDims -I'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\include' consistent_graph_dca.cpp DCA.cpp helper.cpp qp_simplex_fixed2.cpp linear_algebra.cpp -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\lib\intel64_win' -L'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\tbb\lib\intel64_win\vc_mt' mkl_intel_lp64_dll.lib mkl_tbb_thread_dll.lib mkl_core_dll.lib tbb.lib`

##### C++
There is a `main` function in the file `consistent_graph_dca.cpp` that demonstrates how to run the algorithms in C++. You can simply comment out the line `#define _MATLAB_` in `consistent_graph_dca.cpp` and complie the code with a C++ compiler to test the algorithms. The simplest compilation option is to use a g++ compiler and run  
`g++ consistent_graph_dca.cpp DCA.cpp helper.cpp qp_simplex_fixed2.cpp linear_algebra.cpp -o main.exe`    
Or compile and run the code in a Visual Studio IDE. Or use the VS command line, e.g., with VS and MKL installed, run the follow command in a VS Native Tools Command Prompt  
`cl /EHsc /openmp consistent_graph_dca.cpp DCA.cpp helper.cpp qp_simplex_fixed2.cpp linear_algebra.cpp /I "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\include" /link /nodefaultlib:vcomp;vcompd /LIBPATH:"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl\lib\intel64_win" /LIBPATH:"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\compiler\lib\intel64_win" mkl_intel_lp64_dll.lib mkl_intel_thread_dll.lib mkl_core_dll.lib libiomp5md.lib`

For more compilation options, please see [MKL compilation tool](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor) and/or [VS compiler options](https://docs.microsoft.com/en-us/cpp/build/reference/compiler-command-line-syntax). 

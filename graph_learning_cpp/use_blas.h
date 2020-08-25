// include guard
#ifndef _USE_BLAS_
#define _USE_BLAS_

/* If the BLAS are not installed, then the following definitions
   can be ignored. If the BLAS are available, then to use them,
   comment out the the next statement (#define NOBLAS) and make
   any needed adjustments to BLAS_UNDERSCORE and the START parameters.
   cg_descent already does loop unrolling, so there is likely no
   benefit from using unrolled BLAS. There could be a benefit from
   using threaded BLAS if the problems is really big. However,
   performing low dimensional operations with threaded BLAS can be
   less efficient than the cg_descent unrolled loops. Hence,
   START parameters should be specified to determine when to start
   using the BLAS. */
#define NOBLAS

/* if BLAS are used, specify the integer precision */
#define BLAS_INT MKL_INT

#define BLAS_START 250 // 250? 300?

/* only use ddot when the vector size >= DDOT_START */
#define DDOT_START BLAS_START

/* only use dcopy when the vector size >= DCOPY_START */
#define DCOPY_START 1000

/* only use daxpy when the vector size >= DAXPY_START */
#define DAXPY_START BLAS_START

/* only use dscal when the vector size >= DSCAL_START */
#define DSCAL_START BLAS_START

/* only use matrix BLAS for transpose multiplication when number of
   elements in matrix >= MATVEC_START */
#define MATVEC_START 1000

#endif
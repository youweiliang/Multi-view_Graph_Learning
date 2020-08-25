#pragma once

#include "use_blas.h"

#ifndef NOBLAS
#define _dnrm2(n, X) cblas_dnrm2(n, X, 1)
#define _dsymv(A, X, Y, n) cblas_dsymv(CblasColMajor, CblasLower, n, 1, A, n, X, 1, 0, Y, 1)
#define _ddot(X, Y, n) cblas_ddot(n, X, 1, Y, 1)
#else
#define _dnrm2(n, X) dnrm2(n, X, 1)
#define _dsymv(A, X, Y, n) asa_matvec(Y, A, X, n, n, 1)
#define _ddot(X, Y, n) asa_dot(X, Y, n)
#endif

#ifndef ONE
#define ONE 1.0
#endif

#ifndef ZERO
#define ZERO 0.0
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef NULL
#define NULL 0
#endif

// compute the 2-norm of a vector
double dnrm2(int N, double* X, int incX);

// Compute the inner product of matrix M: out = M * M'
// Size of M: ld x n, size of out: ld x ld
void matrix_innerproduct(double* M, double* out, int ld, int n);

// matrix multiplication: C = A * B, where A is m-by-k, 
// B is k-by-n, C is m-by-n
void matrix_mul(double* A, double* B, double* C, int m, int k, int n);

// Performs a rank-1 update of a general matrix: A = alpha*x*y'+ A
// where x is an m-element vector, y is an n - element vector,
// A is an m - by - n general matrix.
void rank1ud(double* A, double* x, double* y, int m, int n);

// Solve a system of linear equations A*x = b,
// where A is symmetric, and b could be a matrix, in which
// case number of columns of b should be given (nrhs, default = 1)
// If A is singular, use pseudoinverse of A to solve x, i.e. x = pinv(A)*b
int LE_solver(double* A, double* b, double* x, int n, int nrhs = 1);

// compute the pseudoinverse of A: B = pinv(A)
// A is m-by-n
void pseudoinverse(double* A, double* B, int m, int n);

// Compute the 2-norm of a real symmetric matrix
double matrix2norm(double* H, int n);

// compute the largest eigenvalue of real symmetric matrix A of degree n
double largest_eigv(double* A, int n);

/* =========================================================================
   ==== asa_matvec =========================================================
   =========================================================================
   Compute y = A*x or A'*x where A is a dense rectangular matrix (column major)
   ========================================================================= */
void asa_matvec
(
    double* y, /* product vector */
    double* A, /* dense matrix */
    double* x, /* input vector */
    int     n, /* number of columns of A */
    int     m, /* number of rows of A */
    int     w  /* T => y = A*x, F => y = A'*x */
);

/* =========================================================================
   ==== asa_dot ===========================================================
   =========================================================================
   Compute dot product of x and y, vectors of length n
   ========================================================================= */
double asa_dot
(
    double* x, /* first vector */
    double* y, /* second vector */
    int     n /* length of vectors */
);

double asa_dot0
(
    double* x, /* first vector */
    double* y, /* second vector */
    int     n /* length of vectors */
);

/* =========================================================================
   ==== asa_daxpy ===========================================================
   =========================================================================
   Compute x = x + alpha d
   ========================================================================= */
void asa_daxpy
(
    double* x, /* input and output vector */
    double* d, /* direction */
    double  alpha, /* stepsize */
    int         n  /* length of the vectors */
);

void asa_daxpy0
(
    double* x, /* input and output vector */
    double* d, /* direction */
    double  alpha, /* stepsize */
    int         n  /* length of the vectors */
);

/* =========================================================================
   ==== asa_scale ==========================================================
   =========================================================================
   compute y = s*x where s is a scalar
   ========================================================================= */
void asa_scale
(
    double* y, /* output vector */
    double* x, /* input vector */
    double  s, /* scalar */
    int     n /* length of vector */
);

void asa_scale0
(
    double* y, /* output vector */
    double* x, /* input vector */
    double  s, /* scalar */
    int     n /* length of vector */
);
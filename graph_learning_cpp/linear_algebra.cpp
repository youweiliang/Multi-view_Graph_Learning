#include "linear_algebra.h"
#include "use_blas.h"
#include <string.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <random>

#ifndef NOBLAS
#include "mkl.h"
#endif


/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       Matrix layout in this file is COLUMN MAJOR
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */


// compute the 2-norm of a vector
double dnrm2(int N, double* X, int incX)
{
    int i, n = 1 + (N - 1) * abs(incX);
    double norm2 = 0;
    for (i = 0; i < n; i += incX)
    {
        norm2 += X[i] * X[i];
    }
    return sqrt(norm2);
}

// Compute the inner product of matrix M: out = M * M'
// Size of M: ld x n, size of out: ld x ld
void matrix_innerproduct(double* M, double* out, int ld, int n)
{
    int i, j, k;
    double* p, * end;
    double tmp;
    for (i = 0; i < ld; i++)
    {
        for (j = 0; j < ld; j++)
        {
            tmp = 0;
            k = j - i;
            end = M + ld * (n - 1) + i;
            for (p = M + i; p < end; p += ld)
            {
                tmp += (*p) * (*(p + k));
            }
            out[i * ld + j] = tmp;
        }
    }
}

// matrix multiplication: C = A * B, where A is m-by-k, 
// B is k-by-n, C is m-by-n
void matrix_mul(double* A, double* B, double* C, int m, int k, int n)
{
#ifndef NOBLAS
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, m, B, k, 0.0, C, m);
#else
    for (int i = 0; i < n; i++)
    {
        asa_matvec(C + i * m, A, B + i * k, k, m, 1);
    }
#endif
}

// Performs a rank-1 update of a general matrix: A = alpha*x*y'+ A
// where x is an m-element vector, y is an n - element vector,
// A is an m - by - n general matrix.
void rank1ud(double* A, double* x, double* y, int m, int n)
{
#ifndef NOBLAS
    cblas_dger(CblasColMajor, m, n, 1.0, x, 1, y, 1, A, m);
#else
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; ++j)
        {
            A[j + i * m] += x[j] * y[i];
        }
    }
#endif
}

// Doolittle LU solver, from Philip Wallstedt
// See explanation at http://www.sci.utah.edu/~wallstedt/LU.htm
// Doolittle uses unit diagonals for the lower triangle
int Doolittle(int d, double* S, double* D) {
    for (int k = 0; k < d; ++k) {
        for (int j = k; j < d; ++j) {
            double sum = 0.;
            for (int p = 0; p < k; ++p)
                sum += D[k * d + p] * D[p * d + j];
            D[k * d + j] = (S[k * d + j] - sum); // not dividing by diagonals
        }
        for (int i = k + 1; i < d; ++i) {
            if (D[k * d + k] == 0)
                return 1;
            double sum = 0.;
            for (int p = 0; p < k; ++p)
                sum += D[i * d + p] * D[p * d + k];
            D[i * d + k] = (S[i * d + k] - sum) / D[k * d + k];
        }
    }
    return 0;
}

void solveDoolittle(int d, double* LU, double* b, double* x)
{
    double* y = new double[d];
    for (int i = 0; i < d; ++i) {
        double sum = 0.;
        for (int k = 0; k < i; ++k)sum += LU[i * d + k] * y[k];
        y[i] = (b[i] - sum); // not dividing by diagonals
    }
    for (int i = d - 1; i >= 0; --i) {
        double sum = 0.;
        for (int k = i + 1; k < d; ++k)sum += LU[i * d + k] * x[k];
        x[i] = (y[i] - sum) / LU[i * d + i];
    }
    delete[]y;
}

// Solve a system of linear equations A*x = b, where A is symmetric
// and of size n-by-n, and b could be a matrix, in which case
// the number of columns of b should be given (nrhs, default = 1)
// If A is singular, use pseudoinverse of A to solve x, i.e. x = pinv(A)*b
int LE_solver(double* A, double* b, double* x, int n, int nrhs)
{
    int info;
    double* a = new double[n * n];
    memcpy(a, A, n * n * sizeof(double));
#ifndef NOBLAS
    int* d = new int[n];
    info = LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'L', n, a, n, d);
    if (info > 0)
        goto pinv;
    if (x != b)
        memcpy(x, b, n * sizeof(double));
    LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'L', n, nrhs, a, n, d, x, n);
    delete[]d;
    delete[]a;
    return info;
#endif // !NOBLAS

#ifdef NOBLAS
    info = Doolittle(n, A, a);
    if (info > 0)
        goto pinv;
    for (int i = 0; i < nrhs; ++i)
    {
        solveDoolittle(n, a, b + i * n, x + i * n);
    }
    delete[]a;
    return info;
#endif // NOBLAS

pinv:
#ifdef NOBLAS
    printf("Encounter singular matrix in LE solver. \
        Since no LAPACK is available, unable to solve the linear equations.\
        Please use LAPACK when compiling the program.\n");
    throw "NO LAPACK error";
#else
    pseudoinverse(A, a, n, n);
    matrix_mul(a, b, x, n, n, nrhs);
    delete[]a;
    return info;
#endif
}

#ifndef NOBLAS
// compute the pseudoinverse of A: B = pinv(A)
// A is m-by-n
void pseudoinverse(double* A, double* B, int m, int n)
{
    MKL_INT  k = m < n ? m : n;
    MKL_INT  lda = m; //column-major, n for row-major
    MKL_INT  ldu = m; //column-major, min(m,n) for row-major
    MKL_INT  ldvt = k; //column-major, n for row-major
    MKL_INT  lwork;
    MKL_INT info;
    double wkopt;
    double* work;
    double* s = (double*)malloc(k * sizeof(double));
    double* u = (double*)malloc(ldu * k * sizeof(double));
    double* vt = (double*)malloc(ldvt * n * sizeof(double));

    /* Query and allocate the optimal workspace */
    lwork = -1;
    char jobu = 'S';
    char jobvt = 'S';
    dgesvd(&jobu, &jobvt, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, &info);
    lwork = (MKL_INT)wkopt;
    work = (double*)malloc(lwork * sizeof(double));

    /* Compute SVD */
    dgesvd(&jobu, &jobvt, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);

    /* Check for convergence */
    if (info > 0) {
        printf("The algorithm computing SVD failed to converge.\n");
        throw "SVD fail to converge";
    }

    //u=(s^-1)*U
    MKL_INT incx = 1;

    for (int i = 0; i < k; i++)
    {
        double ss;
        if (s[i] > 1.0e-9)
            ss = 1.0 / s[i];
        else
            ss = s[i];
        dscal(&m, &ss, &u[i * m], &incx);
    }

    //pinv(A)=(Vt)^T *u^T
    double alpha = 1.0, beta = 0.0;
    MKL_INT ld_inva = n;
    dgemm("T", "T", &n, &m, &k, &alpha, vt, &ldvt, u, &ldu, &beta, B, &ld_inva);

    free(s);
    free(u);
    free(vt);
    free(work);
}
#endif

// Compute the 2-norm of a real symmetric matrix
// Warning: without LAPACK, the algorithm may fail to converge when 
// the two largest absolute eigenvalues are the same or close.
double matrix2norm(double* H, int n)
{
#ifndef NOBLAS
    int m, isuppz, info;
    double * w, z;
    w = new double[n];
    info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'N', 'A', 'L', n, H, n, 0, 0, n, n, dlamch("S"), &m, w, &z, 1, &isuppz);
    if (info > 0)
    {
        delete[]w;
        printf("Failed to compute the largest eigenvalue of the matrix, an internal error has occurred in MKL\n");
        throw "MKL internal error";
    }
    else if (info < 0)
    {
        delete[]w;
        printf("%d-th parameter had an illegal value.", info);
        throw "MKL parameter error";
    }
    z = fabs(w[0]);
    for (int i = 1; i < n; ++i)
    {
        if (z < fabs(w[i]))
        {
            z = fabs(w[i]);
        }
    }
    delete[]w;
    return z;
#else
    // power iteration
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::default_random_engine re;
    double vnorm, old_vnorm, diff, * v0, * v1;
    int i, n_iter, max_iter;
    v0 = new double[n];  // initial vectors for power iteration
    v1 = new double[n];

    for (i = 0; i < n; ++i)
    {
       v0[i] = uniform(re);
    }

    diff = 1;
    n_iter = 0;
    max_iter = 10000;

    old_vnorm = _dnrm2(n, v0);

    // U to compute the 2-norm of the Hessian H.
    // Since H is randomly distributed, it's unlikely that its eigenvalues coincide,
    // we don't need to worry power iteration will fail to converge.
    while (diff > 1e-8)
    {
        n_iter++;
        if (n_iter > max_iter)
        {
            printf("Power iteration fail to converge in file linear_algebra.cpp");
            throw "Power iteration fail to converge in file linear_algebra.cpp";
        }
        for (i = 0; i < n; ++i)
           v0[i] /= old_vnorm;

        _dsymv(H, v0, v1, n);  // write the result to v1
        _dsymv(H, v1, v0, n);  // write the result to v0

        vnorm = _dnrm2(n, v0);
        diff = fabs(vnorm - old_vnorm);
        old_vnorm = vnorm;
    }
    
    delete[] v0;
    delete[] v1;
    return sqrt(vnorm);
#endif
}

// compute the largest eigenvalue of real symmetric matrix A of degree n
// Warning: without LAPACK, the algorithm may fail to converge when 
// the two largest absolute eigenvalues are the same or close.
double largest_eigv(double* A, int n)
{
#ifndef NOBLAS
    int m, isuppz, info;
    double w, z;
    info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'N', 'I', 'L', n, A, n, 0, 0, n, n, dlamch("S"), &m, &w, &z, 1, &isuppz);
    if (!info)
        return w;
    else if (info > 0)
    {
        printf("Failed to compute the largest eigenvalue of the matrix, an internal error has occurred in MKL\n");
        throw "MKL internal error";
    }
    else
    {
        printf("%d-th parameter had an illegal value.", info);
        throw "MKL parameter error";
    }

#else
    // power iteration
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::default_random_engine re;
    double vnorm, old_vnorm, diff, * v0, * v1;
    int i, n_iter, max_iter;
    v0 = new double[n];  // initial vectors for power iteration
    v1 = new double[n];

    for (i = 0; i < n; ++i)
    {
       v0[i] = uniform(re);
    }

    diff = 1;
    n_iter = 0;
    max_iter = 10000;

    old_vnorm = _dnrm2(n, v0);
    for (i = 0; i < n; ++i)
    v0[i] /= old_vnorm;

    // U to compute the 2-norm of the Hessian H.
    // Since H is randomly distributed, it's unlikely that its eigenvalues coincide,
    // we don't need to worry power iteration will fail to converge.
    while (1)
    {
        n_iter++;
        if (n_iter > max_iter)
        {
            printf("Power iteration fail to converge in file linear_algebra.cpp");
            throw "Power iteration fail to converge in file linear_algebra.cpp";
        }

        _dsymv(A, v0, v1, n);  // write the result to v1

        vnorm = _dnrm2(n, v1);
        diff = fabs(vnorm - old_vnorm);
        old_vnorm = vnorm;
        if (diff < 1e-8)
        {
            break;
        }
        for (i = 0; i < n; ++i)
        {
           v0[i] = v1[i] / vnorm;
        }
    }

    double eig = 0;
    for (i = 0; i < n; ++i)
    {
        eig += v1[i] / v0[i];
    }
    eig /= n;
    if (eig >= 0)
    {
        delete[] v0;
        delete[] v1;
        return eig;
    }

    // set A = A - eig*I and recompute the largest eigenvalue
    double eig0 = eig;
    for (i = 0; i < n; ++i)
    {
        A[i + i * n] -= eig;
    }

    for (i = 0; i < n; ++i)
    {
       v0[i] = uniform(re);
    }

    diff = 1;
    n_iter = 0;
    max_iter = 10000;

    old_vnorm = _dnrm2(n, v0);
    for (i = 0; i < n; ++i)
    v0[i] /= old_vnorm;

    while (1)
    {
        n_iter++;
        if (n_iter > max_iter)
        {
            printf("Power iteration fail to converge in file linear_algebra.cpp");
            throw "Power iteration fail to converge in file linear_algebra.cpp";
        }

        _dsymv(A, v0, v1, n);  // write the result to v1

        vnorm = _dnrm2(n, v1);
        diff = fabs(vnorm - old_vnorm);
        old_vnorm = vnorm;
        if (diff < 1e-8)
        {
            break;
        }
        for (i = 0; i < n; ++i)
        {
           v0[i] = v1[i] / vnorm;
        }
    }
    eig = 0;
    for (i = 0; i < n; ++i)
    {
        eig += v1[i] / v0[i];
    }
    eig /= n;
    eig += eig0;
    delete[] v0;
    delete[] v1;
    return eig;
#endif
}

/* =========================================================================
   ==== asa_scale0 =========================================================
   =========================================================================
   compute y = s*x where s is a scalar
   ========================================================================= */
void asa_scale0
(
    double* y, /* output vector */
    double* x, /* input vector */
    double  s, /* scalar */
    int     n /* length of vector */
)
{
    int i;
    if (s == -ONE)
    {
        for (i = 0; i < n; i++)
        {
            y[i] = -x[i];
        }
    }
    else
    {
        for (i = 0; i < n; i++)
        {
            y[i] = s * x[i];
        }
    }
    return;
}

/* =========================================================================
   ==== asa_daxpy0 ==========================================================
   =========================================================================
   Compute x = x + alpha d
   ========================================================================= */
void asa_daxpy0
(
    double* x, /* input and output vector */
    double* d, /* direction */
    double  alpha, /* stepsize */
    int         n  /* length of the vectors */
)
{
    int i;
    if (alpha == -ONE)
    {
        for (i = 0; i < n; i++)
        {
            x[i] -= d[i];
        }
    }
    else
    {
        for (i = 0; i < n; i++)
        {
            x[i] += alpha * d[i];

        }
    }
    return;
}

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
)
{
    /* if the blas have not been installed, then hand code the produce */
#ifdef NOBLAS
    int j, l;
    l = 0;
    if (w)
    {
        asa_scale0(y, A, x[0], (int)m);
        for (j = 1; j < n; j++)
        {
            l += m;
            asa_daxpy0(y, A + l, x[j], (int)m);
        }
    }
    else
    {
        for (j = 0; j < n; j++)
        {
            y[j] = asa_dot0(A + l, x, (int)m);
            l += m;
        }
    }
#endif

    /* if the blas have been installed, then possibly call gdemv */
#ifndef NOBLAS
    int j, l;
    BLAS_INT M, N;
    M = (BLAS_INT)m;
    N = (BLAS_INT)n;
    if (m * n < MATVEC_START)
    {
        l = 0;
        if (w)
        {
            asa_scale(y, A, x[0], m);
            for (j = 1; j < n; j++)
            {
                l += m;
                asa_daxpy(y, A + l, x[j], m);
            }
        }
        else
        {
            for (j = 0; j < n; j++)
            {
                y[j] = asa_dot0(A + l, x, (int)m);
                l += m;
            }
        }
    }
    else
    {
        if (w)
            cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, 1.0, A, M, x, 1, 0.0, y, 1);
        else
            cblas_dgemv(CblasColMajor, CblasTrans, M, N, 1.0, A, M, x, 1, 0.0, y, 1);
    }
#endif

    return;
}


/* =========================================================================
   ==== asa_dot ============================================================
   =========================================================================
   Compute dot product of x and y, vectors of length n
   ========================================================================= */
double asa_dot
(
    double* x, /* first vector */
    double* y, /* second vector */
    int     n /* length of vectors */
)
{
#ifdef NOBLAS
    int i;
    double t;
    t = ZERO;
    if (n <= 0) return (t);
    for (i = 0; i < n; i++)
    {
        t += x[i] * y[i];
    }
    return (t);
#endif

#ifndef NOBLAS
    int i;
    double t;
    BLAS_INT N;
    if (n < DDOT_START)
    {
        t = ZERO;
        if (n <= 0) return (t);
        for (i = 0; i < n; i++)
        {
            t += x[i] * y[i];
        }
        return (t);
    }
    else
    {
        N = (BLAS_INT)n;
        return (cblas_ddot(N, x, 1, y, 1));
    }
#endif
}


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
)
{
#ifdef NOBLAS
    int i;
    if (alpha == -ONE)
    {
        for (i = 0; i < n; i++)
        {
            x[i] -= d[i];
        }
    }
    else
    {
        for (i = 0; i < n; i++)
        {
            x[i] += alpha * d[i];
        }
    }
#endif

#ifndef NOBLAS
    int i;
    BLAS_INT N;
    if (n < DAXPY_START)
    {
        if (alpha == -ONE)
        {
            for (i = 0; i < n; i++)
            {
                x[i] -= d[i];
            }
        }
        else
        {
            for (i = 0; i < n; i++)
            {
                x[i] += alpha * d[i];
            }
        }
    }
    else
    {
        N = (BLAS_INT)n;
        cblas_daxpy(N, alpha, d, 1, x, 1);
    }
#endif

    return;
}


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
)
{
    int i;
    if (y == x)
    {
#ifdef NOBLAS
        for (i = 0; i < n; i++)
        {
            y[i] *= s;
        }
#endif
#ifndef NOBLAS
        if (n < DSCAL_START)
        {
            for (i = 0; i < n; i++)
            {
                y[i] *= s;
            }
        }
        else
        {
            BLAS_INT N;
            N = (BLAS_INT)n;
            cblas_dscal(N, s, x, 1);
        }
#endif
    }
    else
    {
        for (i = 0; i < n; i++)
        {
            y[i] = s * x[i];
        }
    }
    return;
}


/* =========================================================================
   ==== asa_dot0 ===========================================================
   =========================================================================
   Compute dot product of x and y, vectors of length n
   ========================================================================= */
double asa_dot0
(
    double* x, /* first vector */
    double* y, /* second vector */
    int     n /* length of vectors */
)
{
    int i;
    double t;
    t = ZERO;
    if (n <= 0) return (t);
    for (i = 0; i < n; i++)
    {
        t += x[i] * y[i];
    }
    return (t);
}
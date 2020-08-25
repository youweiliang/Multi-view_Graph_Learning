#include <stdio.h>
#include <string.h>
#include <math.h>
#include "helper.h"
#include "use_blas.h"

#ifdef NOBLAS
#include "linear_algebra.h"
#else
#include "mkl.h"
#endif

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	   Matrix layout in this file is COLUMN MAJOR
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void a_diag(double* A, const double* w, double* diag, int v, int n)
{
	int i;
	double tmp;
#ifndef NOBLAS
	for (i = 0; i < v; i++)
	{
		tmp = cblas_dnrm2(n, A + i, v);
		diag[i] = tmp * tmp * w[i];
	}
#else
	for (i = 0; i < v; i++)
	{
		tmp = dnrm2(n, A + i, v);
		diag[i] = tmp * tmp * w[i];
	}
#endif
}

void a_whole(double* a_H, double* E, double* C, int v, int n)
{
#ifndef NOBLAS
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, v, v, n, 1.0, E, v, E, v, 0.0, a_H, v);
#endif

#ifdef NOBLAS
	matrix_innerproduct(E, a_H, v, n);
#endif
	int k = v * v;
	for (int i = 0; i < k; i++)
	{
		a_H[i] *= C[i];
	}
}

void a_linear(double* A, double* S, const double* w, double* L, int v, int n)
{
#ifndef NOBLAS
	cblas_dgemv(CblasColMajor, CblasNoTrans, v, n, 1.0, A, v, S, 1, 0.0, L, 1);
#else
	asa_matvec(L, A, S, n, v, 1);
#endif
	for (int i = 0; i < v; i++)
	{
		L[i] *= w[i];
	}
}

void fill_zero(double* A, int n)
{
	for (int i = 0; i < n; i++)
	{
		A[i] = 0.0;
	}
}

double vec_diff(double* v1, double* v2, int n) {
	double tmp, norm = 0;
	int i;
	for (i = 0; i < n; i++) {
		tmp = v1[i] - v2[i];
		norm += tmp * tmp;
	}
	return norm;
}

void matrix_subtraction(double* v1, double* v2, double* output, int n) {
	int i;
	for (i = 0; i < n; i++)
		output[i] = v1[i] - v2[i];
}

// project A into the box of W such that 0 <= A <= W
void box_proj(double* A, double* W, int n)
{
	for (int i = 0; i < n; ++i)
	{
		if (A[i] < 0.0)
		{
			A[i] = 0.0;
		}
		else if (A[i] > W[i])
		{
			A[i] = W[i];
		}
	}
}

// print a column-major matrix (size: n x m)
void print_matrix(double* A, int n, int m, int acc)
{
	int i, j;
	printf("\n");
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			printf("%.*f ", acc, A[n * j + i]);
		}
		printf("\n");
	}
}

// For debug only!
// Compute the objective of consistent graph learning algorithm
// H acts only as temporary storage in this function!
double objective(double* alpha, double* A, double* S,
	const double* w, double* C, double* E, double* H, int v, int n)
{
	int k;
	double ai, f1, f2, fi, tmp;
	f1 = f2 = 0.0;
	for (int i = 0; i < v; i++)
	{
		ai = alpha[i];
		fi = 0;
		for (int j = 0; j < n; j++)
		{
			tmp = ai * A[j * v + i] - S[j];
			fi += tmp * tmp;
		}
		f1 += fi * w[i];
	}

	a_whole(H, E, C, v, n);

	for (int i = 0; i < v; i++)
	{
		for (int j = 0; j < v; j++)
		{
			k = i + j * v;
			f2 += H[k] * alpha[i] * alpha[j];
		}
	}

	return f1 + f2;
}


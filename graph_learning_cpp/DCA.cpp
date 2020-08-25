// Implementation of the difference of convex algorithm (DCA) 
// for non-covex quadratic programming with box constraints

#include <cstdio>
#include <cstring>
#include "linear_algebra.h"
#include "helper.h"

// #define TOL_ITER 3

// Algorithm 2a and 2b in the paper "A Branch and Bound Method via d.c. Optimization 
// Algorithms and Ellipsoidal Technique for Box Constrained Nonconvex Quadratic Problems"
// min_x  0.5 * x^T * A * x - b^T * x
// s.t.   lb <= x <= ub
// Input:
//  A  - Hessian
//  b  - linear term
//  x0 - staring point (must be feasible)
//  lb - lower bound (in consistent graph learning lb = 0, so omitted)
//  ub - upper bound
//  n  - # of rows of A
//  m  - number of QP problems that needed to be solved
//  y  - temperary allocated memory for computation (same size as x0)
void DCA(double* A, double* b, double* x0, double* ub, int n, int m, double* y, int TOL_ITER = 3)
{
	double eigv, tmp, * x = x0;
	int k = n * n;
	double* C = new double[n * n];
	memcpy(C, A, n * n * sizeof(double));

	// !!!! this function would alter A !!!!
	eigv = largest_eigv(A, n);

	if (eigv > 0)  // ALGORITHM 2a
	{
		for (int i = 0; i < k; i++)
		{
			C[i] = -C[i];
		}
		for (int i = 0; i < n; i++)
		{
			C[i + i * n] += eigv;
		}

		for (int j = 0; j < TOL_ITER; j++)
		{
			matrix_mul(C, x, y, n, n, m);

			k = n * m;
			for (int i = 0; i < k; i++)
			{
				tmp = (y[i] + b[i]) / eigv;
				if (tmp < 0.0)  // smaller than lower bound
				{
					x[i] = 0.0;
				}
				else if (tmp > ub[i])
				{
					x[i] = ub[i];
				}
				else
				{
					x[i] = tmp;
				}
			}
		}
		
	}
	// A is negative semi-definite, use ALGORITHM 2b
	// but this case should never happen in consistent graph learning algorithm
	// so comment out the algorithm and add error prompt
	else
	{
		printf("A should not be negative semi-definite in DCA\n");
		throw "negative semi-definite error in DCA.cpp";

		/*for (int j = 0; j < TOL_ITER; j++)
		{
			matrix_mul(C, x, y, n, n, m);

			k = n * m;
			for (int i = 0; i < k; i++)
			{
				tmp = y[i] - b[i];
				if (tmp < 0)
				{
					x[i] = ub[i];
				}
				else
				{
					x[i] = lb[i];
				}
			}
		}*/
	}

	delete[]C;
}


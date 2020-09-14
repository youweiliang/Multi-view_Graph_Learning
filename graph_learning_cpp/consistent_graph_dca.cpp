// if not using within MATLAB, please comment out the next line
#define _MATLAB_ 

#ifdef _MATLAB_
#include "mex.h"
#endif

#include "use_blas.h"
#include "linear_algebra.h"
#include "helper.h"
#include "qp_simplex.h"

#ifndef NOBLAS
#include "mkl.h"
#endif

#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>


/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Matrix layout in this file is COLUMN MAJOR
    If running on machines with Intel processor, it's recommended to install
    the free Intel MKL library for full performance. Another option is to 
    install BLAS and LAPACK (with standard C language API) manually. See the 
    link for this option: http://www.netlib.org/lapack/lapacke.html 
    The last option is to compile without any external libraries. However,
    it may cause the program to fail in rare cases when a singular matrix is 
    encountered in function "LE_solver" in linear_algebra.cpp
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

// Default tolerance for determining convergence
double TOL = 1e-6, TOL_QPS = 1e-6;
int MAX_ITER = 50;

#ifdef _MATLAB_
/************************************************************************
Consistent graph learning:
Combine multiple graphs into one graph.
Inputs:
    v - # of graphs
    n - # of element in each adjacency matrix W^(i)
    W - an array containing the matrix of each graph,
        the i-th row of W store the elements in W^(i),
        size: v x n
    b - penalty parameters in original paper
    w - weight for each graph
    tol - (optional) tolerance to determine convergence
    max_iter - (optional) # of maximum iterations
    control_iter - (optional) control the # of iteration in DCA
************************************************************************/
void mexFunction(int nlhs, mxArray* plhs[],
    int nrhs, const mxArray* prhs[])
#else
double * consistent_graph(const int v, const int n, double* W,
    const double* b, const double* w, double* S, int max_iter = MAX_ITER,
    double tol = TOL, double tol_q = TOL_QPS, bool control_iter = false)
#endif
{
    double* A_Hessian, * L, * tmp;
    double* alpha, * old_alpha, * A, * E, * diag;
    double* a_Hessian, * aH, * B, * C, * aw, * aw2, * zeros, * H;
    double td, wa, tol1, obj, tmp2, tmp3, w_sum;
    int i, j, k, sz_a, n_all, flag, * aset, call_qp;
    
#ifdef _MATLAB_
    int v, n, max_iter = MAX_ITER;
    double tol = TOL, tol_q = TOL_QPS;
    bool control_iter = false;
    double* W, * b, * w, * shape, * S;
    if (nrhs < 5 || nrhs > 9)
        mexErrMsgIdAndTxt("MyToolbox:Consistent:nrhs", "5~9 inputs required.");
    v = (int)mxGetScalar(prhs[0]);
    n = (int)mxGetScalar(prhs[1]);
    W = mxGetPr(prhs[2]);
    b = mxGetPr(prhs[3]);
    w = mxGetPr(prhs[4]);
    S = new double[n];
    if (nrhs == 5)
        goto p5;
    else if (nrhs == 6)
        goto p6;
    else if (nrhs == 7)
        goto p7;
    else if (nrhs == 8)
        goto p8;
    else if (nrhs == 9)
        goto p9;
p9:
    control_iter = (bool)mxGetScalar(prhs[8]);
p8:
    max_iter = (int)mxGetScalar(prhs[7]);
p7:
    tol_q = mxGetScalar(prhs[6]);
p6:
    tol = mxGetScalar(prhs[5]);
p5:

#else

#endif

#ifndef NOBLAS
    int n_thread = mkl_get_max_threads();
    // printf("Using %d threads \n", n_thread);
    mkl_set_num_threads(n_thread);
#endif

    // initialize varibles
    tol1 = tol / v;
    sz_a = v * sizeof(double);
    n_all = v * n;
    alpha = new double[v]();
    old_alpha = new double[v]();
    A = new double[n_all];  // consistent part for each view
    E = new double[n_all]();  // the inconsistency part: E = W - A 
    diag = new double[v];
    a_Hessian = new double[v * v]();
    aH = new double[(v + 1) * (v + 1)];
    B = new double[v + 1];
    
    C = new double[v * v];
    aw = new double[v];
    aw2 = new double[v];
    zeros = new double[v]();
    aset = new int[v]();  // active set indicator
    H = new double[v * v];

    A_Hessian = new double[v * v];
    L = new double[v * n];
    tmp = new double[v];

    double* scale;
    scale = new double[v]();

    // scaling each view properly will accelerate the optimization process
    // all elements in W should be non-negative to allow this to work
    for (i = 0, k = 0; i < n; i++)
    {
        for (j = 0; j < v; j++)
        {
            scale[j] += W[k++];
        }
    }
    for (j = 0; j < v; j++)
    {
        scale[j] = 1.0 / scale[j];
    }
    for (i = 0, k = 0; i < n; i++)
    {
        for (j = 0; j < v; j++)
        {
            W[k++] *= scale[j];
        }
    }

    tmp3 = 0.0;
    for (i = 0; i < v; i++)
    {
        // initialize alpha
        tmp2 = 0.0;
        for (j = 0; j < n; ++j)
        {
            tmp2 += W[j * v + i];
        }
        alpha[i] = 1.0 / tmp2;
        tmp3 += alpha[i];

        for (j = 0; j < v; j++)
        {
            C[i + j * v] = b[i + j * v] * w[i] * w[j];
        }
    }
    w_sum = 0.0;
    for (i = 0; i < v; i++)
    {
        alpha[i] /= tmp3;  // alpha should be summed to 1
        w_sum += w[i];
    }

    memcpy(A, W, n_all * sizeof(double)); // initalize A as W

    old_alpha[0] = 10.0; // to avoid convergence at first iteration
    
    // initialize S
    for (i = 0; i < v; i++)
    {
        aw[i] = alpha[i] * w[i];
        aw2[i] = aw[i] / w_sum;
    }

    asa_matvec(S, A, aw2, n, v, 0);
    //printf("%f\n", objective(alpha, A, S, w, C, E, a_Hessian, v, n));

    int iter = 0;
    int iter_obj = 0;
    int dca_iter = 3;
    call_qp = 0;
    double* objs = new double[max_iter+1]();
    double* a_linear_c = new double[v];

    objs[iter] = objective(alpha, A, S, w, C, E, a_Hessian, v, n);

    while (iter < max_iter) {
        iter++;

        // solve for alpha
        a_diag(A, w, diag, v, n);  // O(v*n)
        
        a_whole(a_Hessian, E, C, v, n);  // O(v^2 * n)

        for (i = 0; i < v; i++)
        {
            a_Hessian[i + i * v] += diag[i];
        }

        a_linear(A, S, w, B, v, n);  // O(v*n)
        memcpy(a_linear_c, B, v * sizeof(double));

        B[v] = 1.0;
        for (i = 0, j = (v+1) * v; i < v; i++, j++)
        {
            memcpy(aH + i * (v + 1), a_Hessian + i * v, v * sizeof(double));
            aH[(v + 1) * i + v] = 1.0;
            aH[j] = 1.0;
        }
        aH[j] = 0.0;
        
        LE_solver(aH, B, B, v + 1);  // O(v^3)

        flag = 0;
        for (i = 0; i < v; i++)
        {
            if (B[i] < 0.0)
            {
                flag = 1;
                break;
            }
        }

        if (flag)
        {
            call_qp += 1; // printf("call qp_simplex\m");
            QPSimplex(a_Hessian, a_linear_c, alpha, aset, v, tol_q);  // O(v^3)
        }
        else
            memcpy(alpha, B, v * sizeof(double));
        
        //printf("%f\n", objective(alpha, A, S, w, C, E, a_Hessian, v, n));
        
        objs[iter] = objective(alpha, A, S, w, C, E, a_Hessian, v, n);
        // if (vec_diff(alpha, old_alpha, v) <= tol1) 
        //     break;
        if (iter >= 5 && 
                (objs[iter-1] - objs[iter]) < 0.05 * (objs[0] - objs[iter]))
        {
            break;
        }
        
        memcpy(old_alpha, alpha, sz_a);

        // solve for S
        for (i = 0; i < v; i++)
        {
            aw[i] = alpha[i] * w[i];
            aw2[i] = aw[i] / w_sum;
        }

        asa_matvec(S, A, aw2, n, v, 0);

        for (i = 0; i < v; i++)
        {
            for (j = 0; j < v; j++)
            {
                k = i + j * v;
                H[k] = C[k] * alpha[i] * alpha[j];
            }
        }

        memcpy(A_Hessian, H, v * v * sizeof(double));

        for (j = 0; j < v; j++)
        {
            A_Hessian[j + j * v] += (w[j] * alpha[j] * alpha[j]);
        }

        matrix_mul(H, W, L, v, v, n);  // O(v^2 * n)

        rank1ud(L, aw, S, v, n);  // O(v * n)

        // Here E acts as temperary storage, not the inconsistent part
        // This function would alter A_Hessian!
        DCA(A_Hessian, L, A, W, v, n, E, dca_iter);  // O(v^2 * n)
        
        if (dca_iter > 1 && control_iter)
            dca_iter--;

        // Here E acts as the inconsistent part
        matrix_subtraction(W, A, E, n_all);

        //printf("%f\n", objective(alpha, A, S, w, C, E, a_Hessian, v, n));
    }
    /*printf("%e\n", objective(alpha, A, S, w, C, E, a_Hessian, v, n));
    printf("iter: %d\n", iter);*/

#ifdef _MATLAB_
    // output fused matrix, alpha (optional), and E (optional)
    plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
    memcpy(mxGetPr(plhs[0]), S, n * sizeof(double));
    delete[]S;
    if (nlhs >= 2) {
        plhs[1] = mxCreateDoubleScalar(call_qp);
    }
    if (nlhs >= 3) {
        plhs[2] = mxCreateDoubleScalar(iter);
    }
    if (nlhs >= 4) {
        plhs[3] = mxCreateDoubleMatrix(v, 1, mxREAL);
        memcpy(mxGetPr(plhs[3]), alpha, v * sizeof(double));
    }
    if (nlhs >= 5) {
        plhs[4] = mxCreateDoubleMatrix(v, 1, mxREAL);
        memcpy(mxGetPr(plhs[4]), scale, v * sizeof(double));
    }
    if (nlhs >= 6) {
        plhs[5] = mxCreateDoubleMatrix(max_iter, 1, mxREAL);
        memcpy(mxGetPr(plhs[5]), objs, max_iter * sizeof(double));
    }
    /*if (nlhs >= 3) {
        plhs[2] = mxCreateDoubleMatrix(v, n, mxREAL);
        memcpy(mxGetPr(plhs[2]), E, n_all * sizeof(double));
    }*/

    delete[]alpha;
#endif

    delete[]old_alpha;
    delete[]A;
    delete[]E;
    delete[]diag;
    delete[]a_Hessian;
    delete[]aH;
    delete[]B;
    delete[]C;
    delete[]aw;
    delete[]aw2;
    delete[]zeros;
    delete[]aset;
    delete[]H;
    delete[]A_Hessian;
    delete[]L;
    delete[]tmp;
    delete[]scale;

#ifndef _MATLAB_
    return alpha;
#endif
}

#ifndef _MATLAB_
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>

void gen(int v, int n, double* W, double* b, double* w, 
    std::default_random_engine re1, std::uniform_real_distribution<double> uniform1)
{
    int i, j;
    double* scale;
    scale = new double[v];
    double thr = 0.1;
    bool incon;
    for (i = 0; i < v; i++)
    {
        scale[i] = -log(uniform1(re1));
    }
    for (i = 0; i < v * n; i += v)
    {
        
        W[i] = uniform1(re1);
        for (j = 1; j < v; j++)
        {
            incon = uniform1(re1) < thr;
            if (incon)
                W[i + j] = W[i] * (1 - 0.6) * scale[j];
            else
                W[i + j] = W[i] * (1 - 0.1 * uniform1(re1)) * scale[j];
        }
        W[i] *= scale[0];
    }

    for (i = 0; i < v; ++i)
    {
        for (j = 0; j < v; j++)
        {
            if (i == j)
                b[i * v + j] = v * 1e-5;
            else
                b[i * v + j] = v * 1e-1;
        }
        w[i] = 1.0 / v;
    }
    delete[]scale;
}

int main()
{
    std::random_device rd;
    //std::knuth_b e2(rd());
    std::default_random_engine re1(rd());
    std::uniform_real_distribution<double> uniform1(0.1, 1);

    int i, j, k, v = 4, n = 1000;
    double* W, * W2, * b, * w, * S;
    W = new double[v * n];
    W2 = new double[v * n];
    b = new double[v * v];
    w = new double[v];
    S = new double[n]();
    gen(v, n, W, b, w, re1, uniform1);
    memcpy(W2, W, v * n * sizeof(double));

    auto t1 = std::chrono::high_resolution_clock::now();
    consistent_graph(v, n, W, b, w, S, 300, 1e-8, 1e-8);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << duration << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    double * alpha = consistent_graph(v, n, W2, b, w, S, 300, 5e-10, 1e-8, true);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << duration << std::endl;

    /*for (i = 0; i < n; i++)
    {
        S[i] *= v;
    }
    for (i = 0, k = 0; i < n; i++)
    {
        for (j = 0; j < v; j++)
        {
            W2[k++] *= alpha[j] * v;
        }
    }
    print_matrix(W2, v, n, 4);
    print_matrix(S, 1, n, 4);*/
    
    delete[]W;
    delete[]W2;
    delete[]b;
    delete[]w;
    delete[]S;
}
#endif

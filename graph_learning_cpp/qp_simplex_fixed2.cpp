#include <string>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <cstdio>
#include <vector>

#include "use_blas.h"
#include "qp_simplex.h"
#include "linear_algebra.h"
#include "helper.h"
#ifndef NOBLAS
#include "mkl.h"
#endif

#ifdef NOBLAS
#include "helper.h"
#endif

//#define _MATLAB_
#ifdef _MATLAB_
//#include "mex.h"
#include "C:\Program Files\Polyspace\R2019b\extern\include\mex.h"

void mexFunction(int nlhs, mxArray* plhs[],
    int nrhs, const mxArray* prhs[])
{

    int n, * A;
    double* H, * b, * x;

    n = (int)mxGetScalar(prhs[0]);
    H = mxGetPr(prhs[1]);
    b = mxGetPr(prhs[2]);
    x = mxGetPr(prhs[3]);
    A = new int[n];
    for (int i = 0; i < n; i++)
    {
        if (x[i] == 0)
            A[i] = 1;
        else
            A[i] = 0;
    }

    QPSimplex(H, b, x, A, n);

    delete[]A;

    plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
    memcpy(mxGetPr(plhs[0]), x, n * sizeof(double));
}
#endif

// A routine using Intel MKL to solve non-convex quadratic programming over the unit simplex.
// min. 1/2x*H*x - b*x 
// s.t. x*1 = 1, x >= 0
// Reference: An Active-Set Algorithmic Framework for Non-Convex Optimization Problems over the Simplex
// Inputs:
//    H - Hessian
//    b - linear term
//    x - current feasible point
//    A - active set indicator (1: in active set, 0: not in active set)
//    n - length of x
void QPSimplex(double *H, double *b, double *x, int *A, int n, double tol)
{
    double vnorm, old_vnorm, mu_i, nL, cl, x_diff, x_in, t;
    double x_sqared_sum, x_cum, max_reduction, lambda, val, min_grad;
    double diff, c2, step_max, step, dtmp, pg_norm, t1, t2, t3;
    double *grad, *x_tmp, *v1, *CL, *f_reduction;
    double *proj_grad, *tmp, *tmp2, *Hz, *rd_grad;
    int i, j, k, J, m, n_iter, info, idx_in, neg_count, cg_step;
    int *idx, *A_old, *B, A_change;
    std::vector<int*> n_count(10);
    grad = new double[n];  // gradient of the quadratic function
    CL = new double[n];
    f_reduction = new double[n];
    proj_grad = new double[n];
    x_tmp = new double[n];
    tmp = new double[n];
    tmp2 = new double[n];
    idx = new int[n];
    B = new int[n];
    A_old= new int[n]();
    Hz = new double[n * n];
    rd_grad = new double[n];

    double alpha = 0.5, beta = 1.5, mu=0.1, rou=0.5;
    double angle = cos(10. / 360.);   // to choose between reduced gradient and projected gradient
    double angle1 = cos(1.1 / 360.);  // to choose between conjugate gradient and gradient
    int n1 = n, count, U, proj_count;
    count = 0;
    neg_count = 0;
    info = -3;
    diff = 1;
    max_reduction = 0;
    A_old[0] = -3;

    vnorm = matrix2norm(H, n);
    nL = n * vnorm / 2;
    /*double old_val;
    old_val = QP_objective(H, b, x, tmp, n);
    val = old_val;*/

    n_iter = 0;
    while (true) 
    {
        n_iter++;

        // Subroutine 1: update the active set

        // Compute the gradient
        compute_grad(H, b, x, grad, n);
        
        J = indexofSmallestElement(grad, n, &min_grad);

        lambda = _ddot(grad, x, n);

        for (i = 0; i < n; i++)
        {
            mu_i = grad[i] - lambda;
            if (mu_i < 0) {
                if (x[i] != 0)
                    A[i] = 0;
                else
                    A[i] = -1;  // free varible but at the boundary of feasible set
                CL[i] = -1;
            }
            else if (x[i] == 0) {
                A[i] = 1;
                CL[i] = DBL_MAX;
            }
            else {  // in this case A[i] need further computation to decide (namely, the value of epsilon)
                CL[i] = (mu_i) / x[i] - nL;  // actually this is n * CL, but a constant coefficient won't matter
                A[i] = 0;  // just set it temporarily
            }
        }
        
        for (i = 0; i < n; ++i)
            idx[i] = i;

        sort_indexes(CL, idx, n);

        x_diff = 0;
        x_sqared_sum = 0;
        for (i = 0; i < n; i++)
        {
            f_reduction[i] = 0.;
        }
        for (i = n - 1; i >= 0; --i) 
        {
            cl = CL[idx[i]];
            if (cl <= 0)
                break;
            x_in = x[idx[i]];
            x_diff += x_in;
            x_sqared_sum += x_in * x_in;
            x_cum = x_sqared_sum + x_diff * x_diff;
            f_reduction[n - 1 - i] = cl * x_cum;
            // for cl == DBL_MAX, f_reduction would be 0 because (DBL_MAX * 0.0) == 0.0
        }

        k = -1;
        max_reduction = 0.;
        for (j = 0; j < n - 1 - i; ++j)
        {
            if (max_reduction < f_reduction[j])
            {
                max_reduction = f_reduction[j];
                k = j;
            }
        }

        x_diff = 0.0;
        for (i = n - 1; i >= n - 1 - k; --i) 
        {
            x_in = x[idx[i]];
            A[idx[i]] = 1;
            x_diff += x_in;
            x[idx[i]] = 0.0;
        }
        if (k != -1)
        {
            x[J] += x_diff;
            A[J] = 0;
        }
        /*old_val = val;
        val = QP_objective(H, b, x, tmp, n);
        if (val > old_val)
        {
            printf("%e > %e\n", val, old_val);
            throw " ";
        }*/
        
        compute_grad(H, b, x, grad, n);

        // update active set A and binding set B
        // the binding constraints are the active constraints whose Lagrange multiplier estimates have the correct sign
        info = -4;
        m = 0;
        for (i = 0; i < n; i++)
        {
            if (x[i] != 0)
            {
                //A[i] = 0;
                B[i] = 0;
                m++;
            }
            else
            {
                //A[i] = 1;
                if (grad[i] >= 0.)
                    B[i] = 1;
                else
                {
                    m++;
                    B[i] = -1;
                }
            }
            if (!(A_old[i] == A[i] || (A_old[i] == 0 && A[i] == -1) || (A_old[i] == -1 && A[i] == 0))) 
            {
                info = -3;  // indicate active set has changed
            }
        }
        if (info != -3)
            count++;
        else
            count = 0;
        memcpy(A_old, A, n * sizeof(int));
        
        cg_step = 0;
        if (count >= n1)
        {
            // projected gradient
            reduced_grad_projection2(grad, proj_grad, B, n);
            
            // reduced gradient (reduced onto the non-active set)
            reduced_grad_projection2(grad, rd_grad, A, n);
            
            if (count >= n1)
            {
                t = asa_dot(rd_grad, proj_grad, n) / _dnrm2(n, rd_grad) / _dnrm2(n, proj_grad);
                if (t >= angle1)  // angle between proj_grad and rd_grad is very small
                    cg_step = 1;
            }
        }


        // active set remains the same for n1 continuous times, which means
        // the optimal active set has probably been identified
        // so switch to constrained conjugate gradient method to explore current working set
        if (cg_step)
        {
            for (i = 0; i < neg_count; i++)
            {
                for (j = 0; j < n; j++)
                {
                    if ((A[j] != 1) && n_count[i][j] != 1)  // check if current active set is a subset of any set in 'n_count'
                        continue;
                    else
                        break;
                }
                if (j != n)
                    continue;
                else  // current active set is a subset of a set in 'n_count'
                {
                    count = 0;
                    goto skip_grad;  // no need to do cgg, because negative curvature would be encountered
                }
            }
            
            // since the x's in active set are fixed to 0, 
            // reduce the quadratic form to optimize the free variables
            j = 0;
            J = 0;
            for (i = 0; i < n; i++)
            {
                if (A[i] == 1)
                {
                    j += n;
                    continue;
                }
                for (k = 0; k < n; k++)
                {
                    if (A[k] == 1)
                    {
                        j += 1;
                        continue;
                    }
                    Hz[J++] = H[j++];
                }
            }
            
            m = n_non_active(A, n);  // # of free variables
            
            j = 0;
            for (i = 0; i < n; i++)
            {
                if (A[i] == 1)
                {
                    continue;
                }
                grad[j] = b[i];  // grad is temperory storage
                x_tmp[j] = x[i];
                j++;
            }

            // here grad and proj_grad act as temperory storage
            info = ccg(Hz, grad, x_tmp, tmp, tmp2, proj_grad, m);
            //printf("info: %d\n", info);
            if (info >= -1)
            {
                // nagetive curvature is detected (info == -1)
                // or optimum is not in this active set (info >= 0)
                // conjugate gradient method is not a good fit
                count = 0;

                // neg_count is the current # of negative c
                if (neg_count == 0)
                {
                    n_count[0] = new int[n];
                    memcpy(n_count[0], A, n * sizeof(int));
                    neg_count++;
                }
                else
                {
                    for (i = 0; i < neg_count; i++)
                    {
                        if (!same_aset(A, n_count[i], n))  // same active set not found
                            continue;
                        else
                            break;
                    }
                    if (i == neg_count)  // same active set not found
                    {

                        if (neg_count < 10)
                        {
                            n_count[neg_count] = new int[n];
                            memcpy(n_count[neg_count], A, n * sizeof(int));
                        }
                        else
                        {
                            n_count.resize(neg_count + 1);
                            n_count[neg_count] = new int[n];
                            memcpy(n_count[neg_count], A, n * sizeof(int));
                        }
                        neg_count++;
                    }
                }
            }

            if (info >= 0 || info == -2)
            {
                k = -2;
                j = 0;
                for (i = 0; i < n; i++)
                {
                    if (A[i] == 1)
                    {
                        continue;
                    }
                    // note that info is the index of the reduced form
                    if (j == info)
                        k = i;  // i is the correct index for the original form
                    x[i] = x_tmp[j++];
                }
                if (k != -2)
                {
                    A[k] = 1;
                    x[k] = 0.;
                    if (grad[k] >= 0.)
                        B[k] = 1;
                    else
                        B[k] = -1;
                }

                for (i = 0; i < n; i++)
                {
                    if (A[i] == -1 && x[i] > 0.)
                        A[i] = 0;
                }
            }
            /*old_val = val;
            val = QP_objective(H, b, x, tmp, n);
            if (val > old_val && info == -1 || val >= old_val && info != -1)
            {
                printf("%e > %e, %d\n", val, old_val, info);
                throw " ";
            }*/
            if (info == -2)
                goto finish;  // conjugate gradient finished
        }
        else 
            goto skip_grad;

        
        // Compute Active-Set Gradient Related Directions
        // We adopt the projected gradient direction
    
        compute_grad(H, b, x, grad, n);
    skip_grad:
        

        // Reduced gradient
        reduced_grad_projection2(grad, rd_grad, A, n);
        // projected gradient
        reduced_grad_projection2(grad, proj_grad, B, n);

        diff = _dnrm2(n, proj_grad);
        if (diff <= tol)
            goto finish;

        t = asa_dot(rd_grad, proj_grad, n) / _dnrm2(n, rd_grad) / _dnrm2(n, proj_grad);
        if (t > angle)  // use reduced gradient
        {
            memcpy(proj_grad, rd_grad, n * sizeof(double));
        }
        else  // use projected gradient, need to change active set
        {
            for (i = 0; i < n; i++)
            {
                if (rd_grad[i] == 0. && proj_grad[i] != 0.)
                {
                    A[i] = 0;
                    // B[i] = 0;
                }
            }
        }

        step_max = DBL_MAX;
        for (i = 0; i < n; i++)
        {
            if (proj_grad[i] == 0.)
                continue;
            dtmp = -x[i] / proj_grad[i];
            if (dtmp > 0 && dtmp < step_max)
            {
                idx_in = i;
                step_max = dtmp;
            }
        }

        c2 = _ddot(grad, proj_grad, n);
        if (c2 > -1e-11)
            goto finish;

        step = exact_line_search_QP(H, proj_grad, c2, step_max, tmp, n);

        if (step == step_max)  // this step hits the boundary
        {
            A[idx_in] = 1;
        }

        asa_daxpy(x, proj_grad, step, n);

        if (step == step_max)  // this step hits the boundary
        {
            x[idx_in] = 0.;
        }

        for (i = 0; i < n; i++)
        {
            if (A[i] == -1 && x[i] > 0.)
                A[i] = 0;
        }

        //old_val = val;
        //val = QP_objective(H, b, x, tmp, n);
        //while (val > old_val)
        //{
        //    printf("%e >= %e, %d, %e, step: %e, c2: %e\n", val, old_val, step == step_max, val - old_val, step, c2);
        //    step /= 2;
        //    asa_daxpy(x, proj_grad, -step, n);
        //    old_val = val;
        //    val = QP_objective(H, b, x, tmp, n);
        //    //throw " ";
        //}

    }
    finish:
    //printf("----------- %d\n\n", n_iter);
    delete[] grad;
    delete[] proj_grad;
    delete[] CL;
    delete[] f_reduction;
    delete[] idx;
    delete[] tmp;
    delete[] tmp2;
    delete[] x_tmp;
    delete[] B;
    delete[] A_old;
    delete[] Hz;
    delete[] rd_grad;

    for (i = 0; i < neg_count; i++)
    {
        delete[] n_count[i];
    }
}

// constrained conjugate gradient
// min_x 0.5 x^T * H * x - c^T * x
// s.t.  1^T * x = 1
// return i if x[i] < 0 during iteration
// return -1 if nagetive curvature is detected
// return -2 if conjugate gradient finish successfully
int ccg(double* H, double* c, double* x, double* r, double* g, double* p, int n)
{
    double a, a_max, b, rg_old, rg, tmp, tmp2, *x_inner, initial, final;
    int j, ret_idx, last_is_inner;
    std::vector<double> xx(n);
    x_inner = &xx[0];
    memcpy(x_inner, x, n * sizeof(double));
    last_is_inner = 1;
    ret_idx = -5;
    
    initial = QP_objective2(H, c, x, n);
    
    compute_grad(H, c, x, r, n);

    tmp = r[0];
    g[0] = 0;
    for (int i = 1; i < n; i++)
    {
        tmp2 = r[i] - tmp;
        g[0] -= tmp2;
        p[i] = -tmp2;
        g[i] = tmp2;
    }
    p[0] = -g[0];

    rg = _ddot(r, g, n);

    for (int i = 0; i < n + 2; i++)
    {
        // here g = H*p
        asa_matvec(g, H, p, n, n, 1);
        tmp = _ddot(g, p, n);
        
        if (tmp <= 0)
            return -1;

        a = rg / tmp;

        asa_daxpy(x, p, a, n);

        asa_daxpy(r, g, a, n);

        // record the last iteration of x that is feasible
        for (j = 0; j < n; j++)
        {
            if (x[j] < 0)
                break;
        }
        if (j == n)  // still within the feasible set
        {
            memcpy(x_inner, x, n * sizeof(double));
            ret_idx = -2;
            last_is_inner = 1;
        }
        else if (last_is_inner == 1)
        {
            // this iteration goes out of feasible set, so let the last x (x_inner) that was in the feasible set hit the boundary
            last_is_inner = 0;
            
            a_max = -a;
            if (a > 0)
            {
                for (j = 0; j < n; j++)
                {
                    if (p[j] >= 0)
                        continue;
                    tmp = x_inner[j] / p[j];

                    if (tmp > a_max)
                    {
                        a_max = tmp;
                        ret_idx = j;
                    }
                }
            }
            else if (a < 0)
            {
                for (j = 0; j < n; j++)
                {
                    if (p[j] <= 0)
                        continue;
                    tmp = x_inner[j] / p[j];

                    if (tmp < a_max)
                    {
                        a_max = tmp;
                        ret_idx = j;
                    }
                }
            }
            else
            {
                throw "unknown error in 'qp_simplex.cpp'!";
            }

            asa_daxpy(x_inner, p, -a_max, n);

            if (ret_idx < 0)
            {
                throw "unknown error in 'qp_simplex.cpp'!";
            }
        }

        tmp = r[0];
        g[0] = 0;
        for (j = 1; j < n; j++)
        {
            tmp2 = r[j] - tmp;
            g[0] -= tmp2;
            g[j] = tmp2;
        }

        rg_old = rg;
        rg = _ddot(r, g, n);
        
        if (rg == 0)
        {
            break;  // optimum found, conjugate gradient finish
        }

        b = rg / rg_old;

        for (j = 0; j < n; j++)
        {
            p[j] = b * p[j] - g[j];
        }
        
    }
    
    final = QP_objective2(H, c, x, n);
    if (final >= initial)
    {
        return -1;
    }

    if (last_is_inner == 0)
    {
        final = QP_objective2(H, c, x_inner, n);
        if (final >= initial)
            return -1;

        memcpy(x, x_inner, n * sizeof(double));
        return ret_idx;
    }

    // ccg succeed
    // but it is possible that this is the negative curvature case
    return ret_idx;
}

// project the gradient onto the hyper-plane
// min_x ||x - g||
// s.t.  1^T * x = 0
void grad_projection(double* g, double* x, int n)
{
    double sum = 0.;
    for (int i = 0; i < n; i++)
    {
        sum += g[i];
    }
    sum /= n;
    for (int i = 0; i < n; i++)
    {
        x[i] = g[i] - sum;
    }
}

// project the reduced gradient onto the hyper-plane
// min_x ||x - g||
// s.t.  1^T * x = 0
// A : reduced set to project onto
// ********************** Warning *******************************
// this function is wrong because it ignore the case A[i] == -1
void reduced_grad_projection(double* g, double* x, int* A, int n)
{
    double sum = 0.;
    int n_free = 0;
    for (int i = 0; i < n; i++)
    {
        if (A[i] == 0)  // free variables, note A[i] may be -1 when x[i] is free variable but at the boundary
        {
            sum += g[i];
            n_free++;
        }
            
    }
    sum /= n_free;
    for (int i = 0; i < n; i++)
    {
        if (A[i] == 0)  // free variables
            x[i] = sum - g[i];
        else
            x[i] = 0.;
    }
}

struct mypair
{
    double number;
    int index;
    bool strict;
    bool zero = false;

    void setval(double n, int i, bool s)
    {
        number = n;
        index = i;
        strict = s;
    }

};

bool mycompare(mypair l, mypair r)
{
    return (l.number > r.number);
}

void reduced_grad_projection2(double* g, double* pg, int* B, int n)
{
    double sum = 0., avg;
    int i, j, m, n_moving = 0;

    for (i = 0, j = 0; i < n; i++)
    {
        if (B[i] == 0 || B[i] == -1)
        {
            n_moving++;
        }
    }
    m = n_moving;

    std::vector<mypair> v(n_moving);

    for (i = 0, j = 0; i < n; i++)
    {
        // free variables or binding variables with wrong Lagrange multiplier sign
        if (B[i] == 0 || B[i] == -1)
        {
            sum += g[i];
            // for B[i] == -1, it is binding variables with wrong Lagrange multiplier sign
            // for A[i] == -1, it is free variables at the boundary of feasible set
            // in both case it need to restrict the projected gradient sign
            v[j++].setval(g[i], i, B[i] == -1);
        }
    }
    avg = sum / n_moving;

    // Sort y into descending order.
    sort(v.begin(), v.end(), mycompare);

    /*for (auto t : v)
    {
        std::cout << t.strict << " ";
    }
    std::cout << std::endl;
    for (auto t : v)
    {
        std::cout << t.number << " ";
    }
    printf("\n%f\n", avg);*/

    for (i = 0; i < n_moving; i++)
    {
        if (v[i].strict)
        {
            if (v[i].number > avg)
            {
                //printf("%d, %f\n", v[i].index, v[i].number);
                v[i].zero = true;
                sum -= v[i].number;
                avg = sum / (--m);
            }
            else
                break;
        }
    }

    for (i = 0; i < n_moving; i++)
    {
        if (v[i].zero)
            v[i].number = 0.;
        else
            v[i].number -= avg;

        pg[v[i].index] = -v[i].number;  // flip the sign to use it as a feasible descent direction
    }
    
    for (i = 0; i < n; i++)
    {
        if (B[i] == 1)
            pg[i] = 0.;
    }
}

void compute_grad(double* H, double* b, double* x, double* grad, int n)
{
#ifndef NOBLAS
    memcpy(grad, b, n * sizeof(double));
    cblas_dsymv(CblasColMajor, CblasLower, n, 1, H, n, x, 1, -1, grad, 1);
#else
    asa_matvec(grad, H, x, n, n, 1);
    for (int i = 0; i < n; i++)
    {
        grad[i] -= b[i];
    }
#endif
}

void compute_grad2(double* H, double* b, double* x, double* grad, int n)
{
#ifndef NOBLAS
    memcpy(grad, b, n * sizeof(double));
    cblas_dsymv(CblasColMajor, CblasLower, n, 1, H, n, x, 1, 1, grad, 1);
#else
    asa_matvec(grad, H, x, n, n, 1);
    for (int i = 0; i < n; i++)
    {
        grad[i] += b[i];
    }
#endif
}

double exact_line_search_QP(double *H, double *_d, 
    double c2, double step_max, double *tmp, int n)
{
    double c1, mid, step;
    _dsymv(H, _d, tmp, n);
    c1 = _ddot(_d, tmp, n) / 2;
    if(c1 == 0)
        if(c2 >= 0) 
            step = 0;
        else 
            step = step_max;
    else {
        mid = c2 / (-2 * c1);
        if(c1 > 0)
            if(mid < 0)
                step = 0;
            else if(mid >= 0 && mid <= step_max)
                step = mid;
            else
                step = step_max;
        else
            if(mid < (step_max / 2))
                step = step_max;
            else
                step = 0;
    }
    return step;
}

// sort indexes based on comparing values in v
// ascending order
void sort_indexes(const double *v, int *idx, int n) {
    std::sort(idx, idx+n, [&v](int i1, int i2) {return *(v+i1) < *(v+i2);});
}

int indexofSmallestElement(double *array, int size, double *min = NULL){
    int index = 0;
    for(int i = 1; i < size; i++)
        if(array[i] < array[index])
            index = i;
    if(min != NULL)
        *min = array[index];
    return index;
}

int Argmin(double *array, int size, int *A){
    int index = 0;
    for(int i = 1; i < size; i++)
        if(array[i] < array[index] && A[i] == 0)
            index = i;
    return index;
}

int Argmax(double *array, int size, int *A, double *x){
    int index = 0;
    for(int i = 1; i < size; i++)
        if(array[i] > array[index] && A[i] == 0 && x[i] > 0)
            index = i;
    return index;
}

double QP_objective(double* H, double* _b, double* _x, double* tmp, int n) {
    double c1, c2, val;
    _dsymv(H, _x, tmp, n);
    c1 = _ddot(_x, tmp, n) / 2;
    c2 = _ddot(_x, _b, n);
    val = c1 - c2;
    return val;
}

// compute 0.5 * x^T * H * x - x^T * b
double QP_objective2(double* H, double* _b, double* _x, int n) {
    double c1, c2, val;
    double* tmp = new double[n];
    _dsymv(H, _x, tmp, n);
    c1 = _ddot(_x, tmp, n) / 2;
    c2 = _ddot(_x, _b, n);
    val = c1 - c2;
    delete[]tmp;
    return val;
}

void print_array(double* a, int n, int acc) {
    std::cout << std::endl << "(";
    for (int i = 0; i < n; i++)
        printf("%.*f ", acc, a[i]);
    std::cout << ")" << std::endl;
}

void print_aset(int* a, int n) {
    std::cout << std::endl << "(";
    for (int i = 0; i < n; i++)
        printf("%d ", a[i]);
    std::cout << ")" << std::endl;
}

void print_aset_idx(int* a, int n) {
    std::cout << std::endl << "[";
    for (int i = 0; i < n; i++)
        if (a[i] == 1)
            printf("%d ", i);
    std::cout << "]" << std::endl;
}

int n_non_active(int* A, int n)
{
    int i, m = 0;
    for (i = 0; i < n; i++)
        if (A[i] != 1)
            m++;
    return m;
}

int check_feasibility(double* x, int n)
{
    double sum = 0.;
    for (int i=0; i<n; i++)
    {
        if (x[i] < 0)
        {
            printf("infeasible at initial point\n");
            printf("x[%d]: %e\n", i, x[i]);
            return 0;
            //throw "infeasible at initial point";
        }
        sum += x[i];
    }
    if (abs(sum - 1.) < 1e-8)
        return 1;
    else
    {
        printf("wrong sum: %.14f\n%e\n", sum, abs(sum - 1.));
        return 0;
    }
        
}

int check_activeset(double* x, int* A, int n)
{
    for (int i = 0; i < n; i++)
    {
        if ((x[i] == 0 && A[i] == 0) || (x[i] > 0 && A[i] != 0))
        {
            printf("wrong active set\n");
            printf("x[%d]: %e, A[%d]: %d\n", i, x[i], i, A[i]);
            return 0;
            //throw "infeasible at initial point";
        }
    }

    return 1;
}

bool same_aset(int* A_old, int* A, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (A_old[i] == A[i] || (A_old[i] == 0 && A[i] == -1) || (A_old[i] == -1 && A[i] == 0))
            continue;
        else
            return false;
    }
    return true;
}
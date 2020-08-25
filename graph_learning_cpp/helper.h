#ifndef _HELPER_
#define _HELPER_


void a_diag(double* A, const double* w, double* diag, int v, int n);

void a_whole(double* a_H, double* E, double* C, int v, int n);

void a_linear(double* A, double* S, const double* w, double* L, int v, int n);

void fill_zero(double* A, int n);

double vec_diff(double* v1, double* v2, int n);

void matrix_subtraction(double* v1, double* v2, double* output, int n);

// project A into the box of W such that 0 <= A <= W
void box_proj(double* A, double* W, int n);

void print_matrix(double* A, int n, int m, int acc = 3);

double objective(double* alpha, double* A, double* S, const double* w, double* C, double* E, double* H, int v, int n);

void DCA(double* A, double* b, double* x0, double* ub, int n, int m, double* y, int dca_iter);

#endif
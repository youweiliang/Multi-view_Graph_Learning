
void QPSimplex(double* H, double* b, double* x, int* A, int n, double tol = 1e-6);

int ccg(double* H, double* c, double* x, double* r, double* g, double* p, int n);

void compute_grad(double* H, double* b, double* x, double* grad, int n);

void compute_grad2(double* H, double* b, double* x, double* grad, int n);

double exact_line_search_QP(double* H, double* _d,
    double c2, double step_max, double* tmp, int n);

void sort_indexes(const double* v, int* idx, int n);

int indexofSmallestElement(double* array, int size, double* min);

int Argmin(double* array, int size, int* A);

int Argmax(double* array, int size, int* A, double* x);

int n_non_active(int* A, int n);

double QP_objective(double* H, double* _b, double* _x, double* tmp, int n);

double QP_objective2(double* H, double* _b, double* _x, int n);

void print_array(double* a, int n, int acc=6);

void print_aset(int* a, int n);

void print_aset_idx(int* a, int n);

void grad_projection(double* g, double* x, int n);

void reduced_grad_projection(double* g, double* x, int* A, int n);

void reduced_grad_projection2(double* g, double* x, int* B, int n);

int check_feasibility(double* x, int n);

int check_activeset(double* x, int* A, int n);

bool same_aset(int* A_old, int* A, int n);
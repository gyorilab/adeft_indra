import cython
from libc.math cimport pow as cpow
from libc.math cimport fabs, exp, log, log1p, pi, sqrt
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

from numpy.random import PCG64
from numpy.random cimport bitgen_t
from numpy.random.c_distributions cimport random_beta

from scipy.optimize.cython_optimize cimport brentq
from scipy.special.cython_special cimport loggamma


@cython.cdivision(True)
cdef double gamma_star(double a):
    """Scaled Gamma Function

    The Gamma function divided by Stirling's approximation
    """
    cdef double log_output
    if a == 0:
        return float('inf')
    elif a < 8:
        log_output = (loggamma(a) + a - 0.5*log(2*pi) -
                      (a - 0.5)*log(a))
        return exp(log_output)
    else:
        # Use Neme's approximation for sufficiently large input
        return cpow((1 + 1/(12*cpow(a, 2) - 1/10)), a)


@cython.cdivision(True)
cdef double D(double p, double q, double x):
    cdef double part1, part2, x0, sigma, tau
    if x == 0 or x == 1:
        return 0.0
    part1 = sqrt(p*q/(2*pi*(p+q))) * \
        gamma_star(p+q)/(gamma_star(p)*gamma_star(q))
    x0 = p/(p+q)
    sigma = (x - x0)/x0
    tau = (x0 - x)/(1 - x0)
    part2 = exp(p*(log1p(sigma) - sigma) + q*(log1p(tau) - tau))
    return part1 * part2


@cython.cdivision(True)
cdef inline double coefficient(int n, double p, double q, double x):
    cdef int m
    m = n // 2
    if n % 2 == 0:
        return m*(q-m)/((p+2*m-1)*(p+2*m)) * x
    else:
        return -(p+m)*(p+q+m)/((p+2*m)*(p+2*m+1)) * x


@cython.cdivision(True)
cdef double K(double p, double q, double x, double tol=1e-12):
    cdef int n
    cdef double delC, C, D
    delC = coefficient(1, p, q, x)
    C, D = 1 + delC, 1
    n = 2
    while fabs(delC) > tol:
        D = 1/(D*coefficient(n, p, q, x) + 1)
        delC *= (D - 1)
        C += delC
        n += 1
    return 1/C


@cython.cdivision(True)
cdef double betainc(float p, float q, float x):
    if x > p/(p+q):
        return 1 - betainc(q, p, 1-x)
    else:
        return D(p, q, x)/p * K(p, q, x)


@cython.cdivision(True)
cdef prevalence_cdf(double theta, int n, int t,
                    double sensitivity, double specificity):
    cdef double c1, c2, numerator, denominator
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    numerator = (betainc(t+1, n-t+1, c1 + c2*theta) - betainc(t+1, n-t+1, c1))
    denominator = betainc(t+1, n-t+1, c1 + c2) - betainc(t+1, n-t+1, c1)
    if denominator == 0:
        if t > n/2:
            return 1.0 if theta == 1.0 else 0.0
        elif t < n/2:
            return 1.0 if theta == 0.0 else 1.0
    return numerator/denominator


ctypedef struct interval_params:
    int n
    int t
    double sens
    double spec
    double alpha


@cython.cdivision(True)
cdef double f1(double theta, void *args):
    cdef interval_params *params = <interval_params *> args
    return prevalence_cdf(theta, params.n, params.t,
                          params.sens, params.spec) - params.alpha/2


@cython.cdivision(True)
cdef double f2(double theta, void *args):
    cdef interval_params *params = <interval_params *> args
    return prevalence_cdf(theta, params.n, params.t,
                          params.sens, params.spec) - 1 + params.alpha/2


cdef void prevalence_credible_interval_exact(int n, int t,
                                             double sens, double spec,
                                             double alpha, double *result):
    cdef double c1, c2, denominator, left, right
    cdef interval_params args
    args.n, args.t = n, t
    args.sens, args.spec, args.alpha = sens, spec, alpha
    c1, c2 = 1 - spec, sens + spec - 1
    denominator = betainc(t+1, n-t+1, c1 + c2) - betainc(t+1, n-t+1, c1)
    if denominator == 0:
        if t > n/2:
            result[0] = 1.0
            result[1] = 1.0
            return
        elif t < n/2:
            result[0] = 0.0
            result[1] = 0.0
            return
    left = brentq(f1, 0, 1, &args, 1e-3, 1e-3, 100, NULL)
    right = brentq(f2, 0, 1, &args, 1e-3, 1e-3, 100, NULL)
    result[0] = left
    result[1] = right


cdef void _prevalence_credible_interval(int n, int t,
                                        double sens_a, double sens_b,
                                        double spec_a, double spec_b,
                                        double alpha, int num_samples,
                                        double *result):
    cdef int i
    cdef bitgen_t *rng
    cdef double temp_result[2]
    cdef double sens, spec, left_acc, right_acc
    cdef double _sens_a, _sens_b, _spec_a, _spec_b
    cdef const char *capsule_name = "BitGenerator"
    cdef double *sens_array
    cdef double *spec_array

    sens_array = <double *> PyMem_Malloc(num_samples * sizeof(double))
    spec_array = <double *> PyMem_Malloc(num_samples * sizeof(double))
    
    x = PCG64()
    capsule = x.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    i = 0
    _sens_a, _sens_b, _spec_a, _spec_b = sens_a, sens_b, spec_a, spec_b
    with x.lock:
        for i in range(num_samples):
            sens_array[i] = random_beta(rng, _sens_a, _sens_b)
            spec_array[i] = random_beta(rng, _spec_a, _spec_b)
    i = 0
    for i in range(num_samples):
        prevalence_credible_interval_exact(n, t,
                                           sens_array[i],
                                           spec_array[i],
                                           alpha, temp_result)
        result[0] += temp_result[0]
        result[1] += temp_result[1]
    PyMem_Free(sens_array)
    PyMem_Free(spec_array)
    result[0] = result[0]/num_samples
    result[1] = result[1]/num_samples


def prevalence_credible_interval(n, t, sens_a, sens_b, spec_a, spec_b, alpha,
                                 num_samples=5000):
    cdef double result[2]
    cdef int _n, _t, _num_samples
    cdef double _sens_a, _sens_b, _spec_a, _spec_b, _alpha
    _n, _t, _num_samples, _alpha = n, t, num_samples, alpha
    _sens_a, _sens_b, _spec_a, _spec_b, = sens_a, sens_b, spec_a, spec_b
    _prevalence_credible_interval(_n, _t, _sens_a, _sens_b, _spec_a, _spec_b,
                                  _alpha, _num_samples, result)
    return (result[0], result[1])

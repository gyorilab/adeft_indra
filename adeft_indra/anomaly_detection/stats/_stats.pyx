import cython
import numpy as np
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

def py_betainc(a, b, x):
    return betainc(a, b, x)


ctypedef double (*prevalence_function)(double, int, int, double, double)


cdef double beta_sample(double theta, int n, int t,
                        double sens_a, double sens_b,
                        double spec_a, double spec_b,
                        prevalence_function func,
                        int num_samples):
    cdef int i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double result
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
    with x.lock, nogil:
        for i in range(num_samples):
            sens_array[i] = random_beta(rng, sens_a, sens_b)
            spec_array[i] = random_beta(rng, spec_a, spec_b)
    result = 0
    i = 0
    for i in range(num_samples):
        result += func(theta, n, t, sens_array[i], spec_array[i])
    PyMem_Free(sens_array)
    PyMem_Free(spec_array)
    return result/num_samples


cdef double prevalence_cdf_known_sens_spec(double theta, int n, int t,
                                           double sensitivity,
                                           double specificity):
    cdef double c1, c2, numerator, denominator
    c1, c2 = 1 - specificity, sensitivity + specificity - 1
    numerator = (betainc(t+1, n-t+1, c1 + c2*theta) - betainc(t+1, n-t+1, c1))
    denominator = betainc(t+1, n-t+1, c1 + c2) - betainc(t+1, n-t+1, c1)
    if denominator == 0:
        return theta
    return numerator/denominator


cdef double _prevalence_cdf(double theta, int n, int t, double sens_a,
                            double sens_b, double spec_a, double spec_b,
                            int num_samples):
    return beta_sample(theta, n, t, sens_a, sens_b, spec_a, spec_b,
                       prevalence_cdf, num_samples)


def prevalence_cdf_points(theta, n, t, sens_a, sens_b, spec_a, spec_b,
                          num_samples=10000, step_size=0.01):
    vals = np.fromiter((_prevalence_cdf(theta, n, t, sens_a, sens_b, spec_a, spec_b,
                                        num_samples) for theta in
                        np.arange(0, 1 + step_size, step_size)), dtype=float)
    return vals

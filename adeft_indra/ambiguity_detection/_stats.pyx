from libc.math cimport pow as cpow
from libc.math cimport fabs, exp, log, log1p, pi, sqrt

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


def prevalence_credible_interval_exact(n, t, sens, spec, alpha):
    cdef double c1, c2, denominator, left, right
    cdef interval_params args
    args.n, args.t = n, t
    args.sens, args.spec, args.alpha = sens, spec, alpha
    c1, c2 = 1 - spec, sens + spec - 1
    denominator = betainc(t+1, n-t+1, c1 + c2) - betainc(t+1, n-t+1, c1)
    if denominator == 0:
        if t > n/2:
            return (1.0, 1.0)
        elif t < n/2:
            return (0.0, 0.0)
    left = brentq(f1, 0, 1, &args, 1e-3, 1e-3, 100, NULL)
    right = brentq(f2, 0, 1, &args, 1e-3, 1e-3, 100, NULL)
    return (left, right)

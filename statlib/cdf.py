# statlib/cdf.py
# =============================================================================
#
# This file is part of SimElements.
# ----------------------------------
#
#  SimElements is a software package designed to be used for continuous and 
#  discrete event simulation. It requires Python 3.0 or later versions.
# 
#  Copyright (C) 2010  Nils A. Kjellbert 
#  E-mail: <info(at)ambinova(dot)se>
# 
#  SimElements is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published by 
#  the Free Software Foundation, either version 3 of the License, or 
#  (at your option) any later version. 
# 
#  SimElements is distributed in the hope that it will be useful, 
#  but WITHOUT ANY WARRANTY; without even the implied warranty of 
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
#  GNU General Public License for more details. 
# 
#  You should have received a copy of the GNU General Public License 
#  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
#
# ------------------------------------------------------------------------------
"""
Module with functions for the cdf of various probability distributions. 
"""
# ------------------------------------------------------------------------------

from math   import exp, sqrt, exp, atan, log, log10, sin, cos
from bisect import bisect

from statlib.pdf       import dpoisson, dnormal, dstable_sym
from numlib.specfunc   import lnfactorial, lngamma, beta, erfc1
from numlib.quadrature import qromberg
from numlib.talbot     import talbot
from numlib.miscnum    import fsign
from misclib.numbers   import is_nonneginteger, is_posinteger, kept_within
from machdep.machnum   import MACHEPS, TWOMACHEPS, FOURMACHEPS, MINFLOAT
from misclib.errwarn   import Error, warn
from misclib.mathconst import SQRT05, SQRT2, PI, PIHALF, PIINV

# ------------------------------------------------------------------------------

def cunifab(left, right, x):
    """
    The cdf of the uniform distribution with support on [left, right].
    """

    # Input check -----------------
    assert right > left, "support range must be positive in cunifab!"
    assert left <= x and x <= right, \
                    "variate must be within support range in cunifab!"
    # -----------------------------

    cdf  =  (x-left) / float(right-left)

    #cdf  = kept_within(0.0, cdf, 1.0)  # Not really needed
    
    return cdf

# end of cunifab

# ------------------------------------------------------------------------------

def ctriang(left, mode, right, x):
    """
    The cdf of the triangular distribution with support 
    on [left, right] and with mode 'mode'. 
    """

    # Input check -----------------------
    assert right > left, "support range must be positive in ctriang!"
    assert left <= mode and mode <= right, \
                    "mode must be within support range in ctriang!"
    assert left <=  x   and   x  <= right, \
                    "variate must be within support range in ctriang!"
    # -----------------------------------

    spant = right - left
    spanl = mode  - left
    spanr = right - mode

    if spanr == 0.0:
        cdf  =  (x-left)**2 / float((spant*spanl))

    elif spanl == 0.0:
        cdf  =  1.0 - (right-x)**2 / float((spant*spanr))

    elif x <= mode:
        cdf  =  (x-left)**2 / float((spant*spanl))

    else:
        cdf  =  1.0 - (right-x)**2 / float((spant*spanr))


    cdf  = kept_within(0.0, cdf, 1.0)
    
    return cdf

# end of ctriang

# ------------------------------------------------------------------------------

def ctri_unif_tri(a, b, c, d, x):
    """
    The cdf of the triangular-uniform-triangular distribution with 
    support on [a, d] and with break points in b and c.
              ------
    pdf:    /        \
           /           \
    ------              -------
    """

    # Input check -----------------------
    assert d > a, "support range must be positive in ctri_uinf_tri!"
    assert a <= b and b <= c and c <= d, \
         "break points must in order and within support range in ctri_unif_tri!"
    assert a <= x and x <= d, \
                  "variate must be within support range in ctri_unif_tri!"
    # -----------------------------------


    if c == b:
        cdf = ctriang(a, b, d)

    else:
        h  =  2.0 / (d+c-b-a)   # Height of entire pdf trapezoid
        if x < b:
            cdf   =  0.5 * h * (x-a)**2 / (b-a)
        elif x > c:
            ccdf  =  0.5 * h * (d-x)**2 / (d-c)
            cdf   =  1.0 - ccdf
        else:   # b <= x <= c
            cdf   =  h * (0.5*(b-a) + x - b)

    cdf  = kept_within(0.0, cdf, 1.0)
    
    return cdf

# end of ctri_unif_tri

# ------------------------------------------------------------------------------

def cbeta(a, b, x1, x2, x, betaab=False, tolf=FOURMACHEPS, itmax=128):
    """
    The cdf of the beta distribution:
    f = x**(a-1) * (1-x)**(b-1) / beta(a, b)
    a, b >= 0; 0 <= x <= 1
    F is the integral = the incomplete beta or the incomplete beta ratio 
    function depending on how the incomplete beta function is defined.

    NB It is possible to gain efficiency by providing the value of the complete 
    beta function beta(a, b) as a pre-computed input (may be computed using 
    numlib.specfunc.beta) instead of the default "False".

    a and/or b <= 1 is handled using a recurrence formula from Abramowitz & 
    Stegun (G is the gamma function, B is the beta function and I is the 
    incomplete beta):
    I(a, b, x)   =  G(a+b) / (G(a+1)*G(b)) * x^a * (1-x)^b  +  I(a+1, b, x)
    G(a+b) / (G(a+1)*G(b)) = 1/(a*B(a, b))  
    I(a, b, x)   =  1 - I(b, a, 1-x)
    I(b, a, 1-x) =  G(a+b) / (G(b+1)*G(a)) * x^a * (1-x)^b  +  I(b+1, a, 1-x)
    So then - if a < 1:
    I(a, b, x)  =  1/(a*B(a, b)) * x^a * (1-x)^b  +  I(a+1, b, x)
    or, if b < 1:
    I(a, b, x)  =  1  -  1/(b*B(a, b)) * x^a * (1-x)^b  -  I(b+1, a, 1-x)
    If both a and b < 1 then first
    I(a+1, b, x)  =  1 - 1/(b*B(a+1, b)) * x^(a+1) * (1-x)^b - I(b+1, a+1, 1-x)
    and then
    I(a, b, x)  =  1/(a*B(a, b)) * x^a * (1-x)^b  +  I(a+1, b, x)

    For the general interval must hold that x2 > x1  !!!!
    """

    assert a  >= 0.0, "both parameters must be non-negative in cbeta!"
    assert b  >= 0.0, "both parameters must be non-negative in cbeta!"
    assert x2 >= x1,  "support range must not be negative in cbeta!"
    assert x1 <= x and x <= x2, \
            "variate must be within support range in cbeta!"

    if a == 1.0 and b == 1.0: return cunifab(x1, x2, x)

    y  =  (x-x1) / (x2-x1)
    cy =  1.0 - y

    if y == 0.0: return 0.0
    if y == 1.0: return 1.0
    
    # -------------------------------------------------------------------------
    def _betainc(_a, _b, _y):
        comp = (_a+1.0) / (_a+_b+2.0)
        _cy  = 1.0 - _y
        _poverbeta = pow(_y, _a) * pow(_cy, _b) / beta(_a, _b)
        if _y < comp:
            bi  =  _poverbeta * _betaicf(_a, _b, _y, tolf, itmax) / _a
        else:
            bi  =  1.0 -  _poverbeta * _betaicf(_b, _a, _cy, tolf, itmax) / _b
        return bi
    # -------------------------------------------------------------------------

    if betaab: betaf = betaab
    else:      betaf = beta(a, b)

    if a <= 1.0 and b <= 1.0:
        ap1  =  a + 1.0
        # beta(a, b)    = gamma(a) * gamma(b) / gamma(a+b)
        # beta(a+1, b)  = gamma(a+1) * gamma(b) / gamma(a+b+1)
        # gamma(a+1)    = a * gamma(a)
        # gamma(a+b+1)  = (a+b) * gamma(a+b)
        # beta(a+1, b)  = a * gamma(a) * gamma(b) / ((a+b)*gamma(a+b)) = 
        #               = a * beta(a, b) / (a+b)
        betaf1 = a * beta(a, b) / (a+b)  # = beta(ap1, b)
        cdf    = 1.0 - (1.0/(b*betaf1))*pow(y, ap1)*pow(cy, b) - \
                                                 _betainc(b+1.0, ap1, cy)
        poverbeta = pow(y, a) * pow(cy, b) / betaf
        cdf +=  poverbeta / a

    elif a <= 1.0:
        poverbeta = pow(y, a) * pow(cy, b) / beta(a, b)
        cdf  =  poverbeta/a + _betainc(a+1.0, b, y)

    elif b <= 1.0:
        poverbeta = pow(y, a) * pow(cy, b) / beta(a, b)
        cdf  =  1.0 - poverbeta/b - _betainc(b+1.0, a, cy)

    else:
        cdf  =  _betainc(a, b, y)

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cbeta

# ----------------------------------------------------------

def _betaicf(a, b, y, tolf, itmax):
    # Auxiliary function with continued fractions expansion 
    # for cbeta (cf Abramowitz & Stegun):

    apb = a + b
    ap1 = a + 1.0
    am1 = a - 1.0
    c   = 1.0
    d   = 1.0 - y*apb/ap1
    if abs(d) < MINFLOAT: d = MINFLOAT
    d   = 1.0 / d
    h   = d

    converged = False
    itmaxp1   = itmax + 1
    for k in range(1, itmaxp1):
        fk   = float(k)
        tfk  = fk + fk
        aa   = fk * (b-fk) * y / ((am1+tfk)*(a+tfk))
        d    = 1.0 + aa*d
        if abs(d) < MINFLOAT: d = MINFLOAT
        c    = 1.0 + aa/c
        if abs(c) < MINFLOAT: c = MINFLOAT
        d    = 1.0 / d
        h   *= d * c
        aa   = - (a+fk) * (apb+fk) * y / ((a+tfk)*(ap1+tfk))
        d    = 1.0 + aa*d
        if abs(d) < MINFLOAT: d = MINFLOAT
        c    = 1.0 + aa/c
        if abs(c) < MINFLOAT: c = MINFLOAT
        d    = 1.0 / d
        dl   = d * c
        h   *= dl
        if abs(dl-1.0) < tolf:
            converged = True
            break

    if not converged:
        warn("cbeta has not converged for itmax = " + \
                     str(itmax) + " and tolf = " + str(tolf))

    return h

# end of _betaicf

# ------------------------------------------------------------------------------

def ckumaraswamy(a, b, x1, x2, x):
    """
    The cdf of the Kumaraswamy distribution:
    f = a*b*x**(a-1) * (1-x**a)**(b-1)
    F = 1 - (1-x**a)**b
    a, b >= 0; 0 <= x <= 1
    
    The Kumaraswamy distribution is similar to the beta distribution !!!
    
    x2 > x1  !!!!
    """

    assert a  >= 0.0, "both parameters must be non-negative in ckumaraswamy!"
    assert b  >= 0.0, "both parameters must be non-negative in ckumaraswamy!"
    assert x2 >  x1, "support range must be positive in ckumaraswamy!"
    assert x1 <= x and x <= x2, \
               "variate must be within support range in ckumaraswamy!"

    y  =  (x-x1) / (x2-x1)

    cdf  =  1.0 - (1.0-y**a)**b

    cdf  =  kept_within(0.0, cdf, 1.0)

    return cdf

# end of ckumaraswamy

# ------------------------------------------------------------------------------

def csinus(a, b, x):
    """
    The cdf of the "sinus distribution" with support on [left, right]. 
    """

    assert right > left,     "support range must be a positive float in csinus!"
    assert left <= x <= right, "variate must be within support range in csinus!"

    const = PI / (right-left)
    cdf   = 0.5 * (1.0 - cos(const*(x-left)))

    cdf   = kept_within(0.0, cdf, 1.0)

    return cdf

# end of csinus

# ------------------------------------------------------------------------------
 
def cgeometric(phi, k):
    """
    The geometric distribution with p(K=k) = phi * (1-phi)**(k-1)  and 
    P(K>=k) = sum phi * (1-phi)**k = 1 - q**k where q = 1 - phi and  
    0 < phi <= 1 is the success frequency or "Bernoulli probability" and 
    K >= 1 is the number of  trials to the first success in a series of 
    Bernoulli trials. It is easy to prove that P(k) = 1 - (1-phi)**k: 
    let q = 1 - phi. p(k) = (1-q) * q**(k-1) = q**(k-1) - q**k. 
    Then P(1) = p(1) = 1 - q. P(2) = p(1) + p(2) = 1 - q + q - q**2 = 1 - q**2. 
    Induction can be used to show that P(k) = 1 - q**k = 1 - (1-phi)**k 
    """

    assert 0.0 < phi and phi <= 1.0, \
                     "success frequency must be in (0.0, 1.0] in cgeometric!"
    assert is_posinteger(k), \
                 "number of trials must be a positive integer in cgeometric!"

    cdf  =  1.0 - (1.0-phi)**k

    return cdf

# end of cgeometric

# ------------------------------------------------------------------------------

def cpoisson(lam, tspan, n):
    """
    The Poisson distribution: p(N=n) = exp(-lam*tspan) * (lam*tspan)**n / n!
    n = 0, 1,...., inf
    """

    # Input check -----------
    assert  lam  >= 0.0, "Poisson rate must not be negative in cpoisson!"
    assert tspan >= 0.0, "time span must not be negative in cpoisson!"
    assert is_nonneginteger(n), \
                "variate must be a non-negative integer in cpoisson!"
    # -----------------------

    cdf = 1.0 - cgamma(n+1.0, lam, tspan)

    return cdf

# end of cpoisson

# ------------------------------------------------------------------------------

def cexpo(mean, x):
    """
    cdf for the exponential distribution with mean = 1/lambda (mean >=0.0)
    """

    # Input check ----
    assert mean >  0.0, "mean must be positive in cexpo!"
    assert x    >= 0.0, "variate must not be negative in cexpo!"
    # ----------------

    cdf  =  1.0 - exp(-x/mean)

    cdf  = kept_within(0.0, cdf, 1.0)
    
    return cdf

# end of cexpo

# ------------------------------------------------------------------------------

def chyperexpo(means, pcumul, x):
    """
    The hyperexponential distribution f = sumk pk * exp(x/mk) / mk, 
    F = sumk pk * (1-exp(x/mk))
    
    NB Input to the function is the list of CUMULATIVE PROBABILITIES ! 
    """

    lm = len(means)
    lp = len(pcumul)

    # Input check -----------------
    assert  x  >= 0.0, "variate must not be negative in chyperexpo!"
    if x == 0.0: return 0.0
    assert lp == lm, \
        "number of means must be equal to the number of pcumuls in chyperexpo!"
    errortextm = "all means must be positive floats in chyperexpo!"
    errortextp = "pcumul list is not in order in chyperexpo!"
    assert means[0] >= 0.0,                      errortextm
    assert pcumul[-1]  == 1.0,                   errortextp
    if lm == 1: return cexpo(means[0], x)
    assert 0.0 < pcumul[0] and pcumul[0] <= 1.0, errortextp
    # -----------------------------

    summ    = pcumul[0] * (1.0-exp(-x/means[0]))
    nvalues = lp
    for k in range (1, nvalues):
        assert means[k]  >  0.0, errortextm
        pdiff = pcumul[k] - pcumul[k-1]
        assert pdiff >= 0.0, errortextp
        summ += pdiff * (1.0-exp(-x/means[k]))
    cdf = summ

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of chyperexpo

# ------------------------------------------------------------------------------

def cNexpo(means, x):
    """
    cdf of a distribution of a sum of exponentially distributed 
    random variables. 
    
    NB Means are allowed to be equal (but the function is slow)!!!
    """

    assert x >= 0.0, "variate must not be negative in cNexpo!"

    if x == 0.0: return 0.0

    number = len(means)

    if number == 1: return cexpo(means[0], x)

    lam  = []
    for k in range(0, number):
        assert means[k] > 0.0, "All means must be positive floats in cNexpo!"
        lam.append(1.0/means[k])

    # ----------------------------
    def ftilde(z):
        zprod = complex(1.0)
        for k in range(0, number):
            zprod  =  zprod * complex(lam[k]) / (z+lam[k])
        zprod = zprod / z
        return zprod
    #-----------------------------

    sigma =  TWOMACHEPS * min(lam)
    cdf   =  talbot(ftilde, x, sigma)

    cdf   =  kept_within(0.0, cdf, 1.0)

    return cdf

# end of cNexpo

# ------------------------------------------------------------------------------

def cNexpo2(means, x):
    """
    cdf of a distribution of a sum of exponentially distributed 
    random variables. 
    
    NB No two means are allowed to be equal - if equal means are 
    desired, use cNexpo instead (slower, though)!!!
    """

    assert x >= 0.0, "variate must not be negative in cNexpo2"

    if x == 0.0: return 0.0

    number = len(means)

    if number == 1: return cexpo(means[0], x)

    lam     = []
    exps    = []
    product = 1.0
    for k in range(0, number):
        assert means[k] > 0.0, "all means must be positive floats in cNexpo2!"
        lam.append(1.0/means[k])
        exps.append(exp(-lam[k]*x))
        product = product*lam[k]

    divisor = []
    for k in range(0, number):
        divisor.append(lam[k])
        for j in range(0, number):
            if j != k: divisor[k] = divisor[k]*(lam[j]-lam[k])

    ccdf = 0.0
    try:
        for k in range(0, number):
            ccdf += exps[k] / divisor[k]
        cdf  = 1.0 - product*ccdf
    except (ZeroDivisionError, OverflowError):
        raise OverflowError("means too close in cNexpo2!")

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cNexpo2

# ------------------------------------------------------------------------------

def cexpo_gen(a, b, c, x):
    """
    The generalized continuous exponential distribution (x in R):
    x <= c: f  =  [a*b/(a+b)] * exp(+a*[x-c])
            F  =   [b/(a+b)]  * exp(+a*[x-c])
    x >= c: f  =  [a*b/(a+b)] * exp(-b*[x-c])
            F  =  1 - [a/(a+b)]*exp(-b*[x-c])
    a > 0, b > 0
    
    NB The symmetrical double-sided exponential sits in claplace!
    """

    assert a > 0.0, "'a' parameter must be positive in cexpo_gen!"
    assert b > 0.0, "'b' parameter must be positive in cexpo_gen!"

    if x <= c:  cdf  =  (b/(a+b)) * exp(a*(x-c))
    else:       cdf  =  1.0 - (a/(a+b))*exp(-b*(x-c))

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cexpo_gen

# ------------------------------------------------------------------------------

def cemp_exp(values, npexp, x, ordered=False, check=False):
    """
    The mixed expirical/exponential distribution from Bratley, Fox and Schrage.
    A polygon (piecewise linearly interpolated cdf with equal probability for 
    each interval between the ) is used together with a (shifted) exponential 
    for the tail. The distribution is designed so as to preserve the mean of 
    the input sample.
    
    The input is a tuple/list of observed points and an integer (npexp) 
    corresponding to the number of (the largest) points that will be used 
    to formulate the exponential tail (the default value of npexp will raise 
    an assertion error so something >= 0 must be prescribed).
    
    NB it is assumed that x is in [0.0, inf) !!!!!!!!!!!!
    
    The function may also be used for a piecewise linear cdf without the 
    exponential tail (setting npexp = 0) - corrections are made to maintain 
    the mean in this case as well!!! 
    """

    nvalues = len(values)
    if check:
        assert is_nonneginteger(npexp), \
            "No. of points for exp tail in cemp_exp must be a non-neg integer!"
        assert npexp <= nvalues, \
                "Number of points for exponential tail in cemp_exp too large!"
        for v in values:
            assert value >= 0.0, \
                      "All inputs must be non-negative in cemp_exp!"

    if npexp == nvalues:
        mean = sum(values)/float(nvalues)
        return cexpo(mean, x)

    vcopy = list(values)
    if not ordered:
        valueskm1 = values[0]
        for k in range(1, nvalues):
            valuesk = values[k]
            if valuesk >= valueskm1:
                valueskm1 = valuesk
            else:
                vcopy.sort()
                break

    if vcopy[0] != 0.0:
        vcopy.insert(0, 0.0)
        nindex = nvalues
    else:
        nindex = nvalues - 1

    breaki = nindex - npexp  # The last index of the piecewise linear part
    vcopyb = vcopy[breaki]
    if x > vcopyb:  # Compute the mean of the shifted exponential
        theta  = 0.0
        k0     = breaki + 1
        nip1   = nindex + 1
        for k in range(k0, nip1):
            theta += vcopy[k]
        theta += vcopyb*(0.5-npexp)
        theta /= npexp
        cdf    = 1.0 - npexp * exp(-(x-vcopyb)/theta) / nindex

    else:
        # Find the right interval using a binary search
        left   = bisect(vcopy, x) - 1
        vcopyl = vcopy[left]
        try:
            cdf = left + (x-vcopyl)/(vcopy[left+1]-vcopyl)
        except ZeroDivisionError:
            cdf = left
        cdf /= nindex

    cdf = kept_within(0.0, cdf, 1.0)
    return cdf

# end of cemp_exp

# ------------------------------------------------------------------------------

def cerlang(nshape, phasemean, x):
    """
    The cdf of the Erlang distribution.
    Represents the sum of nshape exponentially distributed random variables, 
    all having "phasemean" as mean
    """

    if nshape == 1:
        cdf = cexpo(phasemean, x)

    else:
        assert is_posinteger(nshape), \
                  "shape parameter must be a positive integer in cerlang!"
        assert  phasemean  >= 0.0, \
                           "phase mean must not be negative in cerlang!"
        assert      x      >= 0.0, \
                           "variate must not be negative in cerlang!"
        y    =  x / float(phasemean)
        cdf  = 1.0
        term = 1.0
        cdf  = term
        for k in range(1, nshape):
            term = term * y / k
            cdf  = cdf + term
        
        cdf = 1.0 - exp(-y)*cdf

        cdf = kept_within(0.0, cdf, 1.0)
    
    return cdf

# end of cerlang

# ------------------------------------------------------------------------------

def cerlang_gen(nshapes, pcumul, phasemean, x):
    """
    The generalized Erlang distribution - the Erlang equivalent of the hyperexpo
    distribution f = sumk pk * ferlang(m, nk), F = sumk pk * Ferlang(m, nk), the
    same mean for all phases.
    
    NB Input to the function is the list of CUMULATIVE PROBABILITIES ! 
    """

    ln = len(nshapes)
    lp = len(pcumul)

    # Input check -----------------
    assert  x  >= 0.0, "variate must not be negative in cerlang_gen!"
    assert lp == ln, \
      "number of shapes must be equal to the number of pcumuls in cerlang_gen!"
    if ln == 1: return derlang(nshapes[0], phasemean, x)
    errortextn = "all nshapes must be positive integers i cerlang_gen!"
    errortextm = "the mean must be a positive float in cerlang_gen!"
    errortextp = "pcumul list is not in order in cerlang_gen!"
    for n in nshapes: assert is_posinteger(n),   errortextn
    assert phasemean > 0.0,                      errortextm
    assert pcumul[-1]  == 1.0,                   errortextp
    assert 0.0 < pcumul[0] and pcumul[0] <= 1.0, errortextp
    # -----------------------------

    summ    = pcumul[0] * cerlang(nshapes[0], phasemean, x)
    nvalues = lp
    for k in range (1, nvalues):
        pdiff = pcumul[k] - pcumul[k-1]
        assert pdiff >= 0.0, errortextp
        summ += pdiff * cerlang(nshapes[k], phasemean, x)
    cdf = summ

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cerlang_gen

# ------------------------------------------------------------------------------

def ccoxian(means, probs, x):
    """
    The Coxian phased distribution, which is based on the exponential.
    probs is a list of probabilities for GOING ON TO THE NEXT PHASE rather 
    than reaching the absorbing state prematurely. The number of means must 
    (of course) be one more than the number of probabilities! 
    
    NB means are allowed to be equal (but the function is slow). 
    """

    lm = len(means)
    try:
        lp     = len(probs)
        probsl = probs
    except TypeError:   #  probs is not provided as a list
        probsl = [probs]
        lp     = len(probsl)

    assert lm == lp + 1, \
                         "lengths of input lists are not matched in ccoxian!"

    if lm == 2:
        assert means[0] >= 0.0 and means[1] >= 0.0, \
                                 "all means must be non-negative in ccoxian!"
        assert 0.0 <= probsl[0] and probsl[0] <= 1.0, \
                       "probabilities must be within 0.0 and 1.0 in ccoxian!"
        cdf = (1.0-probsl[0])*cexpo(means[0], x) + probsl[0]*cNexpo(means, x)

    else:
        sub      = lp*[1.0]
        for k in range(1, lp): sub[k] = sub[k-1]*probsl[k-1]

        freq     = lm*[0.0]
        freq[0]  = 1.0 - probsl[0]
        freq[-1] = sub[-1]*probsl[-1]
        for k in range(1, lp): freq[k] = sub[k]*(1-probsl[k])

        summ     = freq[0] * cexpo(means[0], x)
        for k in range(1, lm):
            summ += freq[k] * cNexpo(means[0:(k+1)], x)
        cdf = summ


    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of ccoxian

# ------------------------------------------------------------------------------

def ccoxian2(means, probs, x):
    """
    The Coxian phased distribution, which is based on the exponential.
    probs is a list of probabilities for GOING ON TO THE NEXT PHASE rather 
    than reaching the absorbing state prematurely. The number of means must 
    (of course) be one more than the number of probabilities! 
    
    NB No two means[k] must be equal - if equal means are desired, use 
    ccoxian instead (slower, however). 
    """

    lm = len(means)
    try:
        lp     = len(probs)
        probsl = probs
    except TypeError:   #  probs is not provided as a list
        probsl = [probs]
        lp     = len(probsl)

    assert lm == lp + 1, \
                       "lengths of input lists are not matched in ccoxian2!"

    if lm == 2:
        assert means[0] >= 0.0 and means[1] >= 0.0, \
                               "all means must be non-negative in ccoxian2!"
        assert 0.0 <= probsl[0] and probsl[0] <= 1.0, \
                     "probabilities must be within 0.0 and 1.0 in ccoxian2!"
        cdf = (1.0-probsl[0])*cexpo(means[0], x) + probsl[0]*cNexpo2(means, x)

    else:
        sub      = lp*[1.0]
        for k in range(1, lp): sub[k] = sub[k-1]*probsl[k-1]

        freq     = lm*[0.0]
        freq[0]  = 1.0 - probsl[0]
        freq[-1] = sub[-1]*probsl[-1]
        for k in range(1, lp): freq[k] = sub[k]*(1-probsl[k])

        summ     = freq[0] * cexpo(means[0], x)
        for k in range(1, lm):
            summ += freq[k] * cNexpo2(means[0:(k+1)], x)
        cdf = summ


    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of ccoxian2

# ------------------------------------------------------------------------------

def ckodlin(gam, eta, x):
    """
    The Kodlin distribution, aka the linear hazard rate distribution:
    f = (gam + eta*x) * exp{-[gam*x + (1/2)*eta*x**2]},
    F = 1 - exp{-[gam*x + (1/2)*eta*x**2]};  x, gam, eta >= 0
    """

    assert gam >= 0.0, "'gam' parameter must not be negative in ckodlin!"
    assert eta >= 0.0, "'eta' parameter must not be negative in ckodlin!"
    assert  x  >= 0.0, "variate must not be negative i ckodlin!"

    cdf  =  1.0 - exp(-(gam*x + 0.5*eta*x**2))

    cdf  =  kept_within(0.0, cdf, 1.0)

    return cdf

# end of ckodlin

# ------------------------------------------------------------------------------

def claplace(loc, scale, x):
    """
    The Laplace distribution
    f = ((1/2)/s))*exp(-abs([x-l]/s))
    F = (1/2)*exp([x-l]/s)  {x <= 0},  F = 1 - (1/2)*exp(-[x-l]/s)    {x >= 0}
    s > 0
    """

    assert scale > 0.0, "scale must be a positive float in claplace!"

    if x <= 0.0: cdf =       0.5*exp((x-loc)/float(scale))
    else:        cdf = 1.0 - 0.5*exp((loc-x)/float(scale))

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of claplace

# ------------------------------------------------------------------------------

def cexppower(loc, scale, alpha, x, lngam1oalpha=False, \
                                    tolf=FOURMACHEPS, itmax=128):
    """
    The exponential power distribution 
    f  =  (a/s) * exp(-abs([x-l]/s)**a) / [2*gamma(1/a)]
    F  =  1/2 * [1 + sgn(x-l) * Fgamma(1/a, abs([x-l]/s)**a)],   x in R
    s, a > 0
    where Fgamma is the gamma distribution cdf.

    NB It is possible to gain efficiency by providing the value of the 
    natural logarithm of the complete gamma function ln(gamma(1.0/alpha)) 
    as a pre-computed input (may be computed using numlib.specfunc.lngamma) 
    instead of the default 'False'.

    tolf and itmax are the numerical control parameters of cgamma.
    """

    assert scale > 0.0, \
               "scale parameter must be a positive float in cexppower!"
    assert alpha > 0.0, \
            "shape parameter alpha must be a positive float in cexppower!"

    if alpha == 1.0: return claplace(loc, scale, x)

    ainv = 1.0/alpha
    xml  = x - loc

    if not lngam1oalpha: lng1oa = lngamma(ainv)
    else:                lng1oa = lngam1oalpha
    cg   = cgamma(ainv, 1.0, abs(xml/scale)**alpha, lng1oa, tolf, itmax)
    cdf  = 0.5 * (fsign(xml)*cg + 1.0)

    cdf  = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cexppower

# ------------------------------------------------------------------------------

def cgamma(alpha, lam, x, lngamalpha=False, tolf=FOURMACHEPS, itmax=128):
    """
    The gamma distrib. f = lam * exp(-lam*x) * (lam*x)**(alpha-1) / gamma(alpha)
    F is the integral = the incomplete gamma or the incomplete gamma / complete 
    gamma depending on how the incomplete gamma function is defined.
    x, lam, alpha >= 0
    tolf  =  allowed fractional error in computation of the incomplete function
    itmax =  maximum number of iterations to obtain accuracy 

    NB It is possible to gain efficiency by providing the value of the 
    natural logarithm of the complete gamma function ln(gamma(alpha)) 
    as a pre-computed input (may be computed using numlib.specfunc.lngamma) 
    instead of the default 'False'.
    """

    assert alpha >= 0.0, "alpha must not be negative in cgamma!"
    assert  lam  >= 0.0, "lambda must not be negative i cgamma!"
    assert   x   >= 0.0, "variate must not be negative in cgamma!"
    assert tolf  >= 0.0, "tolerance must not be negative in cgamma!"
    assert is_posinteger(itmax), \
           "maximum number of iterations must be a positive integer in cgamma!"

    if alpha == 1.0: return cexpo(1.0/lam, x)

    lamx  = lam * x
    if lamx == 0.0: return 0.0
    if lngamalpha: lnga = lngamalpha
    else:          lnga = lngamma(alpha)

    # -------------------------------------------------------------------------
    def _gamser():
        # A series expansion is used for lamx < alpha + 1.0
        # (cf. Abramowitz & Stegun)
        apn  = alpha
        summ = 1.0 / apn
        dela = summ
        converged = False
        for k in range(0, itmax):
            apn  += 1.0
            dela  = dela * lamx / apn
            summ += dela
            if abs(dela) < abs(summ)*tolf:
                converged = True
                return  summ * exp(-lamx+alpha*log(lamx)-lnga), converged
        return  summ * exp(-lamx+alpha*log(lamx)-lnga), converged
    # -------------------------------------------------------------------------
    def _gamcf():
        # A continued fraction expansion is used for 
        # lamx >= alpha + 1.0 (cf. Abramowitz & Stegun):
        gold = 0.0
        a0   = 1.0
        a1   = lamx
        b0   = 0.0
        b1   = 1.0
        fac  = 1.0
        converged = False
        for k in range(0, itmax):
            ak  = float(k+1)
            aka = ak - alpha
            a0  = (a1+a0*aka) * fac
            b0  = (b1+b0*aka) * fac
            akf = ak * fac
            a1  = lamx*a0 + akf*a1
            b1  = lamx*b0 + akf*b1
            if a1 != 0.0:
                fac = 1.0 / a1
                g   = b1 * fac
                if abs(g-gold) < abs(g)*tolf:
                    converged = True
                    return  1.0 - exp(-lamx+alpha*log(lamx)-lnga) * g, converged
                gold = g
        return  1.0 - exp(-lamx+alpha*log(lamx)-lnga) * g, converged
    # -------------------------------------------------------------------------

    if lamx < alpha + 1.0:
        cdf, converged = _gamser()
    else:
        cdf, converged = _gamcf()

    if not converged:
        warn("cgamma has not converged for itmax = " + \
                     str(itmax) + " and tolf = " + str(tolf))

    cdf   = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cgamma

# ------------------------------------------------------------------------------

def cpareto(lam, xm, x):
    """
    The cdf of the Pareto distribution:
    f = lam * xm**lam / x**(lam+1)
    F = 1 - (xm/x)**lam
    x in [xm, inf)
    lam > 0
    For lam < 1 all moments are infinite
    For lam < 2 all moments are infinite except for the mean
    """

    assert lam >= 0.0, "lambda must not be negative in cpareto!"
    assert xm  >= 0.0, "lower limit must not be negative in cpareto!"
    assert x   >= xm, "variate must not be smaller than lower limit in cpareto!"

    cdf = 1.0 - pow(xm/x, lam)

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cpareto

# ------------------------------------------------------------------------------

def cpareto_zero(lam, xm, x):
    """
    The cdf of the Pareto distribution with the support shifted to [0, inf) :
    f = lam * xm**lam / (x+xm)**(lam+1)
    F = 1 - [xm/(x+xm)]**lam
    x in [0, inf)
    lam > 0
    For lam < 1 all moments are infinite
    For lam < 2 all moments are infinite except for the mean
    """

    assert lam >= 0.0, "lambda must not be negative in cpareto_zero!"
    assert xm  >= 0.0, "'xm' must not be negative in cpareto_zero!"
    assert x   >= 0.0, "variate must not be negative in cpareto_zero!"

    cdf = 1.0 - pow(xm/(x+xm), lam)

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cpareto_zero

# ------------------------------------------------------------------------------

def crayleigh(sigma, x):
    """
    The cdf of the Rayleigh distribution:
    f = (x/s**2) * exp[-x**2/(2*s**2)]
    F = 1 - exp[-x**2/(2*s**2)]
    x >= 0
    """

    assert sigma != 0.0, "sigma must not be 0 in crayleigh!"
    assert x >= 0.0, "variate must not be negative in crayleigh!"

    a   =  x / float(sigma)
    cdf =  1.0 - exp(-0.5*a**2)

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of crayleigh

# ------------------------------------------------------------------------------

def cweibull(c, scale, x):
    """
    The cdf of the Weibull distribution:
    f = exp[-(x/s)**(c-1)] / s
    F = 1 - exp[-(x/s)**c]
    x >= 0, s > 0, c >= 1 
    """

    if c == 1.0:
        cdf = cexpo(prob, scale, x)

    else:
        assert   c   >= 1.0, "shape parameter 'c' must be >= 1.0 in cweibull!"
        assert scale >  0.0, "scale must be positive in cweibull!"
        assert   x   >= 0.0, "variate must not be negative in cweibull!"

        cdf = 1.0 - exp(-(x/float(scale))**c)

        cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cweibull

# ------------------------------------------------------------------------------

def cextreme_I(type, mu, scale, x):
    """
    Extreme value distribution type I (aka the Gumbel distribution or 
    Gumbel distribution type I):
    F = exp{-exp[-(x-mu)/scale]}       (max variant)
    f = exp[-(x-mu)/scale] * exp{-exp[-(x-mu)/scale]} / scale
    F = 1 - exp{-exp[+(x-mu)/scale]}   (min variant)
    f = exp[+(x-mu)/scale] * exp{-exp[+(x-mu)/scale]} / scale

    type must be 'max' or 'min'
    scale must be > 0.0
    """

    assert scale > 0.0, "scale must be positive in cextreme_I!"

    if   type == 'max':  cdf = exp(-exp(-(x-mu)/float(scale)))
    elif type == 'min':  cdf = 1.0 - exp(-exp((x-mu)/float(scale)))
    else:
        raise Error("type must be either 'max' or 'min' in cextreme_I!")

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cextreme_I

# ------------------------------------------------------------------------------

def cextreme_gen(type, shape, mu, scale, x):
    """
    Generalized extreme value distribution:

    F = exp{-[1-shape*(x-mu)/scale]**(1/shape)}       (max version)
    f = [1-shape*(x-mu)/scale]**(1/shape-1) * 
                               exp{-[1-shape*(x-mu)/scale]**(1/shape)} / scale

    F = 1 - exp{-[1+shape*(x-mu)/scale]**(1/shape)}   (min version)
    f = [1+shape*(x-mu)/scale]**(1/shape-1) * 
                               exp{-[1+shape*(x-mu)/scale]**(1/shape)} / scale
    shape  < 0 => Type II
    shape  > 0 => Type III
    shape -> 0 => Type I - Gumbel

    type must be 'max' or 'min'
    scale must be > 0.0

    A REASONABLE SCHEME SEEMS TO BE mu = scale WHICH SEEMS TO LIMIT THE
    DISTRIBUTION TO EITHER SIDE OF THE Y-AXIS!
    """

    if shape == 0.0:
        cdf = cextreme_I(type, mu, scale, x)
    
    else:
        assert scale > 0.0, "scale must be positive in cextreme_gen!"

        if type == 'max':
            crucial = 1.0 - shape*(x-mu)/float(scale)
            if crucial <= 0.0 and shape < 0.0:
                cdf = 0.0
            else:
                y   = crucial ** (1.0/shape)
                cdf = exp(-y)

        elif type == 'min':
            crucial = 1.0 + shape*(x-mu)/float(scale)
            if crucial <= 0.0 and shape < 0.0:
                cdf = 1.0
            else:
                y   = crucial ** (1.0/shape)
                cdf = 1.0 - exp(-y)

        else:
            raise Error("type must be either 'max' or 'min' in cextreme_gen!")


    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cextreme_gen

# ------------------------------------------------------------------------------

def clogistic(mu, scale, x):
    """
    The logistic distribution:
    f = exp[-(x-m)/s] / (s*{1 + exp[-(x-m)/s]}**2)
    F = 1 / {1 + exp[-(x-m)/s]}
    x in R
    m is the mean and mode, s is a scale parameter (s > 0)
    """

    assert scale > 0.0, "scale must be positive in clogistic!"

    cdf = 1.0 / (1.0 + exp(-(x-mu)/float(scale)))

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of clogistic

# ------------------------------------------------------------------------------
# This is a department with SYMMETRICAL, stable distributions
# ------------------------------------------------------------------------------

def ccauchy(location, scale, x):
    """
    The cdf of the Cauchy distribution (also known 
    as the Lorentzian or Lorentz distribution):
    f = 1 / [s*pi*(1 + [(x-m)/s]**2)]
    F = (1/pi)*arctan((x-m)/s) + 1/2
    
    scale > 0.0  
    """

    assert scale > 0.0, "scale must be a positive float in ccauchy!"

    x   =  (x-location) / float(scale)
    #cdf =  atan(x)/PI + 0.5
    cdf =  PIINV*atan(x) + 0.5

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of ccauchy

# ------------------------------------------------------------------------------

def cnormal(mu, sigma, x):
    """
    cdf for the normal (Gaussian) distribution based on the erfc1 function 
    that offers an estimated maximum fractional error < 50*machine epsilon.
    
    sigma > 0.0
    """

    assert sigma > 0.0, "sigma must be a positive float in cnormal!"

    x   =  (x-mu) / float(sigma)
    y   =  SQRT05 * abs(x)
    cdf =  0.5 * erfc1(y)
    if x > 0.0: cdf = 1.0 - cdf

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cnormal

# ------------------------------------------------------------------------------

def clognormal(mulg, sigmalg, x):
    """
    cdf for the lognormal distribution based on the cnormal function above.
    The log10-converted form is assumed for mulg and sigmalg: 
    mulg is the mean of the log10 (and the log10 of the median) of 
    the random variate, NOT the log10 of the mean of the non-logged 
    variate!, and sigmalg is the standard deviation of the log10 of 
    the random variate, NOT the log10 of the standard deviation of 
    the non-logged variate!!
    
    sigmalg > 0.0
    """

    assert x >= 0.0, "variate must be non-negative in clognormal!"

    try:                return cnormal(mulg, sigmalg, log10(x))
    except ValueError:  return 0.0

# end of clognormal

# ------------------------------------------------------------------------------

def cfoldednormal(muunfold, sigmaunfold, x):
    """
    The cdf of a random variable that is the absolute value of a variate drawn 
    from the normal distribution (i. e. the distribution of a variate that is 
    the absolute value of a normal variate, the latter having muunfold as its 
    mean and sigmaunfold as its standard deviation). 
    
    sigmaunfold >= 0.0
    """

    # sigmaunfold > 0.0 assertion i cnormal
    assert x >= 0.0, "x must be a positive float in cfoldednormal!"

    cdf = cnormal(muunfold, sigmaunfold, x) - cnormal(muunfold, sigmaunfold, -x)

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of cfoldednormal

# ------------------------------------------------------------------------------

def cstable_sym(alpha, location, scale, x):
    """
    Cumulative distribution of a SYMMETRICAL stable distribution where alpha is 
    the tail exponent. For numerical reasons alpha is restricted to [0.25, 0.9] 
    and [1.125, 1.9] - but alpha = 1.0 (the Cauchy) and alpha = 2.0 (scaled 
    normal) are also allowed!

    Numerics are somewhat crude but the fractional error is mostly < 0.001 - 
    sometimes much less - and the absolute error is almost always < 0.001 - 
    sometimes much less... 

    NB This function is slow, particularly for small alpha !!!!!
    """

    # NB Do not change the numerical parameters - they are matched - they 
    # are also matched with the parameters in the corresponding pdf function 
    # dstable_sym on which this function is partly based! Changes in dstable_sym
    # are likely to require changes in this function!

    assert 0.25 <= alpha and alpha <= 2.0,  \
                               "alpha must be in [0.25, 2.0] in cstable_sym!"
    if alpha < 1.0: assert alpha <= 0.9,    \
                                "alpha <= 1.0 must be <= 0.9 in cstable_sym!"
    if alpha > 1.0: assert alpha >= 1.125,  \
                               "alpha > 1.0 must be >= 1.125 in cstable_sym!"
    if alpha > 1.9: assert alpha == 2.0,    \
                                  "alpha > 1.9 must be = 2.0 in cstable_sym!"
    assert scale > 0.0,      "scale must be a positive float in cstable_sym!"

    if alpha == 1.0: return ccauchy(location, scale, x)
    if alpha == 2.0: return cnormal(location, SQRT2*scale, x)

    x  =  (x-location) / float(scale)
    s  =  fsign(x)
    x  =  abs(x)

    if x == 0.0: return 0.5

    if alpha < 1.0:
        if x <= 1.0:
            tolromb =  0.5**12 / (alpha*alpha)
            cdf     = _stable_sym_int(alpha, x, tolromb, 10)
        else:
            cdf     = _cstable_sym_big(alpha, x, MACHEPS)

    elif alpha > 1.0:
        y1 = -2.212502985 + alpha*(3.03077875081 - alpha*0.742811132)
        dy =  0.130 * sqrt((alpha-1.0))
        y2 =  y1 + dy
        y1 =  y1 - dy
        y1 =  pow(10.0, y1)
        y2 =  pow(10.0, y2)
        if x <= y1:
            cdf = _cstable_sym_small(alpha, x, MACHEPS)
        elif x <= y2:
            c1  = (x-y1)/(y2-y1)
            c2  = 1.0 - c1
            cdf = c2*_cstable_sym_small(alpha, x, MACHEPS) + \
                  c1*_stable_sym_tail(alpha, x)
        else:
            cdf = _stable_sym_tail(alpha, x)

    if s < 0.0: cdf = 1.0 - cdf
    cdf = kept_within(0.0, cdf, 1.0)
    return cdf

# end of cstable_sym

# -------------------------------------------------------------

def _cstable_sym_small(alpha, x, tolr):
    """
    A series expansion for small x due to Bergstrom. Converges 
    for x < 1.0 and in practice also for somewhat larger x.
    The function uses the Kahan summation procedure 
    (cf. Dahlquist, Bjorck & Anderson). 
    """

    summ    =  0.0
    c       =  0.0
    fact    = -1.0
    xx      =  x*x
    xpart   =   x
    k       =   0
    zero2   =  zero1  =  False
    while True:
        k       +=  1
        summo    =  summ
        twokm1   =  2*k - 1
        twokm1oa =  float(twokm1)/alpha
        r        =  lngamma(twokm1oa) - lnfactorial(twokm1)
        term     =  exp(r) * xpart
        fact     = - fact
        term    *=  fact
        y        =  term + c
        t        =  summ + y
        if fsign(y) == fsign(summ):
            f = (0.46*t-t) + t
            c = ((summ-f)-(t-f)) + y
        else:
            c = (summ-t) + y
        summ     =  t
        if abs(summ-summo) < tolr*abs(summ) and abs(term) < tolr and zero2:
            break
        xpart *=  xx
        if abs(term) < tolr:
            if zero1: zero2 = True
            else:     zero1 = True
    summ +=  c
    cdf   =  summ/(PI*alpha) + 0.5

    cdf  =  kept_within(0.5, cdf, 1.0)
    return cdf

# end of _cstable_sym_small

# --------------------------------------------------

def _cstable_sym_big(alpha, x, tolr):
    """
    A series expansion for large x due to Bergstrom. 
    Converges for x > 1.0
    The function uses the Kahan summation procedure 
    (cf. Dahlquist, Bjorck & Anderson). 
    """

    summ = 0.0
    c    = 0.0
    fact = 1.0
    k    =  0
    zero = False
    while True:
        k     +=  1
        summo  =  summ
        ak     =  alpha * k
        akh    =  0.5 * ak
        r      =  lngamma(ak) - lnfactorial(k)
        term   =  exp(r) * sin(PIHALF*ak) / pow(x, ak)
        fact   = - fact
        term  *=  fact
        y      =  term + c
        t      =  summ + y
        if fsign(y) == fsign(summ):
            f = (0.46*t-t) + t
            c = ((summ-f)-(t-f)) + y
        else:
            c = (summ-t) + y
        summ   =  t
        if abs(summ-summo) < tolr*abs(summ) and abs(term) < tolr and zero:
            break
        if abs(term) < tolr: zero = True
    summ +=  c
    #cdf   =  summ/PI + 1.0
    cdf   =  PIINV*summ + 1.0

    cdf  =  kept_within(0.5, cdf, 1.0)
    return cdf

# end of _cstable_sym_big

# ----------------------------------------

def _stable_sym_tail(alpha, x):
    """
    An asymptotic expression for the tail.
    """

    #calpha = exp(lngamma(alpha)) * sin(PIHALF*alpha) / PI
    calpha = PIINV * exp(lngamma(alpha)) * sin(PIHALF*alpha)

    try:
        cdf = calpha / x**alpha

    except ZeroDivisionError:
        cdf = log(calpha) - alpha*log(x)
        try:                  cdf = exp(cdf)
        except OverflowError: cdf = 0.0

    except OverflowError:
        cdf = log(calpha) - alpha*log(x)
        try:                  cdf = exp(cdf)
        except OverflowError: cdf = 0.0

    cdf = 1.0 - cdf

    cdf = kept_within(0.5, cdf, 1.0)
    return cdf

# end of _stable_sym_tail

# ------------------------------------------------

def _stable_sym_int(alpha, x, tolromb, mxsplromb):
    """
    Integration of the standard pdf
    (nb a change of integration variable is made!)
    """

    assert alpha < 1.0, "alpha must be < 1.0 in _stable_sym_int!"

    onema     = 1.0 - alpha
    oneoonema = 1.0 / onema
    aoonema   = alpha * oneoonema

    # -------------------------------------------------------------------------
    def _func(t):
        return dstable_sym(alpha, 0.0, 1.0, pow(t, oneoonema)) * pow(t, aoonema)
    # -------------------------------------------------------------------------

    cdf  = oneoonema * qromberg(_func, 0.0, pow(x, onema), \
                            'cstable_sym/_stable_sym_int', tolromb, mxsplromb)
    cdf += 0.5

    cdf = kept_within(0.5, cdf, 1.0)
    return cdf

# end of _stable_sym_int

# ------------------------------------------------------------------------------

def clevy(scale, x):
    """
    The cdf of the Levy distribution (stable distribution with 
    alpha = 1/2 and beta = 1, aka the Cournot distribution). 
    This is actually the right-skewed Levy!
    f = sqrt(s/2pi) * (1/x)**(3/2) * exp(-s/2x)
    F = erfc(sqrt(s/2x))
    
    s >= 0.0, x >= 0
    """

    assert scale >= 0.0, "scale must not be negative in clevy!"
    assert   x   >= 0.0, "variate must not be negative in clevy!"

    # The cdf of the Levy can be handled since it is an "incomplete gamma 
    # function", but it seems to be more simple than that:

    try:
        cdf = erfc1(sqrt(0.5*scale/x))
    except (OverflowError, ZeroDivisionError):
        return 0.0

    cdf = kept_within(0.0, cdf, 1.0)

    return cdf

# end of clevy

# ------------------------------------------------------------------------------
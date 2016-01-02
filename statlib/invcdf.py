# statlib/invcdf.py
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
MODULE WITH FUNCTIONS FOR INVERTING VARIOUS PROBABILITY DISTRIBUTIONS. 
NB. Some functions may return float('inf') or float('-inf') !
"""
# ------------------------------------------------------------------------------

from math   import exp, sqrt, tan, acos, modf, log10
from bisect import bisect

from statlib.cdf       import chyperexpo, cNexpo, cNexpo2, cerlang, cerlang_gen
from statlib.cdf       import ccoxian, ccoxian2, cgamma, cbeta, cexppower
from statlib.cdf       import cnormal, cfoldednormal, cstable_sym
from statlib.pdf       import dhyperexpo, dNexpo, dNexpo2, derlang, derlang_gen
from statlib.pdf       import dcoxian, dcoxian2, dfoldednormal, dstable_sym
from statlib.pdf       import dgamma, dexppower
from statlib.binco     import binprob
from numlib.solveq     import znewton, z2nddeg_real, zbrent
from numlib.miscnum    import fsign, safediv, safelog, safepow
from misclib.numbers   import is_integer, is_posinteger, is_nonneginteger
from misclib.numbers   import kept_within
from machdep.machnum   import *
from misclib.errwarn   import Error
from misclib.mathconst import PI, PIINV

# ------------------------------------------------------------------------------

def iunifab(prob, left=0.0, right=1.0):
    """
    The inverse of a uniform distribution between left and right:
    F = (x-left) / (right-left); left <= x <= right 
    """

    _assertprob(prob, 'iunifab')
    # ---
    assert right >= left, "support range must not be negative in iunifab!"

    x = prob*(right-left) + left

    x = kept_within(left, x, right)

    return x

# end of iunifab

# ------------------------------------------------------------------------------

def itriang(prob, left, mode, right):
    """
    The inverse of a triangular distribution between left and right 
    having its peak at x = mode 
    """

    _assertprob(prob, 'itriang')
    # ---
    assert left <= mode and mode <= right, \
                                  "mode out of support range in itriang!"

    span    =  right - left
    spanlo  =  mode  - left
    spanhi  =  right - mode
    #height  =  2.0 / span
    #surf1   =  0.5 * spanlo * height
    #surf1   =  spanlo/float(span)

    #if prob <= surf1:
    if prob <= spanlo/float(span):
        #x  =  sqrt(2.0*spanlo*prob/height)
        x  =  sqrt(spanlo*span*prob)
    else:
        #x  =  span - sqrt(2.0*spanhi*(1.0-prob)/height)
        x  =  span - sqrt(spanhi*span*(1.0-prob))
    x += left

    x  = kept_within(left, x, right)

    return x

# end of itriang

# ------------------------------------------------------------------------------

def itri_unif_tri(prob, a, b, c, d):
    """
    The inverse of triangular-uniform-triangular distribution with support on 
    [a, d] and with break points in b and c
              ------
    pdf:    /        \
           /           \
    ------              -------
    """


    # Input check -----------------------
    assert 0.0 <= prob and prob <= 1.0,  \
                          "probability out of support range in itri_unif_tri!"
    assert d > a,                     "nonexistent support for itri_unif_tri!"
    assert a <= b and b <= c and c <= d, \
                                    "break points scrambled in itri_unif_tri!"
    # -----------------------------------


    dcba   =  d + c - b - a
    h      =  2.0 / dcba
    first  =  0.5 * h * (b-a)
    poh    =  0.5 * prob * dcba

    if prob <= first:
        x  =  sqrt(2.0*(b-a)*poh) + a

    elif first < prob <= first + h*(c-b):
        x  =  (c-b)*(poh-0.5*(b-a)) + b

    else:
        x  =  d - sqrt((d-c)*dcba*(1.0-prob))


    x  = kept_within(a, x, d)
    
    return x

# end of itri_unif_tri

# ------------------------------------------------------------------------------

def idiscrete(prob, values, qumul):
    """
    Computes the inverse of a user-defined discrete cdf. 

    'values' is a list/tuple with numbers, and 'qumul' are the corresponding 
    CUMULATIVE FREQUENCIES such that qumul[k] = P(x<=values[k]). The number of 
    values must be equal to the number of cumulative frequencies.
        
    The cumulative frequencies must of course obey qumul[k+1] >= qumul[k],
    otherwise an exception will be raised!
    
    The 'values' list/tuple does not have to be sorted in order for the 
    function to return the inverse!
    """

    # Input check ------------
    _assertprob(prob, 'idiscrete')
    assert len(values) == len(qumul), \
                   "Input vectors are of unequal length in idiscrete!"
    assert qumul[-1]  == 1.0
    assert 0.0 <= qumul[0] and qumul[0] <= 1.0
    # ---
    nvalues = len(values)
    for k in range(1, nvalues):
        assert qumul[k] >= qumul[k-1], \
                 "qumul vector is not in order in idiscrete!"
        pass
    # ---

    vcopy = list(values)
    ordered = True
    for k in range(1, nvalues):
        if values[k] < values[k-1]:
            vcopy.sort()    # Not sorted: sort!
            ordered = False
            break
    if ordered:       # OK, vardena var ordnade efter storlek
        qcumul = qumul
    else:              # Annars: stuva om i qumul
        hash = {}
        hash[values[0]] = qumul[0]
        for k in range(1, nvalues):
            hash[values[k]]  =  qumul[k] - qumul[k-1]
        qcumul = [hash[vcopy[0]]]
        for k in range(1, nvalues):
            qcumul.append(qcumul[k-1]+hash[vcopy[k]])

    #k = binSearch(qcumul, prob)[0]   # Only the first of two outputs is needed
    k = bisect(qcumul, prob)      # bisect used instead of binSearch
    x = vcopy[k]

    return x

# end of idiscrete

# ------------------------------------------------------------------------------

def ichistogram(prob, values, qumul):
    """
    Calculates the inverse value of an input CUMULATIVE histogram.
    'values' is a list/tuple with FLOATS in ascending order - A MUST! 
    These values represent bin end points and must be one more than 
    the number of cumulative frequencies, and where...
    ...'qumul' are the corresponding CUMULATIVE FREQUENCIES such that 
    qumul[k] = P(x<=values[k+1]).

    The cumulative frequencies must of course obey qumul[k+1] >= qumul[k],
    otherwise an exception will be raised!
        
    The values of the random variate are assumed to be uniformly 
    distributed within each bin.
    """

    nvalues = len(values)
    nqumul = len(qumul)

    # Input check ---
    _assertprob(prob, 'ichistogram')
    errtxt1 = "Lengths of input lists are incompatible in ichistogram!"
    assert nqumul == nvalues-1, errtxt1
    assert qumul[-1]  == 1.0, "last qumul in list must be 1.0 in ihistogram!"
    assert 0.0 <= qumul[0] and qumul[0] <= 1.0, \
                    "first qumul in list must be in [0.0, 1.0] in ihistogram!"
    # ---
    errtxt2 = "qumul list is not in order in ichistogram!"
    for k in range(1, nqumul):
        assert qumul[k] >= qumul[k-1], errtxt2
        pass
    # ---
    errtxt3 = "values list is not in order in ichistogram!"
    for k in range(1, nvalues):
        assert values[k] >= values[k-1], errtxt3
        pass
    # ---

    # The routine itself.................

    pcopy = list(qumul)
    pcopy.insert(0, 0.0)
    
    #k      = binSearch(pcopy, prob)[0]  # Only the 1st of two outputs is needed
    k      = bisect(pcopy, prob)
    pdiff  = prob - pcopy[k-1]
    pinter = pcopy[k]  - pcopy[k-1]
    vinter = values[k] - values[k-1]
    x      = values[k-1] + vinter*pdiff/float(pinter)

    return x

# end of ichistogram

# ------------------------------------------------------------------------------

def ichistogram_int(prob, values, qumul):
    """
    Calculates the inverse value of an input CUMULATIVE histogram.
    'values' is a list/tuple with INTEGERS in ascending order - A MUST! 
    These values represent bin end points and must be one more than 
    the number of cumulative frequencies, and where...
    ...'qumul' are the corresponding CUMULATIVE FREQUENCIES such that 
    qumul[k] = P(x<=values[k+1]).

    NB The first element of the values list is will never be returned!
    The first integer to be returned is values[0] + 1   !!!!

    The cumulative frequencies must of course obey qumul[k+1] >= qumul[k],
    otherwise an exception will be raised!
        
    The values of the integer random variate are assumed to be uniformly 
    distributed within each bin.
    """

    nvalues = len(values)
    nqumul = len(qumul)

    # Input check ---
    _assertprob(prob, 'ichistogram_int')
    errtxt1 = "Lengths of input lists are incompatible in ichistogram_int!"
    assert nqumul == nvalues-1, errtxt1
    assert qumul[-1]  == 1.0
    assert 0.0 <= qumul[0] and qumul[0] <= 1.0
    # ---
    errtxt2 = "qumul list is not in order in ichistogram_int!"
    for k in range(1, nqumul):
        assert qumul[k] >= qumul[k-1], errtxt2
        pass
    # ---
    errtxt3 = "all values in values list must be integers in ichistogram_int!"
    errtxt4 = "values list is not in order in ichistogram_int!"
    assert is_integer(values[0]), errtxt3
    for k in range(1, nvalues):
        assert is_integer(values[k]), errtxt3
        assert values[k] >= values[k-1], errtxt4
        pass
    # ---

    # The routine itself.................

    pcopy = list(qumul)
    pcopy.insert(0, 0.0)

    k      = bisect(pcopy, prob)      # bisect used instead of binSearch
    pdiff  = prob - pcopy[k-1]
    pinter = pcopy[k]  - pcopy[k-1]
    vinter = values[k] - values[k-1]
    x      = values[k-1] + vinter*pdiff/float(pinter)
    x      = int(x+1.0)

    return x

# end of ichistogram_int

# ------------------------------------------------------------------------------

def ibeta(prob, a, b, x1=0.0, x2=1.0, betaab=False):
    """
    The beta distribution:
    f = x**(a-1) * (1-x)**(b-1) / beta(a, b)
    a, b >= 0; 0 <= x <= 1
    F is the integral = the incomplete beta or the incomplete beta ratio 
    function depending on how the incomplete beta function is defined.
    
    x2 >= x1 !!!!
    
    NB It is possible to provide the value of the complete beta 
    function beta(a, b) as a pre-computed input (may be computed 
    using numlib.specfunc.beta) instead of the default "False", 
    a feature that will make ibeta 30 % faster!
    """

    # Everything will be checked in cbeta

    if a == 1.0 and b == 1.0: return iunifab(prob, x1, x2)

   # -----------------------------------------------------------
    def _fi(x):
        return cbeta(a, b, x1, x2, x, betaab) - prob
   # -----------------------------------------------------------

    x = zbrent(_fi, x1, x2, 'ibeta', tolf=SQRTMACHEPS)

    x = kept_within(x1, x, x2)
    
    return x

# end of ibeta

# ------------------------------------------------------------------------------

def ikumaraswamy(prob, a, b, x1=0.0, x2=1.0):
    """
    The Kumaraswamy distribution: f = a*b*x**(a-1) * (1-x**a)**(b-1)
                                  F = 1 - (1-x**a)**b
                                 a, b >= 0; 0 <= x <= 1
    The Kumaraswamy distribution is VERY similar to the beta distribution!!!
    
    x2 >= x1 !!!!
    """

    _assertprob(prob, 'ikumaraswamy')
    assert a  >  0.0, "both shape parameters in ikumaraswamy must be positive!"
    assert b  >  0.0, "both shape parameters in ikumaraswamy must be positive!"
    assert x2 >= x1,  "support range in ikumaraswamy must not be negative!"

    y  =  (1.0 - (1.0-prob)**(1.0/b)) ** (1.0/a)

    x  =  y*(x2-x1) + x1

    x  =  kept_within(x1, x, x2)

    return x

# end of ikumaraswamy

# ------------------------------------------------------------------------------

def isinus(prob, left=0.0, right=1.0):
    """
    The inverse of the sinus distribution. 
    """

    _assertprob(prob, 'isinus')
    assert right >= left, "support range must not be negative in isinus!"

    #x  =  left  +  (right-left) * acos(1.0-2.0*prob) / PI
    x  =  left  +  (right-left) * PIINV*acos(1.0-2.0*prob)

    x  =  kept_within(left, x, right)

    return x

# end of isinus

# ------------------------------------------------------------------------------

def ibinomial(prob, n, phi, normconst=10.0):
    """
    The binomial distribution: p(N=k) = bincoeff * phi**k * (1-phi)**(n-k), 
    n >= 1, k = 0, 1,...., n 
    """

    # Input check -----------
    _assertprob(prob, 'ibinomial')
    assert is_posinteger(n),        "n must be a positive integer in ibinomial!"
    assert 0.0 <= phi and phi <= 1.0, \
                          "success frequency out of support range in ibinomial!"
    assert normconst >= 10.0, \
           "parameter limit for normal approx. in ibinomial must not be < 10.0!"
    # -----------------------

    onemphi = 1.0 - phi

    if phi < 0.5: w = normconst * onemphi / phi
    else:         w = normconst * phi / onemphi

    if n > w:
        k = int(round(inormal(prob, n*phi, sqrt(n*phi*onemphi))))

    else:
        k   = 0
        cdf = binProb(n, phi)[1]
        while True:
            if cdf[k] <= prob:  k = k + 1
            else:               break

    k = kept_within(0, k, n)

    return k

# end of ibinomial

# ------------------------------------------------------------------------------
 
def igeometric(prob, phi):
    """
    The geometric distribution with p(K=k) = phi * (1-phi)**(k-1)  and 
    P(K>=k) = sum phi * (1-phi)**k = 1 - q**k where q = 1 - phi and  
    0 < phi <= 1 is the success frequency or "Bernoulli probability" and 
    K >= 1 is the number of  trials to the first success in a series of 
    Bernoulli trials. It is easy to prove that P(k) = 1 - (1-phi)**k: 
    let q = 1 - phi. p(k) = (1-q) * q**(k-1) = q**(k-1) - q**k. 
    Then P(1) = p(1) = 1 - q. P(2) = p(1) + p(2) = 1 - q + q - q**2 = 1 - q**2. 
    Induction can be used to show that P(k) = 1 - q**k = 1 - (1-phi)**k 
    
    The algorithm is taken from ORNL-RSIC-38, Vol II (1973). 
    """

    _assertprob(prob, 'igeometric')
    assert 0.0 <= phi and phi <= 1.0, \
                      "success frequency must be in [0.0, 1.0] in igeometric!"


    if phi == 1.0: return 1   # Obvious...

    q  =  1.0 - phi

    if phi < 0.25:            # Use the direct inversion formula
        lnq   = - safelog(q)
        ln1mp = - safelog(1.0 - prob)
        kg    =  1 + int(ln1mp/lnq)

    else:             # Looking for the passing point is more efficient for 
        kg = 1        # phi >= 0.25 (it's still inversion)
        u  = prob
        a  = phi
        while True:
            u = u - a
            if u > 0.0:
                kg += 1
                a  *= q
            else:
                break

    return kg

# end of igeometric

# ------------------------------------------------------------------------------

def ipoisson(prob, lam, tspan):
    """
    The Poisson distribution: p(N=n) = exp(-lam*tspan) * (lam*tspan)**n / n!
    n = 0, 1,...., infinity 
    """

    # Input check -----------
    assert 0.0 <= prob and prob < 1.0, \
                          "probability out of support range in ipoisson!"
    assert  lam  >= 0.0, "Poisson rate must not be negative in ipoisson!"
    assert tspan >= 0.0, "time span must not be negative in ipoisson!"
    # -----------------------

    lamtau = lam*tspan

    if lamtau < 64.0:           # Just hammer away ...
        n = 0
        p = exp(-lamtau)
        c = p
        f = float(lamtau)
        while c <= prob:
            n  = n + 1
            p *= f/n
            c += p

    else:                       # ... otherwise use a normal approximation
        n = int(round(inormal(prob, lamtau, sqrt(lamtau))))
        n = max(0, n)

    return n

# end of ipoisson

# ------------------------------------------------------------------------------

def iexpo(prob, mean=1.0):
    """
    The inverse of the exponential distribution with mean = 1/lambda: 
    f = (1/mean) * exp(-x/mean)
    F = 1 - exp(-x/mean)
    
    x >= 0, mean >= 0.0 
    """

    _assertprob(prob, 'iexpo')
    # ---
    assert mean >= 0.0, "mean of variate in iexpo must be a non-negative float!"

    x  =  - mean * safelog(1.0-prob)

    #x  =  kept_within(0.0, x)   # Not really needed

    return x

# end of iexpo

# ------------------------------------------------------------------------------

def ihyperexpo(prob, means, qumul):
    """
    The inverse of the hyperexponential distribution 
    f = sumk pk * exp(x/mk) / mk
    F = sumk pk * (1-exp(x/mk)) 
    
    NB Input to the function is the list of CUMULATIVE FREQUENCIES ! 
    
    NB Slow function ! 
    """

    _assertprob(prob, 'ihyperexpo')
    # Everything else will be checked in iexpo, chyperexpo and dhyperexpo

    if len(means) == 1 and len(qumul) == 1 and qumul[0] == 1.0:
        x = iexpo(prob, means[0])

    else:
       # ------------------------------------
        def _fifi2fid(x):
            x      = kept_within(0.0, x)
            cdf    = chyperexpo(means, qumul, x)
            pdf    = dhyperexpo(means, qumul, x)
            fi     = cdf - prob
            if pdf <= 0.0:
                if fi == 0.0: fi2fid = 1.0
                else:         fi2fid = MAXFLOAT
            else:
                fi2fid = fi/pdf
            return fi, fi2fid
       # ------------------------------------

        x = znewton(_fifi2fid, 0.0, 'ihyperexpo', tolf=SQRTMACHEPS)
        x = kept_within(0.0, x)

    return x

# end of ihyperexpo

# ------------------------------------------------------------------------------

def iemp_exp(prob, values, npexp=0, ordered=False):
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

    _assertprob(prob, 'iemp_exp')
    assert is_nonneginteger(npexp), \
        "No. of points for exp. tail in iemp_exp must be a non-neg integer!"

    nvalues = len(values)
    for k in range(0, nvalues):
        assert values[k] >= 0.0, \
                        "All values in list must be non-negative in iemp_exp!"

    if npexp == nvalues:
        mean = sum(values)
        return iexpo(prob, mean)

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

    if vcopy[0] == 0.0:   # Remove the first zero if any - it will be
        del vcopy[0]      # returned later on!!!!!!!!
        nvalues = nvalues - 1

    errtxt2 = "Number of points for exponential tail in iemp_exp too large!"
    assert npexp <= nvalues, errtxt2

    pcomp = 1.0 - prob
    nred  = pcomp*nvalues
    if npexp > pcomp*nvalues:
        breaki = nvalues - npexp - 1  # The last ix of the piecewise linear part
        breakp = vcopy[breaki]    # The last value of the piecewise linear part
        summ   = 0.0
        k0 = breaki + 1
        for k in range(k0, nvalues):
            summ += vcopy[k] - breakp
        theta  = (0.5*breakp + summ) / npexp
        q  =  npexp / float(nvalues)
        try:
            x  =  breakp - theta*safelog(nred/npexp)
        except ValueError:
            x  =  float('inf')
    else:
        vcopy.insert(0, 0.0)               # A floor value must be inserted
        v  =  nvalues*prob
        i  =  int(v)
        x  =  vcopy[i] + (v-i)*(vcopy[i+1]-vcopy[i])
        if npexp == 0:  # Correction to maintain mean when there is no exp tail
            x  =  (nvalues+1.0) * x / nvalues

    x = kept_within(0.0, x)

    return x

# end of iemp_exp

# ------------------------------------------------------------------------------

def iNexpo(prob, means):
    """
    A distribution of a sum of exponential random variables.
    
    NB means are allowed to be equal!!! 
    """

    _assertprob(prob, 'iNexpo')
    # Everything else will be checked in iexpo, cNexpo and dNexpo

    if len(means) == 1:
        x = iexpo(prob, means[0])

    else:
       # ------------------------------------
        def _fifi2fid(x):
            x    = kept_within(0.0, x)
            cdf  = cNexpo(means, x)
            pdf  = dNexpo(means, x)
            fi   = cdf - prob
            if pdf <= 0.0:
                if fi == 0.0: fi2fid = 1.0
                else:         fi2fid = MAXFLOAT
            else:
                fi2fid = fi/pdf
            return fi, fi2fid
       # ------------------------------------

        x = znewton(_fifi2fid, max(means), 'iNexpo', tolf=SQRTMACHEPS)

        x = kept_within(0.0, x)
    
    return x

# end of iNexpo

# ------------------------------------------------------------------------------

def iNexpo2(prob, means):
    """
    A distribution of a sum of exponential random variables.
    
    NB No two means are allowed to be equal!!!! 
    """

    _assertprob(prob, 'iNexpo2')
    # Everything else will be checked in iexpo, cNexpo2 and dNexpo2

    if len(means) == 1:
        x = iexpo(prob, means[0])

    else:
       # ------------------------------------
        def _fifi2fid(x):
            x      = kept_within(0.0, x)
            cdf    = cNexpo2(means, x)
            pdf    = dNexpo2(means, x)
            fi     = cdf - prob
            if pdf <= 0.0:
                if fi == 0.0: fi2fid = 1.0
                else:         fi2fid = MAXFLOAT
            else:
                fi2fid = fi/pdf
            return fi, fi2fid
       # ------------------------------------

        x = znewton(_fifi2fid, max(means), 'iNexpo2', tolf=SQRTMACHEPS)

        x = kept_within(0.0, x)
    
    return x

# end of iNexpo2

# ------------------------------------------------------------------------------

def iexpo_gen(prob, a, b, c=0.0):
    """
    The generalized continuous exponential distribution (x in R):
    x <= c: f  =  [a*b/(a+b)] * exp(+a*[x-c])
            F  =   [b/(a+b)]  * exp(+a*[x-c])
    x >= c: f  =  [a*b/(a+b)] * exp(-b*[x-c])
            F  =  1 - [a/(a+b)]*exp(-b*[x-c])
    a > 0, b > 0
    
    NB The symmetrical double-sided exponential sits in ilaplace!
    """

    _assertprob(prob, 'iexpo_gen')
    assert a > 0.0
    assert b > 0.0

    r = prob*(a+b)/b

    if r <= 1.0:
        x  =  c  +  safelog(r) / a
    else:
        x  =  c  -  safelog((a+b)*(1.0-prob)/a) / b

    return x

# end of iexpo_gen

# ------------------------------------------------------------------------------

def ierlang(prob, nshape, phasemean=1.0):
    """
    Represents the sum of nshape exponentially distributed random variables, 
    each having the same mean value = phasemean 
    """

    _assertprob(prob, 'ierlang')
    # Everything else will be checked in iexpo, cerlang and derlang

    if nshape == 1:
        x = iexpo(prob, phasemean)

    else:
       # ------------------------------------
        def _fifi2fid(x):
            x      = kept_within(0.0, x)
            cdf    = cerlang(nshape, phasemean, x)
            pdf    = derlang(nshape, phasemean, x)
            fi     = cdf - prob
            if pdf <= 0.0:
                if fi == 0.0: fi2fid = 1.0
                else:         fi2fid = MAXFLOAT
            else:
                fi2fid = fi/pdf
            return fi, fi2fid
       # ------------------------------------

        x = znewton(_fifi2fid, (nshape-1.0)*phasemean, 'ierlang', \
                                                     tolf=SQRTMACHEPS)

        x = kept_within(0.0, x)
    
    return x

# end of ierlang

# ------------------------------------------------------------------------------

def ierlang_gen(prob, nshapes, qumul, phasemean=1.0):
    """
    The inverse of the generalized Erlang distribution - the Erlang 
    equivalent of the hyperexpo distribution f = sumk pk * ferlang(m, nk), 
    F = sumk pk * Ferlang(m, nk), the same mean for all phases.
    
    NB Input to the function is the list of CUMULATIVE FREQUENCIES ! 
    
    NB Slow function !
    """

    _assertprob(prob, 'iexpo')
    # Everything else will be checked in cerlang_gen and derlang_gen

    if len(nshapes) == 1 and len(qumul) == 1 and qumul[0] == 1.0:
        x = ierlang(nshapes[0], phasemean)

    else:
       # -----------------------------------
        def _fifi2fid(x):
            x      = kept_within(0.0, x)
            cdf    = cerlang_gen(nshapes, qumul, phasemean, x)
            pdf    = derlang_gen(nshapes, qumul, phasemean, x)
            fi     = cdf - prob
            if pdf <= 0.0:
                if fi == 0.0: fi2fid = 1.0
                else:         fi2fid = MAXFLOAT
            else:
                fi2fid = fi/pdf
            return fi, fi2fid
       # ------------------------------------

        x = znewton(_fifi2fid, phasemean, 'ierlang_gen', tolf=SQRTMACHEPS)

        x = kept_within(0.0, x)

    return x

# end of ierlang_gen

# ------------------------------------------------------------------------------

def icoxian(prob, means, probs):
    """
    The Coxian phased distribution, which is based on the exponential.
    probs is a list of probabilities for GOING ON TO THE NEXT PHASE rather 
    than reaching the absorbing state prematurely. The number of means must 
    (of course) be one more than the number of probabilities! 
    NB means are allowed to be equal (but the function is slow). 
    """

    _assertprob(prob, 'icoxian')
    # Everything else will be checked in dcoxian and ccoxian

   # ------------------------------------
    def _fifi2fid(x):
        x      = kept_within(0.0, x)
        cdf    = ccoxian(means, probs, x)
        pdf    = dcoxian(means, probs, x)
        fi     = cdf - prob
        if pdf <= 0.0:
            if fi == 0.0: fi2fid = 1.0
            else:         fi2fid = MAXFLOAT
        else:
            fi2fid = fi/pdf
        return fi, fi2fid
   # ------------------------------------

    x = znewton(_fifi2fid, 0.0, 'icoxian', tolf=SQRTMACHEPS)

    x = kept_within(0.0, x)
    
    return x

# end of icoxian

# ------------------------------------------------------------------------------

def icoxian2(prob, means, probs):
    """
    The Coxian phased distribution, which is based on the exponential.
    probs is a list of probabilities for GOING ON TO THE NEXT PHASE rather 
    than reaching the absorbing state prematurely. The number of means must 
    (of course) be one more than the number of probabilities! 
    
    NB No two means[k] must be equal - if equal means are is desired, use 
    icoxian instead (slower, however). 
    """

    _assertprob(prob, 'icoxian2')
    # Everything else will be checked in dcoxian2 and ccoxian2

   # ------------------------------------
    def _fifi2fid(x):
        x      = kept_within(0.0, x)
        cdf    = ccoxian2(means, probs, x)
        pdf    = dcoxian2(means, probs, x)
        fi     = cdf - prob
        if pdf <= 0.0:
            if fi == 0.0: fi2fid = 1.0
            else:         fi2fid = MAXFLOAT
        else:
            fi2fid = fi/pdf
        return fi, fi2fid
   # ------------------------------------

    x = znewton(_fifi2fid, 0.0, 'icoxian2', tolf=SQRTMACHEPS)

    x = kept_within(0.0, x)
    
    return x

# end of icoxian2

# ------------------------------------------------------------------------------

def igamma(prob, alpha, lam=1.0, lngalpha=False, tolf=FOURMACHEPS, itmax=128):
    """
    The gamma distrib. f = lam * exp(-lam*x) * (lam*x)**(alpha-1) / gamma(alpha)
    F is the integral = the incomplete gamma or the incomplete gamma / complete 
    gamma depending on how the incomplete gamma function is defined.
    x, lam, alpha >= 0

    NB It is possible to provide the value of the natural logarithm of the 
    complete gamma function lngamma(alpha) as a pre-computed input (may be 
    computed using numlib.specfunc.lngamma) instead of the default "False", 
    a feature that will make igamma more than 50 % faster!

    tolf and itmax are the numerical control parameters of cgamma.
    """

    _assertprob(prob, 'igamma')
    # Everything else will be checked in cgamma and dgamma

    f, i = modf(alpha)
    if f == 0.0 and lam > 0.0: return ierlang(prob, int(i), 1.0/lam)

   # --------------------------------------------
    def _fifi2fid(x):
        x      = kept_within(0.0, x)
        cdf    = cgamma(alpha, lam, x, lngalpha, tolf, itmax)
        pdf    = dgamma(alpha, lam, x, lngalpha)
        fi     = cdf - prob
        if pdf <= 0.0:
            if fi == 0.0: fi2fid = 1.0
            else:         fi2fid = MAXFLOAT
        else:
            fi2fid = fi/pdf
        return fi, fi2fid
   # -------------------------------------------

    if alpha >= 2.0:  mean = alpha/lam
    else:             mean = SQRTMACHEPS
    x = znewton(_fifi2fid, mean, 'igamma', tolf=SQRTMACHEPS)

    x = kept_within(0.0, x)
    
    return x

# end of igamma

# ------------------------------------------------------------------------------

def ilaplace(prob, loc=0.0, scale=1.0):
    """
    The inverse of the Laplace distribution f = [(1/2)/s)]*exp(-abs([x-l]/s))
    F = (1/2)*exp([x-l]/s)  {x <= l},  F = 1 - (1/2)*exp(-[x-l]/s)  {x >= l}
    s >= 0
    """

    _assertprob(prob, 'ilaplace')
    # ---
    assert scale >= 0.0, "scale parameter in ilaplace must not be negative!"

    if prob <= 0.5:
        x =  safelog(2.0*prob)
        x =  scale*x + loc
    else:
        x = - safelog(2.0*(1.0-prob))
        x =  scale*x + loc

    return x

# end of ilaplace

# ------------------------------------------------------------------------------

def iexppower(prob, loc, scale, alpha, lngam1oalpha=False, \
                                       tolf=FOURMACHEPS, itmax=128):
    """
    The exponential power distribution 
    f  =  (a/s) * exp(-abs([x-l]/s)**a) / [2*gamma(1/a)]
    F  =  1/2 * [1 + sgn(x-l) * Fgamma(1/a, abs([x-l]/s)**a)],   x in R
    s, a > 0
    where Fgamma is the gamma distribution cdf.

    The function uses the igamma function.

    NB It is possible to gain efficiency by providing the value of the 
    natural logarithm of the complete gamma function ln(gamma(1.0/alpha)) 
    as a pre-computed input (may be computed using numlib.specfunc.lngamma) 
    instead of the default 'False'.
    
    tolf and itmax are the numerical control parameters of igamma.
    """

    # The following procedure is used:
    # if prob <= 0.5:
    #     Invert  1/2 * [1 - Pgamma(1/a, ([l-x]/s)**a)]
    #     prob  =  0.5 - 0.5*Pgamma(1/a, abs([l-x]/s)**a)
    #     Pgamma(1/a, ([l-x]/s)**a)  =  1.0 - 2.0*prob
    #     Use igamma to get y = ([l-x]/s)**a
    #     l - x  =  s * y**(1/a)
    #     x  =  l  -  s * y**(1/a)
    # else:
    #     Invert  1/2 * [1 + Pgamma(1/a, abs([x-l]/s)**a)]
    #     prob  =  0.5 + 0.5*Pgamma(1/a, ([x-l]/s)**a)
    #     Pgamma(1/a, ([x-l]/s)**a)  =  2.0*prob - 1.0
    #     Use igamma to get y = ([x-l]/s)**a
    #     x - l  =  s * y**(1/a)
    #     x  =  l  +  s * y**(1/a)


    _assertprob(prob, 'iexppower')
    # ---
    assert scale > 0.0, \
               "scale parameter must be a positive float in iexppower!"
    assert alpha > 0.0, \
            "shape parameter alpha must be a positive float in iexppower!"

    if alpha == 1.0: return ilaplace(prob, loc, scale)

    ainv = 1.0/float(alpha)
    glam = 1.0
    if prob <= 0.5:
        pgamma  =  1.0 - 2.0*prob
        y  =  igamma(pgamma, ainv, glam, lngam1oalpha, tolf, itmax)
        x  =  loc - scale*y**ainv
    else:
        pgamma  =  2.0*prob - 1.0
        y  =  igamma(pgamma, ainv, glam, lngam1oalpha, tolf, itmax)
        x  =  loc + scale*y**ainv

    return x

# end of iexppower

# ------------------------------------------------------------------------------

def iweibull(prob, c, scale=1.0):
    """
    The inverse of the Weibull distribution:
    F = 1 - exp[-(x/s)**c]
    x >= 0, s >= 0, c >= 1 
    """

    if c == 1.0:
        x = iexpo(prob, scale)

    else:
        _assertprob(prob, 'iweibull')
        # ---
        assert   c   >= 1.0, \
                   "shape parameter in iweibull must not be smaller than 1.0!"
        assert scale >= 0.0, "scale parameter in iweibull must not be negative!"

        x  =  scale * safepow(-safelog(1.0-prob), 1.0/c)

        #x  =  kept_within(0.0, x)  # Not really needed

    return x

# end of iweibull

# ------------------------------------------------------------------------------

def iextreme_I(prob, type='max', mu=0.0, scale=1.0):
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

    _assertprob(prob, 'iextreme_I')
    assert scale >= 0.0, "scale parameter must not be negative in iextreme_I!"

    if    type == 'max':
        x = - scale*safelog(-safelog(prob)) + mu

    elif  type == 'min':
        x =   scale*safelog(-safelog(1.0-prob)) + mu

    else:
        raise Error("iextreme_I: type must be either 'max' or 'min'")

    return x

# end of iextreme_I

# ------------------------------------------------------------------------------

def iextreme_gen(prob, type, shape, mu=0.0, scale=1.0):
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
        x = iextreme_I(prob, type, mu, scale)

    else:
        _assertprob(prob, 'iextreme_gen')
        assert scale >= 0.0

        if   type == 'max':
            x = scale*(1.0-(-safelog(prob))**shape)/shape + mu

        elif type == 'min':
            x = scale*((-safelog(1.0-prob))**shape-1.0)/shape + mu

        else:
            raise Error("iextreme_gen: type must be either 'max' or 'min'")

    return x

    # end of iextreme_gen

# ------------------------------------------------------------------------------

def ilogistic(prob, mu=0.0, scale=1.0):
    """
    The inverse of the logistic distribution:
    f = exp[-(x-m)/s] / (s*{1 + exp[-(x-m)/s]}**2)
    F = 1 / {1 + exp[-(x-m)/s]}
    x in R
    m is the mean and mode, s is a scale parameter (s >= 0)
    """

    _assertprob(prob, 'ilogistic')
    assert scale >= 0.0, "scale parameter in ilogistic must not be negative!"

    x  =  mu - scale*safelog(safediv(1.0, prob) - 1.0)

    return x

# end of ilogistic

# ------------------------------------------------------------------------------

def irayleigh(prob, sigma=1.0):
    """
    The inverse of the Rayleigh distribution:
    f = (x/s**2) * exp[-x**2/(2*s**2)]
    F = 1 - exp[-x**2/(2*s**2)]
    x, s >= 0
    """

    _assertprob(prob, 'irayleigh')
    assert sigma >= 0.0, "parameter in irayleigh must not be negative!"

    return sigma * sqrt(2.0*(-safelog(1.0-prob)))  # Will always be >= 0.0

# end of irayleigh

# ------------------------------------------------------------------------------

def ipareto(prob, lam, xm=1.0):
    """
    The inverse of the Pareto distribution: 
    f = lam * xm**lam / x**(lam+1) 
    F = 1 - (xm/x)**lam
    x in [xm, inf)
    lam > 0
    For lam < 1 all moments are infinite
    For lam < 2 all moments are infinite except for the mean
    """

    _assertprob(prob, 'ipareto')
    assert lam >  0.0, "shape parameter lambda in ipareto must be positive!"
    assert xm  >= 0.0, \
          "left support limit parameter xm must not be negative in ipareto!"

    q  =  1.0 - prob

    if q == 0.0: return float('inf')
    
    x  =  xm * safepow(q, -1.0/lam)

    x  =  kept_within(xm, x)

    return x

# end of ipareto

# ------------------------------------------------------------------------------

def ipareto_zero(prob, lam, xm=1.0):
    """
    The inverse of the Pareto distribution with the support shifted to [0, inf):
    f = lam * xm**lam / (x+xm)**(lam+1)
    F = 1 - [xm/(x+xm)]**lam
    x in [0, inf)
    lam > 0
    For lam < 1 all moments are infinite
    For lam < 2 all moments are infinite except for the mean
    """

    _assertprob(prob, 'ipareto_zero')
    assert lam > 0.0, "shape parameter lambda in ipareto_zero must be positive!"
    textxm1 = "left support limit parameter xm of unshifted in ipareto_zero"
    textxm2 = "distribution must not be negative in ipareto_zero!"
    assert xm  >= 0.0, textxm1 + textxm2

    q  =  1.0 - prob

    if q == 0.0: return float('inf')

    x  =  xm * (safepow(q, -1.0/lam) - 1.0)

    x  =  kept_within(0.0, x)

    return x

# end of ipareto_zero

# ------------------------------------------------------------------------------

def ikodlin(prob, gam, eta):
    """
    The inverse of the Kodlin distribution, aka the linear hazard rate distribution:
    f = (gam + eta*x) * exp{-[gam*x + (1/2)*eta*x**2]}
    F = 1 - exp{-[gam*x + (1/2)*eta*x**2]}
    x, gam, eta >= 0
    """

    _assertprob(prob, 'ikodlin')
    assert gam >= 0.0, "no parameters in ikodlin must be negative!"
    assert eta >= 0.0, "no parameters in ikodlin must be negative!"

    # (1/2)*eta*x**2 + gam*x + ln(1-F) = 0

    try:
        a  = 0.5*eta
        b  = gam
        c  = safelog(1.0-prob)
        x1, x2 = z2nddeg_real(a, b, c)
        x  = max(x1, x2)

    except ValueError:
        x  =  float('inf')


    x  =  kept_within(0.0, x)

    return x

# end of ikodlin

# ------------------------------------------------------------------------------

def itukeylambda_gen(prob, lam1, lam2, lam3, lam4):
    """
    The Friemer-Mudholkar-Kollia-Lin generalized Tukey-Lambda distribution.
    lam1 is a location parameter and lam2 is a scale parameter. lam3 and lam4
    are associated with the shape. lam2 must be a positive number. 
    """

    _assertprob(prob, 'itukeylambda_gen')
    # ---
    assert lam2 > 0.0, \
          "shape parameter lam2 must be a positive float in itukeylambda_gen!"

    if lam3 == 0.0:
        q3 = safelog(prob)

    else:
        q3 = (prob**lam3-1.0) / lam3

    if lam4 == 0.0:
        q4 = safelog(1.0-prob)

    else:
        q4 = ((1.0-prob)**lam4 - 1.0) / lam4

    x  =  lam1 + (q3-q4)/lam2

    return x

# end of itukeylambda_gen

# ------------------------------------------------------------------------------
# This is a department with SYMMETRICAL, stable distributions
# ------------------------------------------------------------------------------

def icauchy(prob, location=0.0, scale=1.0):
    """
    The inverse of a Cauchy distribution: f = 1 / [s*pi*(1 + [(x-l)/s]**2)]
                                          F = (1/pi)*arctan((x-l)/s) + 1/2
    (also known as the Lorentzian or Lorentz distribution)
    
    scale must be >= 0 
    """

    _assertprob(prob, 'icauchy')
    # ---
    assert scale >= 0.0, "scale parameter must not be negative in icauchy!"

    x  =  prob - 0.5

    try:
        r  =  tan(PI*x)
        r  =  scale*r + location

    except OverflowError:
        r  =  fsign(x) * float('inf')

    return r

# end of icauchy

# ------------------------------------------------------------------------------

def inormal(prob, mu=0.0, sigma=1.0):
    """
    Returns the inverse of the cumulative normal distribution function.
    Reference: Boris Moro "The Full Monte", Risk Magazine, 8(2) (February): 
    57-58, 1995, where Moro improves on the Beasley-Springer algorithm 
    (J. D. Beasley and S. G. Springer, Applied Statistics, vol. 26, 1977, 
    pp. 118-121). This is further refined by Shaw, c. f. below. 
    Max relative error is claimed to be less than 2.6e-9 
    """

    _assertprob(prob, 'inormal')

    assert sigma >= 0.0, "sigma must not be negative in inormal!"


    #a = ( 2.50662823884, -18.61500062529,  \
    #     41.39119773534, -25.44106049637)   # Moro

    #b = (-8.47351093090,  23.08336743743, \
    #    -21.06224101826,   3.13082909833)   # Moro

    # The a and b below are claimed to be better by William Shaw in a 
    # Mathematica working report: "Refinement of the Normal Quantile - 
    # A benchmark Normal quantile based on recursion, and an appraisal 
    # of the Beasley-Springer-Moro, Acklam, and Wichura (AS241) methods" 
    # (William Shaw, Financial Mathematics Group, King's College, London;  
    # william.shaw@kcl.ac.uk).
    # Max RELATIVE error is claimed to be reduced from 1.4e-8 to 2.6e-9
    # over the central region

    a = ( 2.5066282682076065359, -18.515898959450185753, \
         40.864622120467790785,  -24.820209533706798850)      # Moro/Shaw

    b = ( -8.4339736056039657294, 22.831834928541562628, \
         -20.641301545177201274,   3.0154847661978822127)     # Moro/Shaw

    c = (0.3374754822726147, 0.9761690190917186, 0.1607979714918209, \
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919, \
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187)     # Moro

    x = prob - 0.5

    if abs(x) < 0.42:   # A rational approximation for the central region...
        r  =  x * x
        r  =  x * (((a[3]*r+a[2])*r+a[1])*r+a[0]) / ((((b[3]*r+b[2])*r+\
                                                        b[1])*r+b[0])*r+1.0)
        r  =  sigma*r + mu

    else:               # ...and a polynomial for the tails
        r = prob
        if x > 0.0: r = 1.0 - prob
        try:
            r  =  safelog(-safelog(r))
            r  =  c[0] + r*(c[1] + r*(c[2] + r*(c[3] + r*(c[4] + r*(c[5] +\
                                   r*(c[6] + r*(c[7] + r*c[8])))))))
            if x < 0.0: r = -r
            r  =  sigma*r + mu
        except ValueError:
            r  =  fsign(x) * float('inf')

    return r

# end of inormal

# ------------------------------------------------------------------------------

def ilognormal(mulg, sigmalg, x):
    """
    Inverse of the lognormal distribution based on the inormal function above.
    The log10-converted form is assumed for mulg and sigmalg: 
    mulg is the mean of the log10 (and the log10 of the median) of 
    the random variate, NOT the log10 of the mean of the non-logged 
    variate!, and sigmalg is the standard deviation of the log10 of 
    the random variate, NOT the log10 of the standard deviation of 
    the non-logged variate!!
    
    sigmalg > 0.0
    """

    assert x >= 0.0, "variate must be non-negative in ilognormal!"

    return inormal(mulg, sigmalg, log10(x))

# end of ilognormal

# ------------------------------------------------------------------------------

def ifoldednormal(prob, muunfold, sigmaunfold):
    """
    The inverse of a distribution of a random variable that is the absolute 
    value of a variate drawn from the normal distribution (i. e. the 
    distribution of a variate that is the absolute value of a normal variate, 
    the latter having muunfold as its mean and sigmaunfold as its standard 
    deviation). 
    
    sigmaunfold >= 0.0
    """

    _assertprob(prob, 'ifoldednormal')
    # Everything else will be checked in cfoldednormal and dfoldednormal

   # --------------------------------------------
    def _fifi2fid(x):
        x      = kept_within(0.0, x)
        cdf    = cfoldednormal(muunfold, sigmaunfold, x)
        pdf    = dfoldednormal(muunfold, sigmaunfold, x)
        fi     = cdf - prob
        if pdf <= 0.0:
            if fi == 0.0: fi2fid = 1.0
            else:         fi2fid = MAXFLOAT
        else:
            fi2fid = fi/pdf
        return fi, fi2fid
   # -------------------------------------------

    a = abs(muunfold)
    if a/sigmaunfold > 1.0: start =  a
    else:                   start = 0.0
    x  =  znewton(_fifi2fid, start, 'ifoldednormal', tolf=SQRTMACHEPS)

    x  =  kept_within(0.0, x)
    
    return x

# end of ifoldednormal

# ------------------------------------------------------------------------------

def istable_sym(prob, alpha, location=0.0, scale=1.0):
    """
    The inverse of a SYMMETRICAL stable distribution where alpha is the tail 
    exponent. For numerical reasons alpha is restricted to [0.25, 0.9] and 
    [1.125, 1.9] - but alpha = 1.0 (the Cauchy) and alpha = 2.0 (scaled normal) 
    are allowed!

    Numerics are somewhat crude but the fractional error is < 0.001 - and the
    absolute error is almost always < 0.001 - sometimes much less... 

    NB This function is slow, particularly for small alpha !!!!!
    """

    _assertprob(prob, 'istable_sym')
    # Everything else will be checked in cstable_sym and dstable_sym!

    if alpha == 1.0: return icauchy(prob, location, scale)
    if alpha == 2.0: return inormal(prob, location, SQRT2*scale)

    if prob == 0.0: return -float('-inf')
    if 0.5-MACHEPS <= prob and prob <= 0.5+MACHEPS: return location
    if prob == 1.0: return  float('inf')

   # -----------------------------------------------
    def _fifi2fid(x):
        cdf  = cstable_sym(alpha, location, scale, x)
        pdf  = dstable_sym(alpha, location, scale, x)
        fi   = cdf - prob
        if pdf <= 0.0:
            if fi == 0.0: fi2fid = 1.0
            else:         fi2fid = MAXFLOAT
        else:
            fi2fid = fi/pdf
        return fi, fi2fid
   # -----------------------------------------------

    tolr = 4.8828125e-4  # = 0.5**11 - no reason to spend excess. accuracy here!
    x = znewton(_fifi2fid, location, 'istable_sym', tolf=tolr, tola=MACHEPS)

    return x

# end of istable_sym

# ------------------------------------------------------------------------------
# This is a department with ASYMMETRIC, stable distributions
# ------------------------------------------------------------------------------

def ilevy(prob, scale=1.0):
    """
    The inverse of the Levy distribution (stable distribution 
    with alpha = 1/2 and beta = 1, aka the Cournot distribution). 
    This is actually the right-skewed Levy!
    f = sqrt(s/2pi) * (1/x)**(3/2) * exp(-s/2x)
    F = erfc(sqrt(s/2x))
    s >= 0.0, x >= 0

    Function is based on inormal.
    """

    _assertprob(prob, 'ilevy')
    # ---
    assert scale >= 0.0, "scale parameter must not be negative in ilevy!"

    if prob == 0.0:
        x = 0.0

    elif prob == 1.0:
        x = float('inf')

    else:
        ph = 0.5*prob
        x  = inormal(ph)
        x  = scale / (x*x)

    return x

# end of ilevy

# ------------------------------------------------------------------------------
# Auxiliary function:
# ------------------------------------------------------------------------------

def _assertprob(prob, caller='caller'):
    assert 0.0 <= prob and prob <= 1.0, \
            "input probability must be within [0.0, 1.0] in " + caller + "!"

# ------------------------------------------------------------------------------
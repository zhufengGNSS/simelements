# statlib/pdf.py
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
MODULE WITH FUNCTIONS FOR THE PDF OF VARIOUS PROBABILITY DISTRIBUTIONS. 
NB. Some functions may return float('inf') !
"""
# ------------------------------------------------------------------------------

from math import exp, sqrt, log, log10, sin, cos

from numlib.specfunc   import lnfactorial, lngamma, beta
from numlib.solveq     import zbrent
from numlib.quadrature import qromberg
from numlib.talbot     import talbot
from numlib.miscnum    import fsign
from machdep.machnum   import MACHEPS, TWOMACHEPS
from misclib.numbers   import is_nonneginteger, is_posinteger
from misclib.numbers   import kept_within, ERRCODE
from misclib.errwarn   import Error
from misclib.mathconst import PI, PIHALF, TWOPI, PIINV, SQRT2, SQRTTWOPI, LOGE

# ------------------------------------------------------------------------------

def dunifab(left, right, x):
    """
    The pdf of the uniform distribution with support on [left, right]. 
    """

    # Input check -----------------
    assert right > left, "support range must be positive in dunifab!"
    assert left <= x and x <= right, \
                    "variate must be within support range in dunifab!"
    # -----------------------------

    pdf  =  1.0 / (right-left)   # Will always be >= 0.0

    return pdf

# end of dunifab

# ------------------------------------------------------------------------------

def dtriang(left, mode, right, x):
    """
    The pdf of the triangular distribution with support 
    on [left, right] and with mode 'mode'. 
    """

    # Input check -----------------------
    assert right > left, "support range must be positive in dtriang!"
    assert left <= mode and mode <= right, \
                    "mode must be within support range in dtriang!"
    assert left <=  x   and   x  <= right, \
                    "variate must be within support range in dtriang!"
    # -----------------------------------

    spant = right - left
    spanl = mode  - left
    spanr = right - mode

    if spanr == 0.0:
        pdf  =  2.0 * (x-left) / float((spant*spanl))

    elif spanl == 0.0:
        pdf  =  2.0 * (right-x) / float((spant*spanr))

    elif x <= mode:
        pdf  =  2.0 * (x-left) / float((spant*spanl))

    else:
        pdf  =  2.0 * (right-x) / float((spant*spanr))


    pdf  = kept_within(0.0, pdf)
    
    return pdf

# end of dtriang

# ------------------------------------------------------------------------------

def dtri_unif_tri(a, b, c, d, x):
    """
    The pdf of the triangular-uniform-triangular distribution with 
    support on [a, d] and with break points in b and c.
              ------
    pdf:    /        \
           /           \
    ------              -------
    """

    # Input check -----------------------
    assert d > a, "support range must be positive in dtri_uinf_tri!"
    assert a <= b and b <= c and c <= d, \
         "break points must in order and within support range in dtri_unif_tri!"
    assert a <= x and x <= d, \
                  "variate must be within support range in dtri_unif_tri!"
    # -----------------------------------


    if c == b:
        pdf = dtriang(a, b, d)

    else:
        h = 2.0 / (d+c-b-a)
        if   b <= x <= c:  pdf  =  h
        elif a <= x <  b:  pdf  =  h * (x-a) / (b-a)
        elif c <  x <= d:  pdf  =  h * (1.0 - (x-c)/(d-c))


    pdf  = kept_within(0.0, pdf)
    
    return pdf

# end of dtri_unif_tri

# ------------------------------------------------------------------------------

def dbeta(a, b, x1, x2, x):
    """
    The pdf of the beta distribution:
    f = x**(a-1) * (1-x)**(b-1) / beta(a, b)
    a, b >= 0; 0 <= x <= 1
    F is the integral = the incomplete beta or the incomplete beta / complete 
    beta depending on how the incomplete beta function is defined.
    
    x2 > x1  !!!!
    
    NB  dbeta may return float('inf') for a or b < 1.0!
    """

    assert a  >= 0.0, "both parameters must be non-negative in dbeta!"
    assert b  >= 0.0, "both parameters must be non-negative in dbeta!"
    assert x2 >  x1,  "support range must be positive in dbeta!"
    assert x1 <= x and x <= x2, "variate must be within support range in dbeta!"

    if a == 1.0 and b == 1.0: return dunifab(x1, x2, x)

    c = 1.0 / (x2-x1)
    y = c * (x-x1)

    if a < 1.0 and y <= 0.0:
        pdf = float('inf')

    elif b < 1.0 and y >= 1.0:
        pdf = float('inf')

    else:
        pdf   = c * pow(y, a-1.0) * pow(1.0-y, b-1.0) / betaf

    pdf = kept_within(0.0, pdf)

    return pdf

# end of dbeta

# ------------------------------------------------------------------------------

def dkumaraswamy(a, b, x1, x2, x):
    """
    The pdf of the Kumaraswamy distribution:
    f = a*b*x**(a-1) * (1-x**a)**(b-1)
    F = 1 - (1-x**a)**b
    a, b >= 0; 0 <= x <= 1
    The Kumaraswamy distribution is similar to the beta distribution !!!
    
    x2 > x1  !!!!

    NB  dkumaraswamy may return float('inf') for a or b < 1.0!
    """

    assert a  >= 0.0, "both parameters must be non-negative in dkumaraswamy!"
    assert b  >= 0.0, "both parameters must be non-negative in dkumaraswamy!"
    assert x2 >  x1,  "support range must be positive in dkumaraswamy!"
    assert x1 <= x and x <= x2, \
               "variate must be within support range in dkumaraswamy!"

    c  = 1.0 / (x2-x1)
    y  =  c * (x-x1)

    if a < 1.0 and y <= 0.0:
        pdf = float('inf')

    elif b < 1.0 and y >= 1.0:
        pdf = float('inf')

    else:
        pdf = c * a * b * y**(a-1.0) * (1.0-y**a)**(b-1.0)

    pdf = kept_within(0.0, pdf)

    return pdf

# end of dkumaraswamy

# ------------------------------------------------------------------------------

def dsinus(left, right, x):
    """
    The pdf of the "sinus distribution" with support on [left, right]. 
    """

    assert right > left,     "support range must be a positive float in dsinus!"
    assert left <= x <= right, "variate must be within support range in dsinus!"

    const1 = PI / (right-left)
    const2 = 0.5 * const1
    pdf    = const2 * sin(const1*(x-left))

    pdf   = kept_within(0.0, pdf)

    return pdf

# end of dsinus

# ------------------------------------------------------------------------------
 
def dgeometric(phi, k):
    """
    The geometric distribution with
    p(K=k) = phi * (1-phi)**(k-1)  and 
    P(K>=k) = sum phi * (1-phi)**k = 1 - q**k 
    where q = 1 - phi and  0 < phi <= 1 is the success frequency or 
    "Bernoulli probability" and K >= 1 is the number of  trials to 
    the first success in a series of Bernoulli trials. 
    
    It is easy to prove that P(k) = 1 - (1-phi)**k: 
        Let q = 1 - phi. p(k) = (1-q) * q**(k-1) = q**(k-1) - q**k 
        Then P(1) = p(1) = 1 - q. 
        P(2) = p(1) + p(2) = 1 - q + q - q**2 = 1 - q**2 
        Induction can be used to show that P(k) = 1 - q**k = 1 - (1-phi)**k 
        """

    assert 0.0 < phi and phi <= 1.0, \
                      "Success frequency must be in (0.0, 1.0] in dgeometric!"
    assert is_posinteger(k), \
                   "Number of trials must be a positive integer i dgeometric!"

    pdf  =  phi * (1.0-phi)**(k-1)

    return pdf

# end of dgeometric

# ------------------------------------------------------------------------------

def dpoisson(lam, tspan, n):
    """
    The Poisson distribution: p(N=n) = exp(-lam*tspan) * (lam*tspan)**n / n!
    n = 0, 1,...., inf
    """

    # Input check -----------
    assert  lam  >= 0.0, "Poisson rate must not be negative in dpoisson!"
    assert tspan >= 0.0, "time span must not be negative in dpoisson!"
    assert is_nonneginteger(n), \
                " must be a non-negative integer in dpoisson!"
    # -----------------------

    lamtau = lam*tspan
    ln  =  lamtau + lngamma(n+1) - n*log(lamtau)
    ln  =  kept_within(0.0, ln)
    pdf =  exp(-ln)    # Will always be >= 0.0

    return pdf

# end of dpoisson

# ------------------------------------------------------------------------------

def dexpo(mean, x):
    """
    The pdf of the exponential distribution with mean = 1/lambda (mean >= 0.0).
    """

    # Input check ----
    assert mean >  0.0, "mean must be positive in dexpo!"
    assert x    >= 0.0, "variate must not be negative in dexpo!"
    # ----------------

    pdf  =  exp(-x/mean) / mean   # Will always be >= 0

    return pdf

# end of dexpo

# ------------------------------------------------------------------------------

def dhyperexpo(means, pcumul, x):
    """
    The pdf of the hyperexponential distribution: 
    f = sumk pk * exp(x/mk) / mk
    F = sumk pk * (1-exp(x/mk))
    
    NB Input to the function is the list of CUMULATIVE PROBABILITIES ! 
    """

    lm = len(means)
    lp = len(pcumul)

    # Input check -----------------
    assert  x  >= 0.0, "variate must not be negative in dhyperexpo!"
    assert lp == lm, \
         "number of means must be equal to the number of pcumuls in dhyperexpo!"
    errortextm = "all means must be positive floats in dhyperexpo!"
    errortextp = "pcumul list is not in order in dhyperexpo!"
    assert means[0] > 0.0,                       errortextm
    assert pcumul[-1]  == 1.0,                   errortextp
    if lm == 1: return dexpo(means[0], x)
    assert 0.0 < pcumul[0] and pcumul[0] <= 1.0, errortextp
    # -----------------------------

    summ    = pcumul[0] * exp(-x/means[0]) / means[0]
    nvalues = lp
    for k in range (1, nvalues):
        assert means[k]  >  0.0, errortextm
        pdiff = pcumul[k] - pcumul[k-1]
        assert pdiff >= 0.0, errortextp
        summ += pdiff * exp(-x/means[k]) / means[k]
    pdf = summ

    return pdf

# end of dhyperexpo

# ------------------------------------------------------------------------------

def dNexpo(means, x):
    """
    The pdf of the distribution of a sum of exponentially distributed 
    random variables. 
    
    NB Means are allowed to be equal (but the function is slow)!!! 
    """

    assert x >= 0.0, "variate must not be negative in dNexpo!"

    number = len(means)

    if number == 1: return dexpo(means[0], x)

    if x == 0.0:
        if number == 1:  return 1.0/means[0]
        else:            return 0.0

    lam  = []
    for k in range(0, number):
        assert means[k] > 0.0, "All means must be positive floats in dNexpo!"
        lam.append(1.0/means[k])

    # ----------------------------------------------------
    def _ftilde(z):
        zprod = complex(1.0)
        for k in range(0, number):
            zprod  =  zprod * complex(lam[k]) / (z+lam[k])
        return zprod
    # ----------------------------------------------------

    rpole = - min(lam)
    sigma =  (1.0-TWOMACHEPS) * rpole
    pdf   =  talbot(_ftilde, x, sigma)

    pdf   =  kept_within(0.0, pdf)

    return pdf

# end of dNexpo

# ------------------------------------------------------------------------------

def dNexpo2(means, x):
    """
    The pdf of the distribution of a sum of exponentially distributed 
    random variables.
    
    NB No two means are allowed to be equal!!!!! 
    """

    assert x >= 0.0, "variate must not be negative in dNexpo2!"

    number = len(means)

    if number == 1: return dexpo(means[0], x)

    lam     = []
    exps    = []
    product = 1.0
    for k in range(0, number):
        assert means[k] > 0.0, "All means must be positive floats in dNexpo2!"
        lam.append(1.0/means[k])
        exps.append(exp(-lam[k]*x))
        product = product*lam[k]

    divisor = []
    for k in range(0, number):
        divisor.append(1.0)
        for j in range(0, number):
            if j != k: divisor[k] = divisor[k]*(lam[j]-lam[k])

    pdf = 0.0
    try:
        for k in range(0, number):
            pdf += exps[k] / divisor[k]
        pdf  = product*pdf
    except (ZeroDivisionError, OverflowError):
        raise OverflowError("means too close in dNexpo2!")

    pdf = kept_within(0.0, pdf)

    return pdf

# end of dNexpo2

# ------------------------------------------------------------------------------

def dexpo_gen(a, b, c, x):
    """
    The pdf of the generalized continuous exponential distribution (x in R):
    x <= c: f  =  [a*b/(a+b)] * exp(+a*[x-c])
            F  =   [b/(a+b)]  * exp(+a*[x-c])
    x >= c: f  =  [a*b/(a+b)] * exp(-b*[x-c])
            F  =  1 - [a/(a+b)]*exp(-b*[x-c])
    a > 0, b > 0
    
    NB The symmetrical double-sided exponential sits in dlaplace!
    """

    assert a > 0.0, "'a' parameter must be positive in dexpo_gen!"
    assert b > 0.0, "'b' parameter must be positive in dexpo_gen!"

    if x <= c:  pdf  =  (a*b/(a+b)) * exp( a*(x-c))
    else:       pdf  =  (a*b/(a+b)) * exp(-b*(x-c))

    #pdf = kept_within(0.0, pdf)

    return pdf

# end of dexpo_gen

# ------------------------------------------------------------------------------

def derlang(nshape, phasemean, x):
    """
    The pdf of the Erlang distribution - represents the sum of 
    nshape exponentially distributed random variables all having 
    the same mean = phasemean. 
    """

    if nshape == 1:
        pdf = dexpo(phasemean, x)

    else:
        # Input check -----------------
        assert is_posinteger(nshape), \
                  "shape parameter must be a positive integer in derlang!"
        assert  phasemean  >  0.0, \
                           "phase mean must be positive in derlang!"
        assert      x      >= 0.0, \
                           "variate must not be negative in derlang!"
        # -----------------------------

        y        =  x / float(phasemean)
        nshapem1 = nshape - 1
        try:
            pdf  =  exp(-y) * y**(nshape-1)
        except OverflowError:
            pdf  =  exp(-y + nshapem1*log(y))
        fact =  1
        for k in range(1, nshape): fact = fact*k  # nshape assumed to be small
        factor = 1.0 / (phasemean*fact)           # Now floats
        pdf    = factor * pdf           # Will always be >= 0.0

    return pdf

# end of derlang

# ------------------------------------------------------------------------------

def derlang_gen(nshapes, pcumul, phasemean, x):
    """
    The pdf of the generalized Erlang distribution - the Erlang equivalent 
    of the hyperexpo distribution:
    f = sumk pk * ferlang(m, nk) 
    F = sumk pk * Ferlang(m, nk) 
    with the same mean for all phases.
    
    NB Input to the function is the list of CUMULATIVE PROBABILITIES ! 
    """

    ln = len(nshapes)
    lp = len(pcumul)

    # Input check -----------------
    assert  x  >= 0.0, "variate must not be negative in derlang_gen!"
    assert lp == ln, \
       "number of shapes must be equal to the number of pcumuls in derlang_gen!"
    if ln == 1: return cerlang(nshapes[0], mean, x)
    errortextn = "all nshapes must be positiva heltal i derlang_gen!"
    errortextm = "the mean must be a positive float in derlang_gen!"
    errortextp = "pcumul list is not in order in derlang_gen!"
    for n in nshapes: assert is_posinteger(n),   errortextn
    assert phasemean > 0.0,                      errortextm
    assert pcumul[-1]  == 1.0,                   errortextp
    assert 0.0 < pcumul[0] and pcumul[0] <= 1.0, errortextp
    # -----------------------------

    summ    = pcumul[0] * derlang(nshapes[0], phasemean, x)
    nvalues = lp
    for k in range (1, nvalues):
        pdiff = pcumul[k] - pcumul[k-1]
        assert pdiff >= 0.0, errortextp
        summ += pdiff * derlang(nshapes[k], phasemean, x)
    pdf = summ

    return pdf

# end of derlang_gen

# ------------------------------------------------------------------------------

def dcoxian(means, probs, x):
    """
    The pdf of the Coxian phased distribution, which is based on the 
    exponential. probs is a list of probabilities for GOING ON TO THE 
    NEXT PHASE rather than reaching the absorbing state prematurely. 
    The number of means must (of course) be one more than the number 
    of probabilities! 
    
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
                         "lengths of input lists are not matched in dcoxian!"

    if lm == 2:
        assert means[0] >= 0.0 and means[1] >= 0.0, \
                                 "all means must be non-negative in dcoxian!"
        assert 0.0 <= probsl[0] and probsl[0] <= 1.0, \
                       "probabilities must be within 0.0 and 1.0 in dcoxian!"
        pdf = (1.0-probsl[0])*dexpo(means[0], x) + probsl[0]*dNexpo(means, x)

    else:
        sub      = lp*[1.0]
        for k in range(1, lp): sub[k] = sub[k-1]*probsl[k-1]

        freq     = lm*[0.0]
        freq[0]  = 1.0 - probsl[0]
        freq[-1] = sub[-1]*probsl[-1]
        for k in range(1, lp): freq[k] = sub[k]*(1-probsl[k])

        summ     = freq[0] * dexpo(means[0], x)
        for k in range(1, lm):
            summ += freq[k] * dNexpo(means[0:(k+1)], x)
        pdf = summ


    pdf = kept_within(0.0, pdf)

    return pdf

# end of dcoxian

# ------------------------------------------------------------------------------

def dcoxian2(means, probs, x):
    """
    The pdf of the Coxian phased distribution, which is based on the 
    exponential. probs is a list of probabilities for GOING ON TO THE 
    NEXT PHASE rather than reaching the absorbing state prematurely. 
    The number of means must (of course) be one more than the number 
    of probabilities! 
    
    NB No two means[k] must be equal - if this is desired, use dcoxian 
    instead (slower, however). 
    """

    lm = len(means)
    try:
        lp     = len(probs)
        probsl = probs
    except TypeError:   #  probs is not provided as a list
        probsl = [probs]
        lp     = len(probsl)

    assert lm == lp + 1, \
                       "lengths of input lists are not matched in dcoxian2!"

    if lm == 2:
        assert means[0] >= 0.0 and means[1] >= 0.0, \
                               "all means must be non-negative in dcoxian2!"
        assert 0.0 <= probsl[0] and probsl[0] <= 1.0, \
                     "probabilities must be within 0.0 and 1.0 in dcoxian2!"
        pdf = (1.0-probsl[0])*dexpo(means[0], x) + probsl[0]*dNexpo2(means, x)

    else:
        sub      = lp*[1.0]
        for k in range(1, lp): sub[k] = sub[k-1]*probsl[k-1]

        freq     = lm*[0.0]
        freq[0]  = 1.0 - probsl[0]
        freq[-1] = sub[-1]*probsl[-1]
        for k in range(1, lp): freq[k] = sub[k]*(1-probsl[k])

        summ     = freq[0] * dexpo(means[0], x)
        for k in range(1, lm):
            summ += freq[k] * dNexpo2(means[0:(k+1)], x)
        pdf = summ


    pdf = kept_within(0.0, pdf)

    return pdf

# end of dcoxian2

# ------------------------------------------------------------------------------

def dkodlin(gam, eta, x):
    """
    The pdf of the Kodlin distribution, aka the linear hazard rate distribution:
    f = (gam + eta*x) * exp{-[gam*x + (1/2)*eta*x**2]} 
    F = 1 - exp{-[gam*x + (1/2)*eta*x**2]} 
    x, gam, eta >= 0
    """

    assert gam >= 0.0, "'gam' parameter must not be negative in dkodlin!"
    assert eta >= 0.0, "'eta' parameter must not be negative in dkodlin!"
    assert  x  >= 0.0, "variate must not be negative i dkodlin!"

    etax  =  eta * x
    pdf   =  (gam + etax) * exp(-(gam + 0.5*etax)*x)  # Will always be >= 0.0

    return pdf

# end of dkodlin

# ------------------------------------------------------------------------------

def dlaplace(loc, scale, x):
    """
    The pdf of the Laplace distribution:
    f = [(1/2)/s)] * exp(-abs([x-l]/s))
    F = (1/2)*exp([x-l]/s)  {x <= 0},  F = 1 - (1/2)*exp(-[x-l]/s)  {x >= 0}
    s > 0
    """

    assert scale > 0.0, "scale must be positive in dlaplace!"

    pdf  =  0.5 * exp(-abs((x-loc)/float(scale))) / scale
    # Will always be >= 0.0

    return pdf

# end of dlaplace

# ------------------------------------------------------------------------------

def dexppower(loc, scale, alpha, x, lngam1oalpha=False):
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
    """

    assert scale > 0.0, \
               "scale parameter must be a positive float in dexppower!"
    assert alpha > 0.0, \
            "shape parameter alpha must be a positive float in dexppower!"

    if alpha == 1.0: return dlaplace(loc, scale, x)

    sinv = 1.0/float(scale)

    aux1 = - (sinv*abs(x-loc)) ** alpha
    if not lngam1oalpha: aux2 = lngamma(1.0/float(alpha))
    else:                aux2 = lngam1oalpha

    pdf  =  0.5*sinv*alpha * exp(aux1-aux2)   # Will always be >= 0

    return pdf

# end of dexppower

# ------------------------------------------------------------------------------

def dgamma(alpha, lam, x, lngamalpha=False):
    """
    The gamma distrib. f = lam * exp(-lam*x) * (lam*x)**(alpha-1) / gamma(alpha)
    F is the integral = the incomplete gamma or the incomplete gamma / complete 
    gamma depending on how the incomplete gamma function is defined.
    x, lam, alpha >= 0
    
    NB It is possible to gain efficiency by providing the value of the 
    natural logarithm of the complete gamma function ln(gamma(alpha)) 
    as a pre-computed input (may be computed using numlib.specfunc.lngamma) 
    instead of the default 'False'.

    NB  dgamma may return float('inf') for alpha < 1.0!
    """

    assert alpha >= 0.0, "alpha must not be negative in dgamma!"
    assert  lam  >= 0.0, "lambda must not be negative i dgamma!"
    assert   x   >= 0.0, "variate must not be negative in dgamma!"

    lamx  = lam * x
    if lngamalpha: lga = lngamalpha
    else:          lga = lngamma(alpha)

    if alpha < 1.0:
        if lamx == 0.0:
            pdf  = float('inf')
        else:
            alpham1 = alpha - 1.0
            try:
                pdf  = lam * exp(-lamx-lga) / pow(lamx, -alpham1)
            except OverflowError:
                pdf  = lam * exp(-lamx + alpham1*log(lamx) - lga)

    else:
        alpham1 = alpha - 1.0
        try:                  pdf  = lam * exp(-lamx-lga) * pow(lamx, alpham1)
        except OverflowError: pdf  = lam * exp(-lamx + alpham1*log(lamx) - lga)

    return pdf    # Will always be >= 0.0

# end of dgamma

# ------------------------------------------------------------------------------

def dpareto(lam, xm, x):
    """
    The pdf of the Pareto distribution:
    f = lam * xm**lam / x**(lam+1)
    F = 1 - (xm/x)**lam
    x in [xm, inf)
    lam >= 0
    For lam < 1 all moments are infinite
    For lam < 2 all moments are infinite except for the mean
    """

    assert lam >= 0.0, "lambda must not be negative in dpareto!"
    assert xm  >= 0.0, "lower limit must not be negative in dpareto!"
    assert x   >= xm, "variate must not be smaller than lower limit in dpareto!"

    pdf  =  lam * xm**lam / x**(lam+1.0)

    return pdf

# end of dpareto

# ------------------------------------------------------------------------------

def dpareto_zero(lam, xm, x):
    """
    The pdf of the Pareto distribution with the support shifted to [0, inf):
    f = lam * xm**lam / (x+xm)**(lam+1)
    F = 1 - [xm/(x+xm)]**lam
    x in [0, inf)
    lam > 0
    For lam < 1 all moments are infinite
    For lam < 2 all moments are infinite except for the mean
    """

    assert lam >= 0.0, "lambda must not be negative in dpareto_zero!"
    assert xm  >= 0.0, "'xm' must not be negative in dpareto_zero!"
    assert x   >= 0.0, "variate must not be negative in dpareto_zero!"

    pdf  =  lam * xm**lam / (x+xm)**(lam+1.0)

    return pdf

# end of dpareto_zero

# ------------------------------------------------------------------------------

def drayleigh(sigma, x):
    """
    The pdf of the Rayleigh distribution:
    f = (x/s**2) * exp[-x**2/(2*s**2)]
    F = 1 - exp[-x**2/(2*s**2)]
    x >= 0
    """

    assert sigma != 0.0, "sigma must not be 0 in drayleigh!"
    assert   x   >= 0.0, "variate must not be negative in drayleigh!"

    a   =  x / float(sigma**2)
    b   =  0.5 * x * b
    pdf =  a * exp(-b)

    return pdf

# end of drayleigh

# ------------------------------------------------------------------------------

def dweibull(c, scale, x):
    """
    The pdf of the Weibull distribution:
    f = exp[-(x/s)**(c-1)] / s
    F = 1 - exp[-(x/s)**c]
    x >= 0, s > 0, c >= 1 
    """

    if c == 1.0:
        pdf = dexpo(prob, scale, x)

    else:
        assert   c   >= 1.0, "shape parameter 'c' must be >= 1.0 in dweibull!"
        assert scale >  0.0, "scale must be positive in dweibull!"
        assert   x   >= 0.0, "variate must not be negative in dweibull!"

        floatscale = float(scale)
        pdf = exp(-(x/floatscale)**(c-1)) / floatscale  # Will always be >= 0.0

    return pdf

# end of dweibull

# ------------------------------------------------------------------------------

def dextreme_I(type, mu, scale, x):
    """
    The pdf of the extreme value distribution type I 
    (aka the Gumbel distribution or Gumbel distribution type I):
    F = exp{-exp[-(x-mu)/scale]}       (max variant)
    f = exp[-(x-mu)/scale] * exp{-exp[-(x-mu)/scale]} / scale
    F = 1 - exp{-exp[+(x-mu)/scale]}   (min variant)
    f = exp[+(x-mu)/scale] * exp{-exp[+(x-mu)/scale]} / scale
    
    type must be 'max' or 'min'
    scale must be > 0.0
    """

    assert scale > 0.0, "scale must be positive in dextreme_I!"

    if   type == 'max':
        fscale = float(scale)
        h      =  exp(-(x-mu)/fscale)
        g      =  exp(-h)
        pdf    =  h * g / fscale
    elif type == 'min':
        fscale = float(scale)
        h      =  exp((x-mu)/fscale)
        g      =  exp(-h)
        pdf    =  h * g / fscale
    else:
        raise Error("type must be either 'max' or 'min' in cextreme_I!")

    pdf = kept_within(0.0, pdf)

    return pdf

# end of dextreme_I

# ------------------------------------------------------------------------------

def dextreme_gen(type, shape, mu, scale, x):
    """
    The pdf of the generalized extreme value distribution:

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
        pdf = dextreme_I(type, mu, scale, x)
    
    else:
        assert scale > 0.0, "scale must be positive in dextreme_gen!"

        if type == 'max':
            fscale = float(scale)
            epahs  = 1.0 / shape
            crucial = 1.0 - shape*(x-mu)/float(scale)
            if crucial <= 0.0 and shape < 0.0:
                pdf =  0.0
            else:
                g   =  exp(-crucial**epahs)
                h   =  crucial**(epahs-1.0)
                pdf =  h * g / scale

        elif type == 'min':
            fscale = float(scale)
            epahs  = 1.0 / shape
            crucial = 1.0 + shape*(x-mu)/float(scale)
            if crucial <= 0.0 and shape < 0.0:
                pdf =  0.0
            else:
                g   =  exp(-crucial**epahs)
                h   =  crucial**(epahs-1.0)
                pdf =  h * g / scale

        else:
            raise Error("type must be either 'max' or 'min' in dextreme_gen!")


    pdf = kept_within(0.0, pdf)

    return pdf

# end of dextreme_gen

# ------------------------------------------------------------------------------

def dlogistic(mu, scale, x):
    """
    The logistic distribution:
    f = exp[-(x-m)/s] / (s*{1 + exp[-(x-m)/s]}**2)
    F = 1 / {1 + exp[-(x-m)/s]}
    x in R
    m is the mean and mode, s is a scale parameter (s > 0)
    """

    assert scale > 0.0, "scale must be positive in dlogistic!"

    num = exp(-(x-mu)/float(scale))

    pdf = num / (scale*(1.0+num)**2)

    pdf = kept_within(0.0, pdf)

    return pdf

# end of dlogistic

# ------------------------------------------------------------------------------
# This is a department with SYMMETRICAL, stable distributions
# ------------------------------------------------------------------------------

def dcauchy(location, scale, x):
    """
    The pdf of the Cauchy distribution (also known 
    as the Lorentzian or Lorentz distribution):
    f = 1 / [s*pi*(1 + [(x-m)/s]**2)]
    F = (1/pi)*arctan((x-m)/s) + 1/2
    scale > 0.0 
    """

    # Input check -----------------
    assert scale > 0.0, "scale parameter must be > 0.0 in dcauchy!"
    # -----------------------------

    fscale = float(scale)
    x      =  (x-location) / fscale
    d1     =  fscale * PI
    d2     =  1.0 + x*x

    pdf =  1.0 / (d1*d2)

    return pdf

# end of dcauchy

# ------------------------------------------------------------------------------

def dnormal(mu, sigma, x):
    """
    The pdf of the pdf the normal (Gaussian) distribution. 
    sigma must be > 0.0
    """

    # Input check -----------------
    assert sigma > 0.0, "sigma must be positive in dnormal!"
    # -----------------------------

    fsigma = float(sigma)
    x      = (x-mu) / fsigma
    d      = SQRTTWOPI * fsigma
    n      = exp(-0.5*x*x)

    pdf =  n / d

    return pdf

# end of dnormal

# ------------------------------------------------------------------------------

def dlognormal(mulg, sigmalg, x):
    """
    pdf for the lognormal distribution based on the dnormal function above.
    The log10-converted form is assumed for mulg and sigmalg: 
    mulg is the mean of the log10 (and the log10 of the median) of 
    the random variate, NOT the log10 of the mean of the non-logged 
    variate!, and sigmalg is the standard deviation of the log10 of 
    the random variate, NOT the log10 of the standard deviation of 
    the non-logged variate!!
    
    sigmalg > 0.0
    """

    assert x >= 0.0, "variate must be non-negative in dlognormal!"

    try: 
        return LOGE * dnormal(mulg, sigmalg, log10(x)) / x
    except ValueError:
        return 0.0

# end of dlognormal

# ------------------------------------------------------------------------------

def dfoldednormal(muunfold, sigmaunfold, x):
    """
    The pdf of a random variable that is the absolute value of a variate drawn 
    from the normal distribution (i. e. the distribution of a variate that is 
    the absolute value of a normal variate, the latter having muunfold as its 
    mean and sigmaunfold as its standard deviation). 
    """

    # sigma > 0.0 # assertion i dnormal
    assert x >= 0.0, "variate must be a positive float in dfoldednormal!"

    pdf = dnormal(muunfold, sigmaunfold, x) + dnormal(muunfold, sigmaunfold, -x)

    return pdf

# end of cfoldednormal

# ------------------------------------------------------------------------------

def dstable_sym(alpha, location, scale, x):
    """
    The pdf of a SYMMETRICAL stable distribution where alpha is the tail 
    exponent. For numerical reasons alpha is restricted to [0.1, 0.9] and 
    [1.125, 1.9] - but alpha = 1.0 (the Cauchy) and alpha = 2.0 (scaled 
    normal) are also allowed!

    Numerics are somewhat crude but the fractional error is mostly < 0.001 - 
    sometimes much less - and the absolute error is almost always < 0.001 - 
    sometimes much less... 

    NB This function is somewhat slow, particularly for small alpha !!!!!
    """

    # NB Do not change the numerical parameters - they are matched! The 
    # corresponding cdf function cstable_sym is partly based on this 
    # function so changes in one of them are likely to require changes 
    # in the others!

    assert 0.1 <= alpha and alpha <= 2.0,  \
                           "alpha must be in [0.1, 2.0] in dstable_sym!"
    if alpha < 1.0: assert alpha <= 0.9,   \
                           "alpha <= 1.0 must be <= 0.9 in dstable_sym!"
    if alpha > 1.0: assert alpha >= 1.125, \
                          "alpha > 1.0 must be >= 1.125 in dstable_sym!"
    if alpha > 1.9: assert alpha == 2.0,   \
                             "alpha > 1.9 must be = 2.0 in dstable_sym!"
    assert scale > 0.0, "scale must be a positive float in dstable_sym!"

    if alpha == 1.0: return dcauchy(location, scale, x)
    if alpha == 2.0: return dnormal(location, SQRT2*scale, x)

    x  =  (x-location) / float(scale)
    x  =  abs(x)

    # Compute the value at the peak/mode (for x = 0) for later use:
    #peak = exp(lngamma(1.0+1.0/alpha)) / PI
    peak = PIINV * exp(lngamma(1.0+1.0/alpha))

    if alpha < 1.0:
        # For sufficiently small abs(x) the value at the peak is used 
        # (heuristically; based on experimentation). For x >= 1.0 the 
        # series expansion for large x due to Bergstrom is used. For x 
        # "in between" the integral formulation is used:
        if alpha <= 0.25:
            point = 0.5**37
        else:
            point = 1.25 * pow(10.0, -4.449612602 + 3.078368893*alpha)
        if x <= point:
            pdf = peak
        elif x >= 1.0:
            pdf = _dstable_sym_big(alpha, x, MACHEPS)
        else:
            pdf = _dstable_sym_int(alpha, x, 0.5**17, 17)

    elif alpha > 1.0:
        # For sufficiently small abs(x) the series expansion for small x due 
        # to Bergstrom is used. For x sufficiently large x an asymptotic 
        # expression is used. For x "in between" the integral formulation 
        # is used (all limits heuristically based):
        y1 = -2.212502985 + alpha*(3.03077875081 - alpha*0.742811132)
        if x <= pow(10.0, y1):
            pdf = _dstable_sym_small(alpha, x, MACHEPS)
        else:
            pdf = _dstable_sym_int(alpha, x, 0.5**19, 21)

    pdf = kept_within(0.0, pdf, peak)
    return pdf

# end of dstable_sym

# -------------------------------------

def _dstable_sym_small(alpha, x, tolr):
    """
    A series expansion for small x due to Bergstrom. 
    Converges for x < 1.0 and in practice also for 
    somewhat larger x. 
    
    The function uses the Kahan summation procedure 
    (cf. Dahlquist, Bjorck & Anderson). 
    """

    summ   =  0.0
    c      =  0.0
    fact   = -1.0
    xx     =  x*x
    xpart  =  1.0
    k      =   0
    zero2  =  zero1  =  False
    while True:
        k       +=  1
        summo    =  summ
        twokm1   =  2*k - 1
        twokm1oa =  float(twokm1)/alpha
        r        =  lngamma(twokm1oa) - lnfactorial(twokm1)
        term     =  twokm1 * exp(r) * xpart
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
    pdf   =  summ / (PI*alpha)

    pdf   =  kept_within(0.0, pdf)
    return pdf

# end of _dstable_sym_small

# -----------------------------------

def _dstable_sym_big(alpha, x, tolr):
    """
    A series expansion for large x due to Bergstrom. 
    Converges for x > 1.0
    
    The function uses the Kahan summation procedure 
    (cf. Dahlquist, Bjorck & Anderson). 
    """

    summ  = 0.0
    c     = 0.0
    fact  = 1.0
    k     =  0
    zero2 = zero1 = False
    while True:
        k     +=  1
        summo  =  summ
        ak     =  alpha * k
        akh    =  0.5 * ak
        r      =  lngamma(ak) - lnfactorial(k)
        term   = - ak * exp(r) * sin(PIHALF*ak) / pow(x, ak+1)
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
        if abs(summ-summo) < tolr*abs(summ) and abs(term) < tolr and zero2:
            break
        if abs(term) < tolr:
            if zero1: zero2 = True
            else:     zero1 = True
    summ +=  c
    #pdf   =  summ / PI
    pdf   =  PIINV * summ

    pdf   =  kept_within(0.0, pdf)
    return pdf

# end of _dstable_sym_big

# -------------------------------------------------------------------------------------

def _dstable_sym_int(alpha, x, tolromb, mxsplromb):

    """
    The integral formulation of the standard pdf (cf. for instance Matsui, M., 
    and Takemura, A., "Some Improvements in Numerical Evaluation of Symmetric 
    Stable Density and its Derivatives", University of Tokyo Report CIRJE-F-292,
    Aug. 2004.
    """

    # Auxiliary functions for calculating the breakpoint 
    # (the integration interval is broken up into two portions, 
    # cf. Matusi & Takemura) and integral:
    am1   = alpha - 1.0
    aoam1 = alpha/am1
    # -------------
    def _gm1(phi2):
        try: g = (x*cos(phi2)/sin(alpha*phi2))**(aoam1) * \
                                (cos(am1*phi2)/cos(phi2))
        except ZeroDivisionError: g = 0.0
        except OverflowError:     g = 0.0
        return g - 1.0
    # --------------
    def _func(phi1):
        try: g = (x*cos(phi1)/sin(alpha*phi1))**(aoam1) * \
                                (cos(am1*phi1)/cos(phi1))
        except ZeroDivisionError: g = 0.0
        except OverflowError:     g = 0.0
        y = exp(-g)
        if y == 0.0:
            z = g - log(g)
            y = exp(-z)
        else:
            y = g*y
        return y
    # --------------

    # Integrate!
    # First find the break point:
    point2 = zbrent(_gm1, MACHEPS, PIHALF, 'dstable_sym/_dstable_sym_int')
    if point2 == ERRCODE: point2 = 0.5*PIHALF
    # Then perform Romberg quadrature for the 
    # two panels separated by the break point:
    pdf  = qromberg(_func,   0.0,  point2, 'dstable_sym/_dstable_sym_int', \
                                            tolromb, mxsplromb)
    pdf += qromberg(_func, point2, PIHALF, 'dstable_sym/_dstable_sym_int', \
                                            tolromb, mxsplromb)
    pdf *= alpha/(PI*abs(am1)*x)

    # Return:
    return pdf

# end of _dstable_sym_int

# ------------------------------------------------------------------------------
# This is a department with ASYMMETRIC, stable distributions
# ------------------------------------------------------------------------------

def dlevy(scale, x):
    """
    The pdf of the Levy distribution (stable distribution with 
    alpha = 1/2 and beta = 1, aka the Cournot distribution). 
    This is actually the right-skewed Levy!
    
    f = sqrt(s/2pi) * (1/x)**(3/2) * exp(-s/2x)
    F = erfc(sqrt(s/2x))
    s >= 0.0, x >= 0
    """

    # Input check -----------------
    assert scale >= 0.0, "scale must not be negative in dlevy!"
    assert   x   >= 0.0, "variate must not be negative in dlevy!"
    # -----------------------------

    s05 =  0.5 * scale
    #n1  =  sqrt(s05/PI)
    n1  =  sqrt(PIINV*s05)
    d   =  x * sqrt(x)
    n2  =  exp(-s05/x)

    pdf =  n1 * n2 / d

    return pdf

# end of dlevy

# ------------------------------------------------------------------------------
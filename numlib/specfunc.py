# numlib/specfunc.py
# ==============================================================================
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
Module used to compute the value of "special functions". 
"""
# ------------------------------------------------------------------------------

from math import factorial, exp, log

from numlib.miscnum    import fsign
from misclib.numbers   import is_eveninteger, is_nonneginteger
from misclib.numbers   import safeint, kept_within, ERRCODE
from machdep.machnum   import MACHEPS
from misclib.errwarn   import warn
from misclib.mathconst import SQRTTWOPI, SQRTPIINV

_EIGHTMACHEPS = 8.0 * MACHEPS
_ERFC12       = 0.5**23  # 1.1920928955078125e-07
_ERFC21       = 1.5

# ------------------------------------------------------------------------------

def ifactorial(n, integer=True):
    """
    Computation of a factorial, returning an integer. For integer=True a long, 
    exact integer is returned. For integer=False an integer obtained from 
    floating-point arithmetics based on the function lngamma is returned - 
    it is approximate and may not be exact but faster for large n. ERRCODE 
    (cf. misclib.numbers) is returned if a floating-point OverflowError occurs 
    and a warning is sent to stdout (no error can occur when integer=True).
    """

    assert is_nonneginteger(n), \
                 "the argument to ifactorial must be a non-negative integer!"

    if integer:
        fact = factorial(n)

    else:
        try:
            fact = safeint(round(exp(lngamma(n+1.0))), 'ifactorial')
        except OverflowError:
            fact = ERRCODE
            warn("OverflowError in ifactorial - ERRCODE is returned")

    return fact

# end of ifactorial

# ------------------------------------------------------------------------------

def ffactorial(n, integer=True):
    """
    Computation of a factorial, returning a float. If integer=True a rounded 
    float obtained from integer arithmetics is returned. If integer=False a 
    rounded float obtained from floating-point arithmetics based on the 
    function lngamma is returned. 
    
    Rounding may cause the return of an approximate result rather than an 
    exact. 
    
    float('inf') is returned if an OverflowError occurs.
    """

    assert is_nonneginteger(n), \
              "the argument to ffactorial must be a non-negative integer!"

    if integer:
        fact = factorial(n)
        try:
            ffact = round(float(fact))
        except OverflowError:
            ffact = float('inf')

    else:
        try:
            ffact = round(exp(lngamma(n+1.0)))
        except OverflowError:
            ffact = float('inf')

    return ffact

# end of ffactorial

# ------------------------------------------------------------------------------

def lnfactorial(n, integer=True):
    """
    Computation of the natural logarithm of a factorial using integer 
    arithmetics (integer=True) or floating point arithmetics using lngamma 
    (integer=False). In both cases a floating-point number is returned, of 
    course...
    """

    assert is_nonneginteger(n), \
               "the argument to lnfactorial must be a non-negative integer!"

    if integer:
        fact   = factorial(n)
        lnfact = log(fact)

    else:
        lnfact = lngamma(n+1.0)

    return lnfact

# end of lnfactorial

# ------------------------------------------------------------------------------

def lngamma(alpha):
    """
    The natural logarithm of the gamma function for real, positive argument.
    Maximum fractional error can be estimated to < 1.e-13
    
    For alpha < 20.0 lngamma uses Laczos expansion with coefficients taken from 
    http://home.att.net/~numericana/answer/info/godfrey.htm where fractional 
    error of the gamma function using these specific coefficients is claimed 
    to be < 1.e-13
    
    For alpha >= 20.0 the Euler-McLaurin series expansion for ln(gamma) is used 
    (see for instance Dahlquist, Bjorck & Anderson). For EulerMcLaurin the 
    fractional  t r u n c a t i o n  error is less than 2.4e-14 (and always 
    positive).
    """

    assert alpha > 0.0, "Argument must be real and positive in lngamma!"


    if alpha < 20.0:
        # The Laczos expansion with coefficients taken from 
        # http://home.att.net/~numericana/answer/info/godfrey.htm 
        # where fractional error of the gamma function using these
        # particular coefficients is claimed to be < 1.e-13:

        coeff = ( \
                 1.000000000000000174663,  5716.400188274341379136,        \
            -14815.30426768413909044,     14291.49277657478554025,         \
             -6348.160217641458813289,     1301.608286058321874105,        \
              -108.1767053514369634679,       2.605696505611755827729,     \
                -0.7423452510201416151527e-2, 0.5384136432509564062961e-7, \
                -0.4023533141268236372067e-8 ) ;  lm1 = len(coeff) - 1

        g     = 9.0
        arg1  = alpha + 0.5
        arg2  = arg1 + g
        summ  = 0.0
        c     = 0.0
        a     = alpha + 11.0
        for k in range(lm1, 0, -1):  # The Kahan summation procedure is used
            a    -= 1.0
            term  = coeff[k]/a
            y     = term + c
            t     = summ + y
            if fsign(y) == fsign(summ):
                f = (0.46*t-t) + t
                c = ((summ-f)-(t-f)) + y
            else:
                c = (summ-t) + y
            summ  =  t
        summ += c
        summ += coeff[0]
        summ *= SQRTTWOPI

        lngam  =  arg1*log(arg2) - arg2 + log(summ/alpha)


    else:
        # The Euler-McLaurin series expansion for ln(gamma) 
        # (see for instance Dahlquist, Bjorck & Anderson).
        # For alpha >= 20.0 the fractional t_r_u_n_c_a_t_i_o_n 
        # error is less than 2.4e-14 (and always positive):

        alfa     = alpha - 1.0

        #const    =  0.9189385332046727417803296      #  0.5*log(2.0*pi)
        #coeff0   = -1.0
        #coeff1   =  0.0833333333333333333333333333   #  1.0/12.0
        #coeff2   = -0.0027777777777777777777777778   # -1.0/360.0
        #coeff3   =  0.0007936507936507936507936508   #  1.0/1260.0
        #coeff4   = -0.0005952380952380952380952381   # -1.0/1680.0
        #coeff5   =  0.0008417508417508417184175084   #  1.0/1188.0

        oneoa2 =  1.0 / alfa**2
        summ   = -1.0 + oneoa2*\
                  ( 0.0833333333333333333333333333 + oneoa2*\
                  (-0.0027777777777777777777777778 + oneoa2*\
                  ( 0.0007936507936507936507936508 + oneoa2*\
                   -0.0005952380952380952380952381)))
                                               # coeff5 is not used
        summ  *=  alfa

        lngam  =  0.9189385332046727417803296 + (alfa+0.5)*log(alfa) + summ
        #print coeff5 / (alfa**7 * lngam)   # Fractional truncation error - 
                                            # coeff5 used here

    return lngam

# end of lngamma

# ------------------------------------------------------------------------------

def beta(a, b):
    """
    The beta function (uses exp(lngamma(a) + lngamma(b) - lngamma(a+b))). 
    ---------
    NB If you need the logarithm of beta: use lnbeta instead!!!
    """

    assert a > 0.0, "both arguments to beta must be positive floats!"
    assert b > 0.0, "both arguments to beta must be positive floats!"

    return exp(lngamma(a) + lngamma(b) - lngamma(a+b))

# end of beta

# ------------------------------------------------------------------------------

def lnbeta(a, b):
    """
    The natural logarithm of the beta function 
    (uses lngamma(a) + lngamma(b) - lngamma(a+b)). 
    """

    assert a > 0.0, "both arguments to lnbeta must be positive floats!"
    assert b > 0.0, "both arguments to lnbeta must be positive floats!"

    return lngamma(a) + lngamma(b) - lngamma(a+b)

# end of lnbeta

# ------------------------------------------------------------------------------

def erfc1(x, tol=_EIGHTMACHEPS):
    """
    Computation of the complementary error function for real argument.
    Fractional error is estimated to < 50*machine epsilon for abs(x) <= 1.5
    and < 1.e-8 elsewhere (erfc2 is called for abs(x) > 1.5 for numeric reasons).
    
    The function uses a power series expansion for arguments between -1.5 
    and +1.5 (cf. Abramowitz & Stegun) and continued fractions for all other 
    arguments (cf. A. Cuyt et al., "Continued Fractions for Special Functions: 
    Handbook and Software", Universiteit Antwerpen, where a slightly faster 
    converging expression than that of Abramowitz & Stegun's CF is presented. 
    Cuyt's "ER.20" is used here).
    """

    if tol < _EIGHTMACHEPS:
        tol = _EIGHTMACHEPS
        txt1 = "No use using tolerance < 8.0*machine epsilon in erfc1."
        txt2 = " 8.0*machine epsilon is used"
        warn(txt)

    ax  = abs(x)
    xx  = x*x

    if ax <= _ERFC21:
        # Power series expansion (cf. Abramowitz & Stegun)
        k     = 0.0
        sign  = 1.0
        xpart = 1.0
        den1  = 1.0
        #den2  = 1.0
        #term  = sign*xpart/(den1*den2)
        #summ  = term
        summ   = 1.0
        c      = 0.0
        while True: # The Kahan summation proc. (cf. Dahlquist, Bjorck & Anderson)
            k     += 1.0
            summo  = summ
            sign   = -sign
            xpart *= xx
            den1  *= k
            den2   = 2.0*k + 1.0
            term   = sign*xpart/(den1*den2)
            y      = term + c
            t      = summ + y
            if fsign(y) == fsign(summ):
                f = (0.46*t-t) + t
                c = ((summ-f)-(t-f)) + y
            else:
                c = (summ-t) + y
            summ  = t
            if abs(summ-summo) < tol*abs(summ):
                summ += c
                break
        #r = 1.0 - (2.0*ax/SQRTPI)*summ
        r = 1.0 - (2.0*SQRTPIINV*ax)*summ

    else: return erfc2(x)
    """
        # Compute continued fractions:
        # Q = b0 + a1/(b1 + a2/(b2 + a3/(b3 + ......... where ak   
        # are numerator terms and where bk are denominator terms 
        # (and where a0 is always 0).
        # Here:
        # b0 = 0.0
        # a1 = 1.0
        # a2 = 0.5
        # a3 = 1.5
        # a4 = 2.0
        # b1 = b3 etc = x*x
        # b2 = b4 etx = 1.0
        # (cf. Cuyt et al.)

        #k   = 0.0
        bk  = 0.0
        Am1 = 1.0
        Bm1 = 0.0
        A0  = bk
        B0  = 1.0

        k   = 1.0
        bk  = xx
        ak  = 1.0
        Ap1 = bk*A0 + ak*Am1
        Bp1 = bk*B0 + ak*Bm1
        Q   = Ap1/Bp1
        Am1 = A0
        Bm1 = B0
        A0  = Ap1
        B0  = Bp1

        while True:
            k   += 1.0
            Qold = Q
            if is_eveninteger(k): bk = 1.0
            else:                 bk = xx
            ak   = 0.5 * (k-1.0)
            Ap1  = bk*A0 + ak*Am1
            Bp1  = bk*B0 + ak*Bm1
            Q    = Ap1/Bp1
            if abs(Q-Qold) < abs(Q)*tol:
                break
            Am1  = A0
            Bm1  = B0
            A0   = Ap1
            B0   = Bp1

        p  = exp(-xx)
        if p == 0.0: # Take a chance...
            #r = exp(-xx + log(ax*Q/SQRTPI))
            r = exp(-xx + log(SQRTPIINV*ax*Q))
        else:
            #r = ax * p * Q / SQRTPI
            r = SQRTPIINV * ax * p * Q"""

    if x < 0.0: r = 2.0 - r
    r = kept_within(0.0, r, 2.0)
    return r

# end of erfc1

# ------------------------------------------------------------------------------

def erfc2(x):
    """
    Computation of the complementary error function for real argument.
    Faster than erfc1, particularly for intermediate-valued x. 
    Max fractional error is estimated to < 1.e-8.
    
    erfc2 is based on a Chebyshev approximation of f(u) = exp(x*x)*erfc1(x) 
    where the expansion is made with u = (abs(x)-3.75)/(abs(x)+3.75), which 
    is the same trick as that used in the NAG library. erfc2 calls erfc1 for 
    very small abs(x) for reasons of computational speed.
    """

    ax = abs(x)
    #if ax < _ERFC12 or ax > _ERFC21: return erfc1(x)
    if ax < _ERFC12: return erfc1(x)
    
    coeffs = (  0.61014308192320044,    -0.43484127271257744,    \
                0.17635119364360541,    -0.060710795609249295,   \
                0.017712068995693976,   -0.0043211193855670627,  \
                0.00085421667688714289, -0.00012715509060885478, \
                1.1248167243560482e-05,  3.1306388580398804e-07, \
               -2.7098806859582325e-07,  3.073762304683925e-08,  \
                2.515620212140135e-09,  -1.0289300522714484e-09  )

    u    = (ax-3.75)/(ax+3.75)
    d    =  0.0
    dd   =  0.0
    twou =  2.0 * u
    lm1  =  len(coeffs) - 1
    for k in range(lm1, 0, -1): # Clenshaw's recurrence for Chebyshev polynomials
        sv =  d
        d  =  twou*d - dd + coeffs[k]
        dd =  sv
    chbv =  u*d - dd + 0.5*coeffs[0]

    xx = x*x
    p  = exp(-xx)
    if p == 0.0: # Take a chance...
        r = exp(-xx + log(chbv))
    else:
        r = p * chbv

    if x < 0.0: r = 2.0 - r
    r = kept_within(0.0, r, 2.0)
    return r

# end of erfc2

# ------------------------------------------------------------------------------

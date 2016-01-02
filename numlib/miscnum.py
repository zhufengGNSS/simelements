# numlib/miscnum.py
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
Module contains a set of simple functions for numerically related tasks. 
"""
# ------------------------------------------------------------------------------

from math import exp, log

from machdep.machnum import MAXFLOAT

ONETHIRD = 0.3333333333333333333333333

# ------------------------------------------------------------------------------

def fsign(x):
    """
    Returns the sign of the input (real) number as a float: 1.0 or -1.0 
    """

    if x >= 0.0: return  1.0
    else:        return -1.0

# end of fsign

# ------------------------------------------------------------------------------

def isign(x):
    """
    Returns the sign of the input integer as an integer: 1 or -1 
    
    NB. Input must be an integer!!!
    """

    assert isinstance(x, int), "Argument to isign must be an integer!"
    if x >= 0: return  1
    else:      return -1

# end of isign

# ------------------------------------------------------------------------------

def krond(n, m):
    """
    The Kronecker delta: 1.0 if n = m, 0.0 otherwise.
    
    NB The function returns a float!
    """

    assert isinstance(n, int), "Both arguments to krond must be integers!"
    assert isinstance(m, int), "Both arguments to krond must be integers!"

    if n == m: return 1.0
    else:      return 0.0

# end of krond

# ------------------------------------------------------------------------------

def polyeval(a, x):
    """
    Computes the value of a polynomial an*x^n + an-1*x^n-1 + ... +  a0  
    where a is a real-valued vector (a list/tuple of floats), and x is 
    a float (this is, of course, "Horner's rule"!). The input list/tuple
    'a' must be ordered a0, a1, a2 etc.
    """

    b    = reversed(a)
    summ = 0.0
    for coeff in b: summ = summ*x + coeff

    return summ

# end of polyeval

# ------------------------------------------------------------------------------

def polyderiv(a, x):
    """
    Computes the value of the derivative of a polynomial 
    an*x^n + an-1*x^n-1 + ... +  a0  where a is a real-valued 
    vector (a list/tuple of floats), and x is a float. "Horner's 
    rule" is used to crank out the final numerical result. 
    The input list/tuple 'a' must be ordered a0, a1, a2 etc.
    """

    b = []
    n = len(a)
    for k in range(1, n):
    	b.append(k * a[k])

    c    = reversed(b)
    summ = 0.0
    for coeff in c: summ = summ*x + coeff

    return summ

# end of polyderiv

# ------------------------------------------------------------------------------

def realcbrt(x, overflow=float('inf')):
    """
    The real cubic root of a real floating point number (the built-in 
    pow(x, 1/3) and x**(1/3) only works for non-negative x). 
    
    The function is safe in that it does not crash on OverflowError (it 
    uses 'safepow').  'overflow' must not be smaller than MAXFLOAT.
    """

    return fsign(x) * safepow(abs(x), ONETHIRD, overflow)

# end of realcbrt

# ------------------------------------------------------------------------------

def expxm1(x, overflow=float('inf')):
    """
    Computes exp(x) - 1.0 in a manner that somewhat reduces the problem of 
    catastrophic cancellation. Fractional error is estimated to < 1.e-8.
    
    The function is safe in that it does not crash on OverflowError (it 
    uses 'safeexp').  'overflow' must not be smaller than MAXFLOAT.
    """

    if abs(x) >= 0.5**15:
        return  safeexp(x, overflow) - 1.0
    else:
        y  =  x*(120.0 + x*(60.0 + x*(20.0 + x*(5.0 + x))))
        return  y/120.0

# end of expxm1

# ------------------------------------------------------------------------------

def expxm1ox(x, overflow=float('inf')):
    """
    Computes (exp(x)-1.0)/x in a manner that somewhat reduces the problem 
    of catastrophic cancellation. Absolute error is estimated to < 1.e-11. 
    
    The function is safe in that it does not crash on OverflowError (it 
    uses 'safeexp').  'overflow' must not be smaller than MAXFLOAT.
    """

    if abs(x) >= 0.5**15:
        return  (safeexp(x, overflow) - 1.0) / x
    else:
        y  =  x*(60.0 + x*(20.0 + x*(5.0 + x)))
        return  y/120.0 + 1.0

# end of expxm1ox

# ------------------------------------------------------------------------------

def ln1px(x, zeroreturn=float('-inf')):
    """
    Very accurate computation of the natural logarithm of 1.0 + x for 
    all x > -1.0, also for x < machine epsilon (algorithm taken from 
    "What Every Computer Scientist Should Know About Floating-Point 
    Arithmetic", Sun Microsystems Inc., 1994). The function is safe in 
    that it will not crash for ln(0.0) (it uses 'safelog')

    NB. Beginning with Python 2.6 there is a built-in version of ln(1+x) that  
    gives identical results and that should be used if the safe function is 
    not desired: log1p(x)
    """

    assert x >= -1.0, "Argument must be > -1.0 in ln1px!"

    y = 1.0 + x

    if abs(x) > 0.25:
        return safelog(y, zeroreturn)
    else:
        if y == 1.0:  return  x
        else:         return  x * safelog(y, zeroreturn) / (y-1.0)

# end of ln1px

# ------------------------------------------------------------------------------

def safediv(num, den, overflow=float('inf'), zerodiv=float('inf')):
    """
    Carries out "/" division in a safe manner that will not crash on 
    OverflowError or ZeroDivisionError - 'overflow' and 'zerodiv' with 
    the appropriate sign will be returned, respectively. 0.0/0.0 will 
    be returned as 1.0.
    
    NB 'overflow' and 'zerodiv' must be positive floats. 'overflow' and 
    'zerodiv' must not be smaller than MAXFLOAT and 'overflow' must not 
    be greater than 'zerodiv'. 
    """

    assert overflow >= MAXFLOAT and zerodiv >= MAXFLOAT, \
              "both overflow and zerodiv must be >= MAXFLOAT in safediv!"
    assert overflow <= zerodiv, "overflow must be <= zerodiv in safediv!"

    if float(num) == 0.0 and float(den) == 0.0:
        ratio = 1.0

    else:
        try:
            ratio = num/den
            if abs(ratio) == float('inf'):
                ratio = fsign(den)*fsign(num)*overflow
        except ZeroDivisionError:
            ratio = fsign(den)*fsign(num)*zerodiv

    return ratio

# end of safediv

# ------------------------------------------------------------------------------

def safepow(x, a, overflow=float('inf')):
    """
    A "safe" version of the built-in pow function that will not crash on 
    OverflowError. 'overflow' must not be smaller than MAXFLOAT.
    """

    assert overflow >= MAXFLOAT, \
                           "overflow limit must not be < MAXFLOAT in safepow!"

    try:
        return pow(x, a)
    except OverflowError:
        return overflow

# end of safepow

# ------------------------------------------------------------------------------

def safeexp(x, overflow=float('inf')):
    """
    A "safe" version of the built-in exp function that will not crash on 
    OverflowError. 'overflow' must not be smaller than MAXFLOAT. 
    """

    assert overflow >= MAXFLOAT,\
                          "overflow limit must not be < MAXFLOAT in safeexp!"

    try:
        return exp(x)
    except OverflowError:
        return overflow

# end of safeexp

# ------------------------------------------------------------------------------

def safelog(x, zeroerror=float('-inf')):
    """
    A "safe" version of the built-in log function that will not crash for 
    argument = 0.0. 'zeroerror' must not be greater than -MAXFLOAT.
    """

    assert zeroerror <= -MAXFLOAT, \
                         "zeroerror limit must not be > -MAXFLOAT in safelog!"

    if x == 0.0: return zeroerror
    else:        return log(x)

# end of safelog

# ------------------------------------------------------------------------------

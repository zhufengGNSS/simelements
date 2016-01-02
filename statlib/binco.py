# statlib/binco.py
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
Computation of binomial coefficients and frequencies. Floating-point 
arithmetics (integer=False) is faster for n = 1000 while integer 
arithmetics (integer=True) means better accuracy, also for fbincoeff 
and lnbincoeff !
"""
# ------------------------------------------------------------------------------

from math import exp, log

from numlib.specfunc import lngamma
from misclib.numbers import is_posinteger, is_nonneginteger
from misclib.numbers import safeint, kept_within, ERRCODE

# ------------------------------------------------------------------------------

def ibincoeff(n, k, integer=True):
    """
    Computation of a single binomial coefficient n over k, returning an 
    integer (possibly long). For integer=True integer arithmetics is used 
    throughout and there is no risk of overflow. For integer=False a floating- 
    point gamma function approximation is used and the result converted to 
    integer at the end (if an overflow occurs ERRCODE is returned). 
    """

    assert is_posinteger(n), \
               "n in n_over_k in ibincoeff must be a positive integer!"
    assert is_nonneginteger(k), \
           "k in n_over_k in ibincoeff must be a non-negative integer!"
    assert n >= k, "n must be >= k in n_over_k in ibincoeff!"

    if integer:
        ibico = _bicolongint(n, k)

    else:
        try:
            lnbico  =  lngamma(n+1) - lngamma(k+1) - lngamma(n-k+1)
            ibico   =  safeint(round(exp(lnbico)), 'ibincoeff')
        except OverflowError:
            ibico   =  ERRCODE

    return ibico

# end of ibincoeff

# ------------------------------------------------------------------------------

def fbincoeff(n, k, integer=True):
    """
    Computation of a single binomial coefficient n over k, returning a float 
    (an OverflowError returns float('inf'), which would occur for n > 1029 
    for IEEE754 floating-point standard). 
    """

    assert is_posinteger(n), \
               "n in n_over_k in fbincoeff must be a positive integer!"
    assert is_nonneginteger(k), \
           "k in n_over_k in fbincoeff must be a non-negative integer!"
    assert n >= k, "n must be >= k in n_over_k in fbincoeff!"

    if integer:
        bico = _bicolongint(n, k)
        try:
            fbico = round(float(bico))
        except OverflowError:
            fbico = float('inf')

    else:
        try:
            lnbico  =  lngamma(n+1) - lngamma(k+1) - lngamma(n-k+1)
            fbico   =  round(exp(lnbico))
        except OverflowError:
            fbico   =  float('inf')

    return fbico

# end of fbincoeff

# ------------------------------------------------------------------------------

def lnbincoeff(n, k, integer=True):
    """
    Computation of the natural logarithm of a single binomial coefficient 
    n over k
    """

    assert is_posinteger(n), \
               "n in n_over_k in lnbincoeff must be a positive integer!"
    assert is_nonneginteger(k), \
           "k in n_over_k in lnbincoeff must be a non-negative integer!"
    assert n >= k, "n must be >= k in n_over_k in lnbincoeff!"

    if integer:
        bico    =  _bicolongint(n, k)
        lnbico  =  log(bico)

    else:
        lnbico  =  lngamma(n+1) - lngamma(k+1) - lngamma(n-k+1)

    return lnbico

# end of lnbincoeff

# ------------------------------------------------------------------------------

def binfreq(n, integer=True):
    """
    Computes all binomial coefficients for n AND the relative frequencies AND 
    the cumulative frequencies and returns them in two lists. Uses fbincoeff  
    (but floats are returned). 
    """

    assert is_posinteger(n), "argument to binFreq must be a positive integer!"

    z      = 0.0     # A float
    farray = []
    np1    = n + 1

    for k in range(0, np1):
        x  = fbincoeff(n, k, integer)
        z  = z + x        # A float
        farray.append(x)

    parray = []
    carray = []

    y = 0.0
    for k in range(0, np1):
        x  = farray[k] / z
        y  = y + x
        parray.append(x)
        carray.append(y)

    return parray, carray

# end of binfreq

# ------------------------------------------------------------------------------

def binprob(n, phi, integer=True):
    """
    Computes all binomial terms for n given the Bernoulli probability phi. 
    Returns the relative frequencies AND the cumulative frequencies (two lists). 
    """

    assert is_posinteger(n),  \
                      "first argument to binProb must be a positive integer!"
    assert 0.0 <= phi <= 1.0, \
                    "Bernoulli probability must be in [0.0, 1.0] in binProb!"

    farray = []
    np1    = n + 1
    for k in range(0, np1):
        x = fbincoeff(n, k, integer)
        farray.append(x)

    parray = []
    carray = []

    y = 0.0
    q = 1.0 - phi
    for k in range(0, np1):
        x  = farray[k] * phi**k * q**(n-k)
        x  = kept_within(0.0, x, 1.0)
        y  = y + x
        y  = kept_within(0.0, y, 1.0)
        parray.append(x)
        carray.append(y)

    return parray, carray

# end of binprob

# ------------------------------------------------------------------------------

def _bicolongint(n, k):
    """
    A piece of code common to several functions. Used to compute an exact 
    integer value for a binomial coefficient. 
    """

    if k == 0 or k == n:
        bico = 1

    else:
        num0  = max(k, n-k) + 1
        denN  = min(k, n-k)

        np1   = n + 1
        #numer = long(num0)
        numer = num0
        n0p1  = num0 + 1
        for j in range(n0p1, np1):  numer = numer*j

        denNp1 = denN + 1
        denom  = 1
        for j in range(1, denNp1): denom = denom*j

        bico = numer//denom  # Integer division will not give a remainder here

    return bico

# end of _bicolongint

# ------------------------------------------------------------------------------
# numlib/quadrature.py
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

from misclib.numbers import is_nonneginteger
from machdep.machnum import TWOMACHEPS
from misclib.errwarn import warn

# ------------------------------------------------------------------------------

def qromberg(func, a, b, caller='caller', tolf=TWOMACHEPS, maxnsplits=16):
    """
    Romberg integration of a function of one variable over the interval [a, b]. 
    'caller' is the name of the program, method or function calling qromberg.
    'tolf' is the maximum allowed fractional difference between the two last 
    "tail" integrals on the "T table diagonal". 'maxnsplits' is the maximum 
    number of consecutive splits of the subintervals into halves.
    (cf. Davis-Rabinowitz and Dahlquist-Bjorck-Anderson).
    """

    span  =  b - a
    assert span >= 0.0, \
               "Integration limits must be in increasing order in qromberg!"
    assert is_nonneginteger(maxnsplits), \
           "Max number of splits must be a nonnegative integer in qromberg!"

    m  = 1; n  =  1
    summ       =  0.5 * (func(a)+func(b))
    intgrl     =  span * summ
    ttable     =  [[intgrl]]    # First entry into T table (a nested list)
    previntgrl =  (1.0 + 2.0*tolf)*intgrl  # Ascertains intgrl != previntgrl...
    adiff      = abs(intgrl-previntgrl)

    while (adiff > tolf*abs(intgrl)) and (m <= maxnsplits):
        n *= 2
        # We have made a computation for all lower powers-of-2 starting with 1. 
        # The below procedure sums up the new ordinates and then multiplies the 
        # sum with the width of the trapezoids.
        nhalved = n // 2   # Integer division
        h       = span / float(nhalved)
        x       = a + 0.5*h
        subsum  = 0.0
        for k in range(0, nhalved):
            subsum += func(x + k*h)
        summ += subsum
        ttable.append([])
        ttable[m].append(span * summ / n)

        # Interpolation Richardson style:
        for k in range(0, m):    
            aux = float(4**(k+1))
            y   = (aux*ttable[m][k] - ttable[m-1][k]) / (aux - 1.0)
            ttable[m].append(y)
        intgrl     = ttable[m][m]
        previntgrl = ttable[m-1][m-1]
        adiff      = abs(intgrl-previntgrl)
        m += 1

    if m > maxnsplits:
        wtxt1 = "qromberg called by " + caller + " failed to converge.\n"
        wtxt2 = "abs(intgrl-previntgrl) = " + str(adiff)
        wtxt3 = " for " + str(maxnsplits) + " splits"
        warn(wtxt1+wtxt2+wtxt3)

    return intgrl

# end of qromberg

# ------------------------------------------------------------------------------

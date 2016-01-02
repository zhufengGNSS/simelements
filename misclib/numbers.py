# misclib/numbers.py
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
Module contains functions for assigning, checking and manipulating numbers. 
"""
# ------------------------------------------------------------------------------

from math import floor

# ------------------------------------------------------------------------------

ERRCODE  = -9999999999      # Ten nines

# ------------------------------------------------------------------------------

def is_integer(x):
    """
    Logical function. Returns 'True' if argument is an integer number, 
    'False' otherwise.  
    """

    return isinstance(x, int)

# end of is_integer

# ------------------------------------------------------------------------------

def is_posinteger(x):
    """
    Logical function. Returns 'True' if argument is a positive integer, 
    'False' otherwise.  
    """

    return isinstance(x, int) and x > 0

# end of is_posinteger

# ------------------------------------------------------------------------------

def is_nonneginteger(x):
    """
    Logical function. Returns 'True' if argument is a non-negative integer, 
    'False' otherwise. 
    """

    return isinstance(x, int) and x >= 0

# end of is_nonneginteger

# ------------------------------------------------------------------------------

def is_oddinteger(x):
    """
    Logical function. Returns 'True' if argument is an odd, positive integer, 
    'False' otherwise. 
    """

    # % is Python's modulo function

    return isinstance(x, int) and x > 0 and x % 2 != 0

# end of is_oddinteger

# ------------------------------------------------------------------------------

def is_eveninteger(x):
    """
    Logical function. Returns 'True' if argument is an even, positive integer, 
    'False' otherwise. 
    """

    # % is Python's modulo function

    return isinstance(x, int) and x > 1 and x % 2 == 0

# end of is_eveninteger

# ------------------------------------------------------------------------------

def is_fullquotient(num, den):
    """
    Logical function. Returns 'True' if integer division num/den can be 
    made without a remainder, 'False' otherwise. 
    """

    # % is Python's modulo function

    if not is_posinteger(den):
        errtxt = "Denominator must be a positive integer in is_fullquotient!"
        raise ValueError(errtxt)
    elif isinstance(num, int):
        return num % den == 0
    else:
        raise ValueError("Both arguments must be integers in is_fullquotient!")

# end of is_fullquotient

# ------------------------------------------------------------------------------

def kept_within(minimum, x, maximum=float('inf')):
    """
    Makes certain that the input stays in the interval [minimum, maximum]. 
    """

    if   x < minimum: return minimum
    elif x > maximum: return maximum
    else:             return x

# end of kept_within

# ------------------------------------------------------------------------------

def split_int_frac(x, iflag=None):
    """
    Splits input x (real/float) into integral and fractional part.
    Outputs are integral part (integer) and fractional part (float).
    Default: positive input number yields integral part >= 0 and fraction >= 0.0
             negative input number yields integral part <= 0 and fraction <= 0.0
    flag = -1 forces fractional part <= 0.0 and integral part accordingly
    flag =  1 forces fractional part >= 0.0 and integral part accordingly
    flag =  0 forces abs(fractional part) <= 0.5 and integral part accordingly
    """

    IFLAGS = (None, -1, 0, 1)
    assert iflag in IFLAGS, "iflag must be either -1, 1 or 0 in split_int_frac!"

    if iflag == 0:
        i = int(round(x))
        f = x - i

    else:
        i = int(x)
        f = x - i
        if   iflag == -1 and f > 0.0:  i = i + 1
        elif iflag ==  1 and f < 0.0:  i = i - 1

    f = x - i

    return i, f

# end of split_int_frac

# -------------------------------------------------------------------------------

def safeint(x, caller='caller'):
    """
    Returns a safely rounded integer version of an integer-valued input float
    (caller is the name of the calling function/method). 
    Original idea from "Learning Python". 
    """

    truncated = floor(float(x))
    rounded   = round(float(x))

    if truncated == rounded:
        return int(truncated)

    else: 
        errortext = "Conversion of " + str(x) + " to integer not safe in " \
                                     + caller + "!"
        raise TypeError(errortext)

# end of safeint

# ------------------------------------------------------------------------------
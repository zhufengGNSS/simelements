# xbuiltins/xabs.py
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
#-------------------------------------------------------------------------------
"""
Module contains a more powerful variant of Python's built-in 'abs', 
specially designed to handle matrices and vectors from the Matrix class.
"""
#-------------------------------------------------------------------------------

from math import sqrt

from misclib.errwarn import Error

#-------------------------------------------------------------------------------

def xabs(x):
    """
    Replaces the built-in abs. Handles real and complex numbers and scalars 
    as well as lists/tuples, matrix row vectors and matrix column vectors 
    having the nested type of structure defined in the Matrix class.
    
    Arguments:
    ----------
    x       The argument may be a float or a complex number or a list/tuple, 
            or a nested list/tuple vector structure containing floats or
            complex numbers

    Outputs:
    --------
    The absolute value of the input (float) 
    """

    try:
        nrows = len(x)    # OK it's a list

        try:
            ncols = len(x[0])    # OK it's even a matrix vector!

            if   nrows == 1:     # It's a matrix row vector
                a2 = 0.0
                for k in range(0, ncols):
                    z  =  complex(x[0][k])
                    a2 =  a2 + z.real*z.real + z.imag*z.imag
            elif ncols == 1:     # It's a matrix column vector
                a2 = 0.0
                for k in range(0, nrows):
                    z  =  complex(x[k][0])
                    a2 =  a2 + z.real*z.real + z.imag*z.imag
            else:
                raise Error("abs: argument not a vector nor a scalar")

        except TypeError: # It was a list but not a matrix vector
            a2 = 0.0
            for k in range(0, nrows):
                z  =  complex(x[k])
                a2 =  a2 + z.real*z.real + z.imag*z.imag

    except TypeError:   # Well, it was merely a scalar!
        z  =  complex(x)
        a2 =  z.real*z.real + z.imag*z.imag

    return sqrt(a2)

# end of xabs

#-------------------------------------------------------------------------------
# xbuiltins/xmap.py
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
Module contains a more powerful variant of Python's built-in 'map', 
specially designed to handle matrices and vectors from the Matrix class. 
"""
# ------------------------------------------------------------------------------

from array import array

from misclib.matrix    import Matrix
from misclib.iterables import is_darray
from misclib.errwarn   import Error

# ------------------------------------------------------------------------------

def xmap(func, x):
    """
    Replaces the built-in map and can be used for matrices belonging to 
    the misclib.Matrix class, or lists with the same structure, as well as 
    ordinary lists and tuples.

    Arguments:
    ----------
    func    function defined externally that takes 'x' as its 
            sole non-default argument,

    x	    an ordinary list, tuple, array or a Matrix

    Outputs:
    ---------
    The modified list 
    """

    if isinstance(x, Matrix):
        nrows = len(x)
        ncols = len(x[0])    # It's even a matrix!
        z     = []
        for k in range(0, nrows):
            y = []
            for j in range(0, ncols):
                y.append(func(x[k][j]))
            y = array('d', y)
            z.append(y)

    elif is_darray(x):
        nrows = len(x)
        #z     = []
        z     = array('d', z)
        for k in range(0, nrows):
            z.append(func(x[k]))
        #z = array('d', z)

    elif isinstance(x, (tuple, Stack, list)):
        nrows = len(x)
        z     = []
        for k in range(0, nrows):
            z.append(func(x[k]))

    else:
        errtxt1 = "input state vector must be a tuple, a stack, "
        errtxt2 = "a list, a 'd' array or a Matrix!"
        raise Error(errtxt1+errtxt2)

    return z

# end of xmap

# ------------------------------------------------------------------------------

# numlib/singfunc.py
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
A collection of functions having discontinuities or points at which there
is continuity but where the first derivative is not defined. The functions 
are all designed so as to go up continuously to the point(s) - inclusive - and 
then continue with the new circumstances immediately after the singular point, 
such as for instance in the step function: step(0.0, x) = 0.0 for x <= 0.0 and 
step(0.0, x) = 1.0 for x > 0.0. This is to make the functions of this module 
work together with the "tiptoe" method, intended for use in solving systems of 
ODEs and included in the Dynamics class and its heirs. They may otherwise be 
used to handle different kinds of scheduled and unscheduled time-events in 
simulations.
"""
# ------------------------------------------------------------------------------

def step(x1, x):
    """
    Defines a step function of height 1.0 from x > x1 (exclusive) to infinity. 
    Particularly designed to be used with the tiptoe method of the Dynamics 
    class and its heirs. 
    """

    if x <= x1: return 0.0
    else:       return 1.0
    
# end of step

# ------------------------------------------------------------------------------

def sqbump(x1, x2, x):
    """
    Defines a square bump function (the difference between two step functions) 
    of height 1.0 on (x1, x2]. 
    
    Particularly designed to be used with the tiptoe method of the Dynamics 
    class and its heirs. 
    """

    assert x2 >= x1, "Break points must be ordered so that x1 <= x2 in sqbump!"

    if   x <= x1:  return 0.0
    elif x <= x2:  return 1.0
    else:          return 0.0

# end of sqbump

# ------------------------------------------------------------------------------

def ramp(x1, x2, x):
    """
    Defines a linear, upward slope between x1 and x2 up to height 1.0. 
    
    Particularly designed to be used with the tiptoe method of the Dynamics 
    class and its heirs. 
    """

    assert x2 >= x1, "Break points must be ordered so that x1 <= x2 in ramp!"

    if    x2 == x1:  return step(x1, x)
    elif  x  <= x1:  return 0.0
    elif  x  <= x2:  return (x-x1) / (x2-x1)
    else:            return 1.0

# end of ramp

# ------------------------------------------------------------------------------

def trihump(x1, x2, x3, x):
    """
    Defines a triangular hump with end points at x1 and x3 with the peak 
    at x2, height 1.0.
    
    Particularly designed to be used with the tiptoe method of the Dynamics 
    class and its heirs. 
    """

    assert x1 <  x3, "Break points must be ordered so that x1 <  x3 in trihump!"
    assert x2 <= x3, "Break points must be ordered so that x2 <= x3 in trihump!"
    assert x1 <= x2, "Break points must be ordered so that x1 <= x2 in trihump!"

    if    x  <= x1:  return 0.0
    elif  x  >  x3:  return 1.0
    elif  x1 == x2:  return 1.0 - (x-x2) / (x3-x2)
    elif  x2 == x3:  return (x-x1) / (x2-x1)
    elif  x  <= x2:  return (x-x1) / (x2-x1)
    elif  x  <= x3:  return 1.0 - (x-x2) / (x3-x2)
    else:            return 0.0

# end of trihump

# ------------------------------------------------------------------------------
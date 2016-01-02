# numlib/talbot.py
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
Module for the numerical inversion of Laplace transforms. 

"""
# ------------------------------------------------------------------------------

from math import exp, cos, sin, tan

from misclib.numbers   import is_posinteger
from misclib.mathconst import PI

# ------------------------------------------------------------------------------

def talbot(ftilde, tim, sigma, nu=1.0, tau=3.0, ntrap=32):
    """
    Function: computes the value of the inverse Laplace transform of a
              given complex function for a given time using Talbot's 
              algorithm. It calculates the sum in formula (31) on p. 104 
              in A. Talbot: J Inst Maths Applics 23(1979), 97-120. 
              Besides making sure that the special Talbotian integration 
              contour is to the right of all poles (the algorithm works 
              best when all poles are located on the real axis...), the 
              user also has to make sure that the contour does not coincide 
              with other singularities.

    Inputs:
    Name      Description
    ------    -----------
    ftilde    name of function to be inverted (must be a separate complex 
              function having one complex argument)

    tim       time for which the inversion is to be carried out 
              (float; must be > 0.0)

    sigma     shift parameter of the talbot integration contour; determined 
              by the position of the rightmost pole of ftilde: its value 
              (float) must be located to the right of all singularities but
              should be as close to the rightmost singularity as possible 
              (imprves accuracy and helps reducing the number of steps needed 
              for the integration)

    nu        shape parameter (float; > 0.0)

    tau       scale parameter (float; > 0.0)

    ntrap     number of points in trapezoidal integration (integer; > 0)
              (actually: one more will be used for the first term/point).
              Try using fewer than the default - a good choice of sigma
              will help!

    Outputs:
    Name      Description
    ------    -----------
    fnval     value of inverse (float)
    """

    # --------------------------------

    assert tim > 0.0, "time must be a positive float in talbot!"
    assert nu  > 0.0, "the shape parameter must be a positive float in talbot!"
    assert tau > 0.0, "the scale parameter must be a positive float in talbot!"
    assert is_posinteger(ntrap), \
                    "the number of steps must be a positive integer in talbot!"


    # Compute scaled inverted time
    lam    = tau / tim

    # Initiate lists for subsequent summing (positive and negative 
    # terms are summed separately to prevent cancellation)
    termn  = []
    termp  = []

    # Compute the first term of the series
    thetak =  0.0
    term   =  0.5 * _refthetak(ftilde, sigma, nu, tau, lam, thetak)
    if   term < 0.0: termn.append(term)
    elif term > 0.0: termp.append(term)
    #else: no need to add zeros

    # Compute the rest of the terms
    aux = PI / ntrap
    for k in range(1, ntrap):
        thetak =  k * aux
        term   =  _refthetak(ftilde, sigma, nu, tau, lam, thetak)
        if   term < 0.0:  termn.append(term)
        elif term > 0.0:  termp.append(term)
        #else: no need to add zeros

    # Sort the positive terms in ascending order and sum them up
    termp.sort()
    sump  =  sum(trm for trm in termp)

    # Sort the negative terms in descending order and sum them up
    termn.sort()
    termn.reverse()
    sumn  =  sum(trm for trm in termn)

    # Combine sums and multiply by constant factor to obtain final sum/value
    summ   =  sump + sumn
    ratio  =  lam / ntrap
    fnval  =  summ * exp(sigma*tim) * ratio

    return fnval

# end of talbot

# ------------------------------------------------------------------------------

def _refthetak(ftilde, sigma, nu, tau, lam, thetak):
    """
    Function: calculates function Re(f(theta(k))) defined in 
              formula (28) on page 104 in paper by Talbot.

    Inputs:
    Name      Description
    ------    -----------
    ftilde    function whose inverse Laplace transform is sought (complex)
    sigma     shift parameter (float)
    nu        shape parameter (float)
    tau       scale parameter (float)
    lam       lambda = scaled inverse time (float)
    thetak    actual argument (float)

    Outputs:
    Name      Description
    ------    -----------
    rfthtk    functional value (float)
    """

    # ----------------------------------

    # Compute alpha and beta ...
    if thetak == 0.0:
        alpha =  1.0
        beta  =  thetak
    else:
        alpha =  thetak / tan(thetak)
        beta  =  thetak + alpha*(alpha-1.0)/thetak

    # ... compute other auxiliary variables ...
    sing1  =  alpha
    sing2  =  1.0j * nu * thetak   # The following (through h) 
    snu    =  sing1 + sing2        # will also become complex:
    z      =  lam*snu + sigma      # .
    f      =  ftilde(z)            # .
    g      =  f                    # .
    h      =  -1.0j * f            # .

    # ... and some more auxiliary varables
    aux0   =  tau * nu * thetak
    aux1   =  cos(aux0)
    aux2   =  sin(aux0)
    aux3   =  nu*g - beta*h  # complex
    aux4   =  beta*g + nu*h  # complex

    # Compute final result
    aux    =  aux1*aux3 - aux2*aux4  # complex
    rfthtk =  aux * exp(alpha*tau)   # complex
    rfthtk =  rfthtk.real            # real part

    return rfthtk

# end of _refthetak

# ------------------------------------------------------------------------------
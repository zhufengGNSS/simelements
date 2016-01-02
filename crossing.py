# crossing.py
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

from math import sqrt

from numlib.miscnum  import fsign
from numlib.solveq   import zbrent, zbisect
from machdep.machnum import FOURMACHEPS, SQRTTINY

# ------------------------------------------------------------------------------

class Crossing():
    """
    Class inspired by the GASP IV combined simulation package and intended 
    for handling external and internal state-events as well as scheduled and 
    unscheduled time-events. It is used the following way (an example):

        crossing1 = Crossing(diffunc, mode, t0)

    where diffunc is a callable function having the time as its argument and 
    representing a crossing condition indicated by its changing its sign. This 
    may be used to control the flow of computation in combined discrete-dynamic
    /continuous simulations.

    'mode' is the crossing mode related to the search for a crossing point and 
    it may be either 'pos' for diffunc going from negative to positive, 'neg' 
    for positive to negative, or 'any' that covers both. t0 is the initial
    /starting time point.

    Events that change the course of a simulation may be handled using the 
    'crossed' method to look for a crossing point between the present time 
    and the next proposed time point in the simulation:

        u = crossing1.crossed(t, tnext)
        if not u:
            # No action
        else:
            tcrosslo, tcross, tcrosshi = u
            # Some action depending...
            resetXing(<newmode>)  # Optional (and the "< >" indicates 
                                  # that newmode is optional)

    If no crossing takes place in [t, tnext] then False is returned.

    If a crossing takes place in [t, tnext] then a three-element tuple is 
    returned: tcrosslo, tcross, tcrosshi - i. e. the "best estimate" as to 
    the crossing time (the estimate of the root of diffunc = 0.0), surrounded 
    by tcross-eps and tcross+eps where eps is the estimated error around the 
    root, given the tolerances input to the instance object at initiation.

    The estimates returned may for instance be used to "tiptoe" very close to 
    a crossing and then start over with the overlying computations with new 
    circumstances from a time point from where a crossing is certain to have 
    taken place. This may include preventing ODE solvers from trying to step 
    over discontinuities, triggering discrete events, initiating new objects 
    etc. etc.

    NB. If mode == 'pos' and diffunc > 0.0 at initiation or if mode == 'neg' 
    and diffunc < 0.0 at initiation, then crossing is assumed to have taken 
    place already and no crossings will be registered. A warning will however 
    be issued at initiation.
    """
# ------------------------------------------------------------------------------

    def __init__(self, diffunc, mode, t0, solver=zbisect, tolf=FOURMACHEPS, 
                                          tola=SQRTTINY,  maxniter=128):
        """
        Used to initiate the instance object.

        The optional input arguments other than diffunc, the crossing mode 
        related to the search for a crossing point with the 'crossed' method, 
        and the initial time point, are:

        solver    Solver for finding the zero - based on an initial guess as to 
                  a span. zbrent and zbisect are the ones presently available,

        tolf      Desired fractional accuracy of time point at crossing (a 
                  combination of fractional and absolute is actually used: 
                  tolf*abs(tcross) + tola),

        tola      Desired absolute accuracy of time point at crossing (a 
                  combination of fractional and absolute is actually used: 
                  tolf*abs(tcross) + tola),

        maxniter  Maximum number of iterations in the search for crossing points
        
        (how to control the iterations using tolf, tola and maxniter to reach 
        a zero is described in more detail in the solveq module).
        """

        self.__diffunc = diffunc

        self.__mode    = mode[:4]
        assert self.__mode == 'any' or self.__mode == 'pos' or \
               self.__mode == 'neg', \
               "mode must be 'any', 'pos' or 'neg' in Crossing!"

        self.__diffprev = diffunc(t0)
        wtxt2 = str(self) + ". No crossings will be registered"
        if   self.__mode == 'pos' and self.__diffprev > 0.0:
            self.__has_crossed = True
            wtxt1 = "diffunc is positive at the outset in "
            warn(wtxt1+wtxt2)
        elif self.__mode == 'neg' and self.__diffprev < 0.0:
            self.__has_crossed = True
            wtxt1 = "diffunc is negative at the outset in "
            warn(wtxt1+wtxt2)
        else:
            self.__has_crossed = False

        self.__solver   = solver
        self.__tolf     = tolf
        self.__tola     = tola
        self.__maxniter = maxniter

    # end of __init__

# ------------------------------------------------------------------------------

    def crossed(self, t, tnext):
        """
        The method used for actually finding a (possible) crossing point between
        the present time t and proposed/scheduled next time point tnext. It does
        not require that the present time point be the tnext of the previous 
        step - it is possible to let t > previous tnext (by a small amount, 
        naturally) to allow for "tiptoeing" around discontinuities. It is also 
        possible to let t < previous tnext (by a small amount) in order to come 
        as close as possible to a discontinuity before passing it. 
        
        NB1. mode = 'any' allows for multiple crossings, whereas 'pos' and 
        'neg' allows for one crossing only - possible future crossings are not 
        registered for 'pos' and 'neg'. 
                
        NB2. A warning will be issued if the solver fails to converge for the 
        tolerances and maximum number of iterations input at the initiation of 
        the instance object, but a result is always returned.

        NB3. tcrosslo will always be output >=t and tcrosshi will always be 
        output <= tnext.
        """

        assert tnext >= t, \
               "There is something strange with the times input to crossed!"

        # No more crossings
        if self.__has_crossed and self.__mode != 'any': return False

        # Check to see whether the present time has already caused a crossing
        # (in which case it should have been taken care of):
        diff = self.__diffunc(t)
        if (self.__mode == 'any' and fsign(diff) != fsign(self.__diffprev)) \
        or (self.__mode == 'pos' and diff >= 0.0) \
        or (self.__mode == 'neg' and diff <= 0.0):
            self.__has_crossed = True
            return False

        diff = self.__diffunc(tnext)
        if (self.__mode == 'any' and fsign(diff) != fsign(self.__diffprev)) \
        or (self.__mode == 'pos' and diff >= 0.0) \
        or (self.__mode == 'neg' and diff <= 0.0):
            self.__diffprev = diff
            tcross = self.__solver(self.__diffunc, t, tnext, 'crossed', \
                                 self.__tolf, self.__tola, self.__maxniter)
            eps    = self.__tolf*abs(tcross) + self.__tola
            tcrosslo = max(t, tcross-eps)
            tcrosshi = min(tcross+eps, tnext)
            return tcrosslo, tcross, tcrosshi

        else:
            self.__diffprev = diff
            return False

    # end of crossed

# ------------------------------------------------------------------------------

    def reset_xing(self, newmode=False):
        """
        May be used to reset an instance object so that it will register 
        crossings again. The mode can also be changed (using 'pos', 'neg' 
        or 'any' for newmode).
        """

        self.__has_crossed = False
        if newmode:
            self.__mode = newmode[:4]
            assert self.__mode == 'any' or self.__mode == 'pos' or \
                   self.__mode == 'neg', \
                      "newmode must be 'any', 'pos' or 'neg' in reset_xing!"

    # end of reset_xing

# ------------------------------------------------------------------------------

# end of Crossing

# ------------------------------------------------------------------------------
# cumrandstrm.py
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
# -----------------------------------------------------------------------------

from random import Random

from numlib.miscnum  import safelog, safediv
from misclib.numbers import is_posinteger, kept_within

# -----------------------------------------------------------------------------

class CumulRandomStream():
    """
    CLASS FOR INITIATING STREAMS OF RANDOM NUMBERS REPRESENTING CUMULATIVE TIME 
    FOR EVENTS ASSOCIATED WITH HOMOGENEOUS AND INHOMOGENEOUS POISSON PROCESSES. 
    CUMULATIVE TIME IS RECORDED INTERNALLY AND MUST NOT BE KEPT IN RECORD BY 
    THE CALLER.

    NB. All methods return a single random number on each call. 

    CumulRandomStream uses the built-in basic rng from Python's built-in 
    Random class, the so called "Mersenne Twister".

    The class is normally used something like this:
        rstream1 = CumulRandomStream()
        lam1     =     2.5
        tstop    =    24.0
        nrealiz  = 1000
        for k in range(0, nrealiz)
            while True:
                tevent = rstream1.rexpo_cum(lam1)  
                ....
                if tevent > tstop: break
            rstream1.reset()

    If another seed than the default is desired just type 
        rstream1 = CumulRandomStream(|some positive integer|)

    In order to continue with a restart from and time zero in the 
    same simulation the instance must have its methods reset:
        rstream1.reset()

    A SEPARATE STREAM MUST BE INSTANTIATED FOR EACH ONE OF THE VARIATES 
    THAT ARE GENERATED USING THIS CLASS, AND IT MAY BE WISE TO START 
    EACH ONE WITH A SEPARATE SEED!

    NB. Methods may return float('inf') or float('-inf') !!!!!
    """
# -----------------------------------------------------------------------------

    def __init__(self, nseed=2147483647):
        """
        Initiates the instance object and sets cumulative quantities to zero.
        """

        errtxt  = "The seed must be a positive integer in CumulRandomStream\n"
        errtxt += "\t(external feeds cannot be used)"
        assert is_posinteger(nseed), errtxt

        rstream      = Random(nseed)
        self.runif01 = rstream.random

        self.__tcum   = 0.0   # For all except rpieceexpo_cum and rinhomexpo_cum
        self.__cumul  = 0.0   # For rpieceexpo_cum
        self.__ticum  = 0.0   # For rinhomexpo_cum

    # end of __init__

# -----------------------------------------------------------------------------

    def rconst_cum(self, lamc):
        """
        Generates - cumulatively - constantly spaced interarrival times with 
        arrival rate (arrival frequency) lamc from clock time = 0.0. 

        NB  A dummy random number is picked each time the method is called 
        for reasons of synchronization.

        NB. A SEPARATE STREAM MUST BE INSTANTIATED FOR EACH ONE OF THE VARIATES 
        THAT ARE GENERATED USING THIS METHOD, EACH WITH A SEPARATE SEED!! 
        """

        assert lamc >= 0.0, \
                    "Arrival rate must be a non-negative float in rconst_cum!"

        tcum  = self.__tcum
        tcum += safediv(1.0, lamc)

        self.__tcum = tcum
        return tcum

    # end of rconst_cum

# ------------------------------------------------------------------------------

    def rexpo_cum(self, lam):
        """
        Generates - cumulatively - exponentially distributed interarrival times 
        with arrival rate (arrivale frequency) 'lam' from clock time = 0.0.

        NB. A SEPARATE STREAM MUST BE INSTANTIATED FOR EACH ONE OF THE VARIATES 
        THAT ARE GENERATED USING THIS METHOD, EACH WITH A SEPARATE SEED!! 
        """

        assert lam >= 0.0, \
                    "Arrival rate must be a non-negative float in rexpo_cum!"

        tcum  =  self.__tcum
        tcum -=  safediv(1.0, lam) * safelog(self.runif01())

        self.__tcum = tcum
        return tcum

    # end of rexpo_cum

# ------------------------------------------------------------------------------

    def rpiecexpo_cum(self, times, lamt):
        """
        Generates - cumulatively - piecewise exponentially distributed 
        interarrival times from clock time = 0.0 (the algorithm is taken 
        from Bratley, Fox & Schrage).
        
        'times' is a list or tuple containing the points at which the arrival 
        rate (= arrival frequency) changes (the first break time point must 
        be 0.0). 'lamt' is a list or tuple containing the arrival rates between 
        break points. The number of elements in 'times' must be one more than 
        the number of elements in 'lamt'!

        The algorithm cranking out the numbers is cyclic - the procedure 
        starts over from time zero when the last break point is reached. 
        THE PREVENT THE RESTART FROM TAKING PLACE, A (VERY) LARGE NUMBER 
        MUST BE GIVEN AS THE LAST BREAK POINT (the cyclicity is rarely 
        needed or desired in practice).
        
        NB. A SEPARATE STREAM MUST BE INSTANTIATED FOR EACH ONE OF THE VARIATES 
        THAT ARE GENERATED USING THIS METHOD, EACH WITH A SEPARATE SEED!! 
        """

        ntimes  = len(times)
        errtxt1 = "No. of arrival rates and no. of break points "
        errtxt2 = "are incompatible in rpiecexpo_cum!"
        errtext = errtxt1 + errtxt2
        assert len(lamt) == ntimes-1, errtext

        r = ntimes*[0.0]
        for k in range(1, ntimes):
            assert lamt[k-1] >= 0.0, \
                     "All lamt must be non-negative floats in rpiecexpo_cum!"
            r[k]  =  r[k-1]  +  lamt[k-1] * (times[k]-times[k-1])

        cumul  = self.__cumul
        cumul -= safelog(self.runif01())
        
        iseg  = 0
        while r[iseg+1] <= cumul:
            iseg = iseg + 1
        tcum  =  times[iseg] + safediv(cumul-r[iseg], lamt[iseg])

        tcum  =  kept_within(0.0, tcum)

        self.__cumul = cumul   # NB  THIS IS NOT CUMULATIVE TIME!
        return tcum

    # end of rpiecexpo_cum

# ------------------------------------------------------------------------------

    def rinhomexpo_cum(self, lamt, suplamt):
        """
        Generates - cumulatively - inhomogeneously exponentially distributed 
        interarrival times (due to an inhomogeneous Poisson process) from 
        clock time = 0.0 (the algorithm is taken from Bratley, Fox & Schrage).

        'lamt' is an externally defined function of clock time that returns 
        the present arrival rate (arrival frequency). 'suplamt' is another 
        externally defined function that returns the supremum of the arrival 
        rate over the REMAINING time from the present clock time.

        NB. A SEPARATE STREAM MUST BE INSTANTIATED FOR EACH ONE OF THE VARIATES
        THAT ARE GENERATED USING THIS METHOD, EACH WITH A SEPARATE SEED!! 
        """

        errtxt1  = "All values from suplamt must be "
        errtxt1 += "non-negative floats in rinhomexpo_cum!"
        errtxt2  = "All values from lamt must be "
        errtxt2 += "non-negative floats in rinhomexpo_cum!"

        ticum = self.__ticum

        while True:
            lamsup =  suplamt(ticum)
            assert lamsup >= 0.0, errortxt1
            u      =  self.runif01()
            ticum -=  safediv(1.0, lamsup) * safelog(self.runif01())
            lam    =  lamt(ticum)
            assert lam >= 0.0, errortxt2
            if u*lamsup <= lam:
                break

        self.__ticum = ticum
        return ticum

    # end of rinhomexpo_cum

# ------------------------------------------------------------------------------

    def reset(self):
        """
        Used to reset all cumulative parameters/attributes to zero.
        """

        self.__tcum   = 0.0   # For rexpo_cum
        self.__cumul  = 0.0   # For rpieceexpo_cum
        self.__ticum  = 0.0   # For rinhomexpo_cum

    # end of reset

# ------------------------------------------------------------------------------

# end of CumulRandomStream

# ------------------------------------------------------------------------------
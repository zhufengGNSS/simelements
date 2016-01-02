# delays.py
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

from math   import exp
from bisect import bisect

from numlib.solveq   import znewton
from misclib.numbers import kept_within

# ------------------------------------------------------------------------------

class ExpDelay:
    """
    Class used to create exponential delays in systems of ODEs in continuous 
    simulations.
        An exponential delay is characterized by tau*(dy1/dt) + y1 = y0; 1st 
    order, and 2nd order is provided by the above and tau*(dy2/dt) + y2 = y1; 
    etc etc.....  The class can be used for ODEs formulated using dictionaries 
    as well as lists/stacks. 
    
    The order of the delay is implied in the list input vector of the delay (the 
    number of elements of the state dictionary or list/stack that is included in
    the keys or indices is one more than the order: three elements implies 2nd 
    order, etc). state is the FULL present state dictionary or list/stack.

    An example of how it should be used for dictionaries (assumes deriv is a 
    dict):
        id = ExpDelay(['id0', 'id1','id2'], 0.50, 2.5) # Initiate instance obj.
        # Implies 2nd order and a response time of 2.5 to reach 50 % of a step 
        # input height.
        # And for each time step:
        deriv['id0']  = somethingwithoutdelay           # ..or what have you..  
        deriv['id1'], yDeriv['id2']  = id.delay(state)  # Compute for daughters
        somethingafterdelay  = deriv['id2']             # Finally after delay

    and for stacks (assuming that the indices of the elements in the state 
    vector involved in the delay are 7, 8 and 9):
        id  = ExpDelay([7, 8, 9], 0.50, 2.5)     # The rest as above
        # And for each time step (deriv = Stack() must be set for each step):
        deriv.push(somethingwithoutdelay)        # or what have you...
        deriv.push(id.delay(state))  # compute for daughters and append to stack
        somethingafterdelay  = deriv[-1]         # finally after delay

    If a list is used instead of a stack for yDeriv the whole thing gets a bit 
    more complicated since the indices must be kept track of in each time step -
    using stacks allows you to just use the 'push' method of the Stack class. 

    The advantage of using dicts is - besides more clarity due to the 
    possibility of using strings rather than indices to identify the 
    elements of the derivatives vector and the state vector - is that 
    the order in which the different elements of the vectors are treated in 
    the function called by the methods in ODEsSolution is not crucial!
    
    The rate constant object.lam = 1.0/tau is also available to the caller.
    """

    # Class variable used in methods
    __ONETHIRD = 0.3333333333333333333333333

# ------------------------------------------------------------------------------

    def __init__(self, sequence, fraction, time_to):
        """
        Inputs to the class are the list or tuple containing the keys or range 
        associated with the state variables involved in the specific delay 
        (that also determines the order of the delay). 
        
        The characteristic time (time constant) of the delay is computed for 
        later use: if "fraction" is the string 'peak' the time constant for 
        an exponential delay of the present order for the time "time_to" for 
        the response to a delta (spike/pulse/Dirac) input to reach its peak. 
        For "fraction" in [0.0, 1.0] "time_to" is used to compute the time 
        constant for the response to a step input to reach the given fraction 
        of the step height. (The time to the peak and the time to a certain 
        fraction are referred to as "time to rise point" in the error messaging)
        """

        self.__sequence = sequence
        self.__nstop    = len(sequence)
        self.__ndelay   = self.__nstop - 1

        assert time_to >= 0.0, \
               "time to rise point must be a non-negative float in ExpDelay!"

        if fraction == 'peak':
            tau = time_to / self.__ndelay

        else:
            assert 0.0 <= fraction and fraction <= 1.0, \
                "fraction must be in [0.0, 1.0] in ExpDelay!"

            q = 1.0 - fraction
            # ------------------------------------
            def _taudelay(theta):
                fi  = 0.0
                fid = 0.0
                expfactor  = exp(-theta)
                plusfactor = exp( theta)

                if self.__ndelay == 1:
                    fi  = q - expfactor
                    #fid = expfactor
                    fid = plusfactor*q - 1.0
                else:
                    a   = 1.0
                    sum = a
                    for k in range(1, self.__ndelay):
                        a   = a * theta / k
                        sum = sum + a
                    fi  = q - expfactor*sum
                    #fid = expfactor * a
                    fid = (plusfactor*q - sum) / a
                    #fid = fi / fid

                return fi, fid
            # -------------------------------------

            start  = self.__ndelay - ExpDelay.__ONETHIRD
            tchar  = znewton(_taudelay, start, 'ExpDelay')
            tau    = time_to / tchar
            tau    = kept_within(0.0, tau)


        self.lam  = 1.0 / tau

    # end of __init__

# ------------------------------------------------------------------------------

    def delay(self, state):
        """
        The method used to compute the list of values for the derivatives of the 
        elements involved in a specific delay. A list is returned which must be 
        used to fill the full derivatives dict, stack or list involved in the 
        solution of the ODE at hand.
        
        The function takes the full present state vector (a dict, a stack or 
        a list) as its sole input.
        """

        dlist  = []

        for k in range(1, self.__nstop):
            dlist.append(self.lam * (state[self.__sequence[k-1]] -
                                     state[self.__sequence[k]]))

        return dlist

    # end of delay

# ------------------------------------------------------------------------------

# end of ExpDelay

# ------------------------------------------------------------------------------

class StepDelay:
    """
    Class primarily designed to handle step delays in systems of ODEs in 
    continuous simulations, but it may be used for other computational purposes 
    as well. An step delay is characterized by y1(t) = y0(t-tau) where tau is 
    the time delay, and where y may be state variables OR derivatives OR 
    anything. The class may be used with dicts or with lists/stacks. 
    
    The following is an example of how it could be used (delay between 
    y['ident1'] and y['ident2'] in the dict formulation, or between y[1] 
    and y[2] in the list formulation):
        from delays import StepDelay
        y           = {}
        y['ident0'] = 0.0
        y['ident1'] = 0.0
        y['ident2'] = 0.0
        sequence    = ('ident1', 'ident2')
        tdelay      = 2.0
        one_two     = StepDelay(sequence, tdelay)
        for k in range(0, 100):
            t           = float(k)
            y['ident0'] = anything
            y['ident1'] = exp(t)
            y['ident2'] = one_two.delay(y, t)
    
    with a dict, and 
    
        from delays import StepDelay
        y        = [0.0, 0.0, 0.0]
        sequence = (1, 2)
        tdelay   = 2.0
        one_two  = StepDelay(sequence, tdelay)
        for k in range(0, 100):
            t    = float(k)
            y[0] = anything
            y[1] = exp(t)
            y[2] = one_two.delay(y, t)
    
    with a list.
    """
# ------------------------------------------------------------------------------

    def __init__(self, sequence, tdelay):
        """
        Initiates the object with the "parent-daughter" sequence involved in 
        the delay and the delay time. 
        """

        self.__sequence = sequence
        if len(sequence) != 2: \
                "there is something wrong with the sequence in StepDelay"

        assert tdelay >= 0.0, \
                  "delay time must be a positive float in StepDelay!"

        self.__tdelay  = tdelay
        self.__y0hist  = []
        self.__thist   = []

    # end of __init__

# ------------------------------------------------------------------------------

    def delay(self, y, time):
        """
        Creates the delay between the two members of the sequence given the full 
        state vector (dict/list/stack) and the time. The function uses linear 
        interpolation in the parent's temporal history to come up with the value 
        for the "daughter". 
        """

        self.__thist.append(time)
        self.__y0hist.append(y[self.__sequence[0]])

        backtime  = time - self.__tdelay

        if backtime < self.__thist[0]: return 0.0

        ix    = bisect(self.__thist, backtime)
        ixm1  = ix - 1
        thi   = self.__thist[ix]
        tlo   = self.__thist[ixm1]
        y0hi  = self.__y0hist[ix]
        y0lo  = self.__y0hist[ixm1]
        return y0lo + (y0hi-y0lo)*(backtime-tlo)/(thi-tlo)

    # end of delay

# ------------------------------------------------------------------------------

# end of StepDelay

# ------------------------------------------------------------------------------
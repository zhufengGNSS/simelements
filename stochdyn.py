# stochdyn.py
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

from dynamics import Dynamics

# ------------------------------------------------------------------------------

class StochDynamics(Dynamics):
    """
    May be used when noise is added to a problem, i. e. for stochastic 
    differential equations. All responsibility for handling the stochastics 
    is placed on a "noise model" function that is used together with the 
    derivative model normally used by the deterministic (non-stochastic) 
    solver methods.

    Instances are punched out with a construct like (the sequence of breakpoints
    is optional, cf. the tiptoe method/function in the Dynamics class):
        solution = StochDynamics(model, noisemodel, state [, breakpoints]) 
    where "model" is the name of the function of time and the state vector 
    that returns the values of the derivatives. "noisemodel" is the noise model 
    function and "state" can be a dict, a stack, a list or a 'd' array. The 
    solvers (there is only one at the moment) are called like 
        t = tstart
        while t <= tfin:
            tnext    = t + deltat
            t, state = solution.method(t, tnext) 
    where "method" is the name of the solver method; euler_maruyama is the 
    only method available at the moment. 

    StochDynamics inherits from the Dynamics class, and all the deterministic 
    methods in the latter are available here, a feature which makes it simple to
    switch between deterministic and stochastic mode in one single simulation.
    """
# ------------------------------------------------------------------------------

    def __init__(self, model, noisemodel, state, breakpoints=None):
        """
        May be used when noise is added to a problem, i. e. for stochastic 
        differential equations. All responsibility for handling the stochastics 
        is placed on a "noisemodel" function that is used together with the 
        derivative model normally used by the solver methods of the solvers 
        of the Dynamics and StiffDynamics classes.
        """

        # First take care of what is inherited:
        Dynamics.__init__(self, model, state, breakpoints)

        # Then add the separate noise model
        self.__noisemodel = noisemodel

    # end of __init__

# ------------------------------------------------------------------------------

    def euler_maruyama(self, t, tnext):
        """
        A fairly general system of SDEs - where "state" is multi-dimensional 
        (=vector-valued) - may be formulated using the Ito formalism as

           dstate = a(t, state)dt + B(t, state).dW + C(t, state).dJ

        where W are Wiener processes and J generalized Poisson jump processes. 
        In principle B and C have matrix character and also contain information 
        on the correlation between the different individual Wiener processes 
        and the different generalized Poisson jump processes. B and C must be 
        part of and handled by "noisemodel" as must the handling of the 
        vectors of the random increments of W and J. "a" is vector-valued as 
        in the case of systems of non-stochastic ODEs. The deterministic part 
        is solved as usual and the stochastic part is added in a second step 
        in this method, which uses the Euler-Maruyama solver formalism.
        
        For the jump process it must be remembered that all solvers use 
        discrete time stepping, implying that the number of Poisson jumps 
        may be > 1 during the finite deltat. Otherwise the jump process 
        may be prescribed in a number of ways.

        NB The present method can also be used for the Milstein scheme, but 
        all handling related to B and C and their derivatives as well as the 
        stochastic processes must be placed in "noisemodel". 
        """

        assert tnext > t, "time step must be positive in euler_maruyama!"

        deriv  = self._model(t, self._state)
        deltat = tnext - t
        noise  = self.__noisemodel(t, tnext, self._state)
        for n in self._sequence:
            self._state[n]  +=  deltat * deriv[n]
            self._state[n]  +=  noise[n]

        self._restart = False
    
        return t+deltat, self._state

    # end of euler_maruyama

# ------------------------------------------------------------------------------

# end of StochDynamics

# ------------------------------------------------------------------------------
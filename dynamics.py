# dynamics.py
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

from array import array
from copy  import deepcopy

from misclib.matrix   import Matrix
from numlib.matrixops import scaled, flattened
from misclib.numbers  import is_eveninteger, is_posinteger, is_nonneginteger
from machdep.machnum  import MACHEPS, SQRTMACHEPS, TWOMACHEPS, TINY
from misclib.errwarn  import Error, warn

# ------------------------------------------------------------------------------

class Dynamics():
    """
    Class for solving the initial value problem for systems of ordinary 
    differential equations. For uniformity all methods return - besides a 
    list/stack or dict of solutions at each time point - (a suggestion for) 
    the next time point in the time stepping, including those methods that are 
    dependent on externally controlled time stepping and do not have their 
    own time stepping. The methods of this class support state vectors which 
    are either dicts or lists/stacks or 'd' arrays. Method "matrixexp" is a 
    bit special and takes (=requires) a column vector from the misclib.Matrix 
    class as the input state vector.

    Instances are punched out with a construct like (breakpoints is optional, 
    cf tiptoe function):
        solution = Dynamics(model, state [, breakpoints]) 
    where "model" is the name of the function of the time and the state vector 
    that returns the values of the derivatives. "state" can be a dict, a stack, 
    a list, a 'd' array or a Matrix column vector (the only solver method 
    capable of handling the latter is matrixexp). The different solvers 
    (methods of solution) are called like 
        t = tstart
        while t <= tfin:
            tnext    = t + deltat
            t, state = solution.method(t, tnext) 
    where "method" is the name of the solver method; euler, runge_kutta4 etc. 

    If the state vector is the result put out by the program using this class, 
    the corresponding time to put out is "tnext" - if derivatives are the 
    result put out, then "t" is more correct, or it may even depend on whether 
    the solution method is explicit or implicit...

    The advantage in using dicts is - besides more clarity due to the 
    possibility of using text strings rather than indices to identify the 
    elements of the derivatives vector and the state vector - is that the 
    order in which the different elements of the vectors are treated in the 
    function called by the methods in Dynamics is not crucial!
    
    Even if some of the solver methods belonging to this class handles 
    reasonably stiff problems very well, you might consider the StiffDynamics 
    class for problems which are very stiff. The solver methods of the 
    StiffDynamics class are rather slow, though.

    The user if referred to Dahlquist, Bjorck & Anderson Ch. 8 for the theory 
    behind the majority of the methods used in this class.
    """

    # Class variables used in methods
    __ONE6TH        =  0.1666666666666666666666667    # 1.0 /  6.0
    __ONE12TH       =  0.08333333333333333333333333   # 1.0 / 12.0
    __ONE24TH       =  0.04166666666666666666666667   # 1.0 / 24.0
    __ONEP2MACHEPS  =  1.0 + TWOMACHEPS

# ------------------------------------------------------------------------------

    def __init__(self, model, state, breakpoints=None):
        """
        Initiate the model and the state dict/list/array, and initiate 
        history lists used by some of the methods. 
        """

        self._model = model
        self._state = state

        if isinstance(self._state, dict):
            self._sequence = state.keys()   # Creates an iterable, not a list
            self._mode     = 'dict'
        elif isinstance(self._state, (list, array, Matrix)):
            neqs           = len(state)
            self._sequence = range(0, neqs)  # An iterable, not a list
            self._neqs     = neqs
            self._mode     = 'list'
        else:
            errtxt1 = "input state vector must be a dict, a stack, "
            errtxt2 = "a list, a 'd' array or a Matrix!"
            raise Error(errtxt1+errtxt2)

        if self._mode == 'list' and not isinstance(self._state, Matrix):
            self._state = array('d', self._state)

        self._restart = False

        if breakpoints != None:
            try:
                len(breakpoints)
                self._breakpoints = list(breakpoints)
                self._breakpoints.sort()
            except TypeError:
                self._breakpoints = [breakpoints]

        self._derivp = array('d', [])   # For adams_bashforthX and abmX


    # end of __init__

# ------------------------------------------------------------------------------

    def euler(self, t, tnext):
        """
        Simple forward Euler stepping with constant time step. It is fast 
        since it only uses uses one computation of the model/derivative 
        function per time step, but the numerics may be problematic due to 
        the large discretization error - the Euler scheme is only first-order 
        correct. Trying to overcome this by using  very short time steps may 
        produce a large accumulated round-off error. But it is fast, and 
        sometimes useful. 

        """        
        assert tnext > t, "time step must be positive in euler!"

        deriv  = self._model(t, self._state)
        deltat = tnext - t
        for n in self._sequence:
            self._state[n] += deltat*deriv[n]
        tout = t + deltat

        self._restart = False

        return tout, self._state

    # end of euler

# ------------------------------------------------------------------------------

    def euler_plus(self, t, tnext):
        """
        The implicit trapezoidal method
                ynp1 - yn  -  0.5*h*(f(tn, yn) + f(tnp1, ynp1))  =  0 
        used in a single-step predictor-corrector scheme using the simple 
        Euler scheme for the predictor. It uses two computations of the 
        model/derivative function per time step. 
        """

        assert tnext > t, "time step must be positive in euler_plus!"


        deltat =  tnext - t
        h      =  0.5 * deltat

        # Initiate auxiliary dicts/arrays
        if self._mode == 'dict':
            s = {}; derivc = {}
        else:
            NN = float('nan')
            s =array('d', self._neqs*[NN]); derivc =array('d', self._neqs*[NN])

        # Go!
        deriv = self._model(t, self._state)
        for n in self._sequence:
            s[n]  =  self._state[n] + deltat*deriv[n]
        derivc = self._model(tnext, s)
        for n in self._sequence:
            self._state[n] +=  h * (deriv[n]+derivc[n])

        self._state = deepcopy(s)
        del s

        self._restart = False
   
        return t+deltat, self._state

    # end of euler_plus

# ------------------------------------------------------------------------------

    def euler_ex(self, t, tnext):
        """
        An enhanced Euler solver with one Richardson extrapolation step. 
        The solution is first computed using the full time step, then twice 
        consequtively using the halved time step. The the solutions are 
        combined using the Richardson extrapolation scheme. The model
        /derivative function is called twice. 
        """        
        
        assert tnext > t, "time step must be positive in euler_ex!"


        # Initiate auxiliary dicts/arrays
        if self._mode == 'dict':  sfull = {}
        else:                     sfull = array('d', self._neqs*[float('nan')])

        # Get the solution with one full time step:
        deriv = self._model(t, self._state)
        deltat = tnext - t
        for n in self._sequence:
            sfull[n]  = self._state[n] + deltat*deriv[n]

        # Get the solution with two halved time steps:
        thalf  = t + 0.5*deltat
        deltat = tnext - thalf
        for n in self._sequence:
            self._state[n] += deltat*deriv[n]  # Same deriv as for the full step
        deriv = self._model(thalf, self._state) # New deriv based on t + 0.5*dt
        for n in self._sequence:
            self._state[n] += deltat*deriv[n]

        # Use the two solutions for one Richardson extrapolation step 
        # (first order/h-dependent): 
        for n in self._sequence:
            self._state[n]  = 2.0*self._state[n] - sfull[n]

        tout = thalf + deltat

        self._restart = False

        return tout, self._state

    # end of euler_ex

# ------------------------------------------------------------------------------

    def adams_bashforth2(self, t, tnext):
        """
        Adams-Bashforth 2:nd order, a linear multistep method. The method 
        uses the derivative-function-value history and is fast since it only 
        uses one computation of the model/derivative function per time step. 
        Some of the discretization error problems are reduced. The method 
        should work well for smooth problems, but adams_bashforth4 would 
        normally be recommended (this method is used by other methods, a 
        fact that is the main reason for it being here in the first place).

        This method uses the first time step twice to build its history at 
        the very outset. After that the historic points will differ from 
        one another.
        """

        assert tnext > t, "time step must be positive in adams_bashforth2!"


        deltat =  tnext - t
        h  =  0.5 * deltat
        a  =  array('d', (3.0, -1.0))

        if len(self._derivp) == 0:
            deriv0        = self._model(t, self._state)
            self._derivp = 2*[deriv0]

        # Initiate auxiliary dicts/arrays
        if self._mode == 'dict':  deriv = {}
        else:                     deriv = array('d', self._neqs*[float('nan')])

        for n in self._sequence:
            self._state[n]  +=  h * (a[0]*self._derivp[-1][n] + \
                                     a[1]*self._derivp[-2][n])

        tnew  = t + deltat
        deriv = self._model(tnew, self._state)

        if not self._restart:
            del self._derivp[0]
            self._derivp.append(deriv)
        else:
            self._derivp = array('d', [])

        self._restart = False
   
        return tnew, self._state

    # end of adams_bashforth2

# ------------------------------------------------------------------------------

    def adams_bashforth3(self, t, tnext):
        """
        Adams-Bashforth 3:rd order, a linear multistep method. The method 
        uses the derivative-function-value history and is fast since it only 
        uses one computation of the model/derivative function per time step. 
        Some of the discretization error problems are reduced. The method 
        should work well for smooth problems, but adams_bashforth4 would 
        normally be recommended (this method is used by adams_bashforth4, a 
        fact that is the main reason for it being here in the first place). 

        This method uses adams_bashforth2 to build up its history before 
        it has created its own history. Cf. adams_bashforth2 for details.
        """

        assert tnext > t, "time step must be positive in adams_bashforth3!"


        deltat =  tnext - t
        h  =  Dynamics.__ONE12TH * deltat
        a  =  array('d', (23.0, -16.0, 5.0))

        if len(self._derivp) < 3:
            deriv0 = self._model(t, self._state)
            tnew, self._state = self.adams_bashforth2(t, tnext)
            self._derivp.insert(0, deriv0)
            return tnew, self._state

        # Initiate auxiliary dicts/arrays
        if self._mode == 'dict':  deriv = {}
        else:                     deriv = array('d', self._neqs*[float('nan')])

        for n in self._sequence:
            self._state[n]  +=  h * (a[0]*self._derivp[-1][n] + \
                                     a[1]*self._derivp[-2][n] + \
                                     a[2]*self._derivp[-3][n])

        deriv = self._model(tnext, self._state)

        if not self._restart:
            del self._derivp[0]
            self._derivp.append(deriv)
        else:
            self._derivp = array('d', [])

        self._restart = False
   
        return t+deltat, self._state

    # end of adams_bashforth3

# ------------------------------------------------------------------------------

    def adams_bashforth4(self, t, tnext):
        """
        Adams-Bashforth 4:th order, a linear multistep method. The method 
        uses the derivative-function-value history and is fast since it only 
        uses one computation of the model/derivative function per time step. 
        Some of the discretization error problems are reduced. The method 
        should work well for smooth problems. 
        
        This method uses adams_bashforth3 to build up its history before 
        it has created its own history. Cf. adams_bashforth3 for details.
        """        
        
        assert tnext > t, "time step must be positive in adams_bashforth4!"


        deltat =  tnext - t
        h  =  Dynamics.__ONE24TH * deltat
        a  =  array('d', (55.0, -59.0, 37.0, -9.0))

        if len(self._derivp) < 4:
            deriv0 = self._model(t, self._state)
            tnew, self._state = self.adams_bashforth3(t, tnext)
            self._derivp.insert(0, deriv0)
            return tnew, self._state

        # Initiate auxiliary dicts/arrays
        if self._mode == 'dict':  deriv = {}
        else:                     deriv = array('d', self._neqs*[float('nan')])

        for n in self._sequence:
            self._state[n]  +=  h * (a[0]*self._derivp[-1][n] + \
                                     a[1]*self._derivp[-2][n] + \
                                     a[2]*self._derivp[-3][n] + \
                                     a[3]*self._derivp[-4][n])

        deriv = self._model(tnext, self._state)

        if not self._restart:
            del self._derivp[0]
            self._derivp.append(deriv)
        else:
            self._derivp = array('d', [])

        self._restart = False
   
        return t+deltat, self._state

    # end of adams_bashforth4

# ------------------------------------------------------------------------------

    def heun(self, t, tnext):
        """
        Heun's method is really second-order Runge-Kutta. It uses two 
        computations of the model/derivative function per time step. 
        """        
        
        assert tnext > t, "time step must be positive in heun!"


        # Initiate auxiliary dicts/arrays:
        if self._mode == 'dict':
            k1h  = {}; k2  = {}
            yk   = {}
        else:
            NN  = float('nan')
            k1h = array('d', self._neqs*[NN]); k2 = array('d', self._neqs*[NN])
            yk  = array('d', self._neqs*[NN])


        deltat   =  tnext - t
        deltath  =  0.5 * deltat
        th       =  t + deltath

        deriv = self._model(t, self._state)
        for n in self._sequence:
            k1h[n]  =  deltath * deriv[n]
            yk[n]   =  self._state[n] + k1h[n]

        deriv = self._model(th, yk)
        for n in self._sequence:
            k2[n]   =  deltat * deriv[n]


        # Final summation and advancement of time
        for n in self._sequence: 
            self._state[n] += k2[n]

        self._restart = False

        return t+deltat, self._state

    # end of heun

# ------------------------------------------------------------------------------

    def runge_kutta4(self, t, tnext):
        """
        Standard fourth-order Runge-Kutta. It uses four computations of 
        the model/derivative function per time step. 
        """      
        
        assert tnext > t, "time step must be positive in runge_kutta4!"


        # Initiate auxiliary dicts/arrays:
        if self._mode == 'dict':
            k1h  = {}; k2h  = {}; k3h  = {}; k4h  = {}
            yk   = {}
        else:
            NN  = float('nan')
            k1h = array('d', self._neqs*[NN]); k2h = array('d', self._neqs*[NN])
            k3h = array('d', self._neqs*[NN]); k4h = array('d', self._neqs*[NN])
            yk  = array('d', self._neqs*[NN])


        # Initiate useful time slices:
        deltat   =  tnext - t
        deltath  =  0.5 * deltat
        deltat6  =  Dynamics.__ONE6TH * deltat
        th       =  t + deltath

        # OK, go on!
        deriv = self._model(t, self._state)
        for n in self._sequence:
            k1h[n] = deriv[n]
            yk[n]  = self._state[n] + deltath*k1h[n]

        deriv = self._model(th, yk)
        for n in self._sequence: 
            k2h[n] = deriv[n]
            yk[n]  = self._state[n] + deltath*k2h[n]
            
        deriv = self._model(th, yk)
        for n in self._sequence: 
            k3h[n] = deriv[n]
            yk[n]  = self._state[n] + deltat*k3h[n]

        deriv = self._model(tnext, yk)
        for n in self._sequence: 
            k4h[n] = deriv[n]

    
        # Final summation and advancement of time
        for n in self._sequence: 
            self._state[n] += deltat6*(k1h[n] + 2.0*(k2h[n] + k3h[n]) + k4h[n])

        self._restart = False

        return tnext, self._state

    # end of runge_kutta4

# ------------------------------------------------------------------------------

    def rke4(self, t, tnext):
        """
        Runge-Kutta - England 4th order version. The algorithm is the same 
        as the one used in GASP IV using two halved steps, with the difference 
        that adaptive step size control is not used. The model/derivative 
        function is computed eight times per full time step. 
        """        
        
        assert tnext > t, "time step must be positive in rke4!"


        # Initiate auxiliary dicts/arrays:
        if self._mode == 'dict':
            a1  = {}; a2  = {}; a3  = {}; a4  = {}
            a5  = {}; a6  = {}; a7  = {}; a8  = {}
            yk  = {}; y1  = {}
        else:
            NN  = float('nan')
            a1  = array('d', self._neqs*[NN]); a2  = array('d', self._neqs*[NN])
            a3  = array('d', self._neqs*[NN]); a4  = array('d', self._neqs*[NN])
            a5  = array('d', self._neqs*[NN]); a6  = array('d', self._neqs*[NN])
            a7  = array('d', self._neqs*[NN]); a8  = array('d', self._neqs*[NN])
            yk  = array('d', self._neqs*[NN]); y1  = array('d', self._neqs*[NN])


        h    =  0.5 * (tnext-t)
        hh   =  0.5 * h
        t14  =  t + hh
        t12  =  t + h
        t34  =  t12 + hh

        # 1st phase -----------------------
        deriv = self._model(t, self._state)
        for n in self._sequence:
            a1[n]  =  h * deriv[n]
            yk[n]  =  self._state[n] + 0.5*a1[n]

        deriv = self._model(t14, yk)
        for n in self._sequence: 
            a2[n]  =  h * deriv[n]
            yk[n]  =  self._state[n] + 0.25*(a1[n] + a2[n])
            
        deriv = self._model(t14, yk)
        for n in self._sequence: 
            a3[n]  =  h * deriv[n]
            yk[n]  =  self._state[n] - a2[n] + 2.0*a3[n]

        deriv = self._model(t12, yk)
        for n in self._sequence: 
            a4[n]  =  h * deriv[n]

        for n in self._sequence: 
            y1[n]  =  self._state[n] + \
                              Dynamics.__ONE6TH*(a1[n] + 4.0*a3[n] + a4[n])

        # 2nd phase -----------------------
        deriv = self._model(t12, y1)
        for n in self._sequence:
            a5[n]  =  h * deriv[n]
            yk[n]  =  y1[n] + 0.5*a5[n]

        deriv = self._model(t34, yk)
        for n in self._sequence:
            a6[n]  =  h * deriv[n]
            yk[n]  =  y1[n] + 0.25*(a5[n] + a6[n])

        deriv = self._model(t34, yk)
        for n in self._sequence:
            a7[n]  =  h * deriv[n]
            yk[n]  =  y1[n] - a6[n] + 2.0*a7[n]

        deriv = self._model(tnext, yk)
        for n in self._sequence:
            a8[n]  =  h * deriv[n]


        # Final summation and advancement of time ---
        for n in self._sequence:
            self._state[n]  =  y1[n] + \
                            Dynamics.__ONE6TH*(a5[n] + 4.0*a7[n] + a8[n])

        self._restart = False

        return tnext, self._state

    # end of rke4

# ------------------------------------------------------------------------------

    def rkf45(self, t, tnext, individ=False, tolf=(0.5**9)*SQRTMACHEPS, \
                              factor=0.25,   maxniter=10):
        """
        The Runge-Kutta-Fehlberg algorithm (inspired by Forsythe-Malcolm-Moler):
        fourth and fifth order Runge-Kutta with Fehlberg's coefficients is used 
        in an adaptive step size scheme that uses the difference between fourth 
        and fifth order solutions to control the stepping. Six computations of 
        the model/derivative function are used in each iteration. Efficient, 
        accurate and also stable for most problems.
        
        When individ==False, 'tolf' is the maximum allowed fractional difference 
        between the sum of the absolute values of the state vector variables 
        for the fourth and fifth order solutions. When individ==True, the 
        comparison between the fourth and fifth order solutions is made for 
        each of the stage variables. 'factor' is the factor by which the 
        present time step is multiplied when the fractional difference is 
        greater than tolf. 'maxniter' is the maximum number of iterations made. 
        The computation is NOT stopped if convergence is not reached, but a 
        warning will be issued to stdout.
        
        NB. This method uses adaptive step size control which means that it may 
        compute the model/derivative function for times prior to what was just 
        used in the same single call. This might cause problems if there are 
        setups in the model that uses values of the derivatives or states from 
        prior calls. Short time steps might make this a minor problem, but "ten 
        cuidado"...
        """        
        
        assert tnext > t, "time step must be positive in rkf45!"

        if tolf < MACHEPS:
            tolf = MACHEPS
            wtext  = "tolerance smaller than machine epsilon is not recommended"
            wtext += " in rkf45. Machine epsilon is used instead"
            warn(wtext)

        assert is_posinteger(maxniter), \
                            "maxniter must be an integer > 0 in rkf45!"

        C0  = 0.11574074074074074       # 25.0/216.0
        C1  = 0.0
        C2  = 0.54892787524366471       # 1408.0/2565.0
        C3  = 0.53533138401559455       # 2197.0/4104.0
        C4  = - 0.2                     # -1.0/5.0
        C5  = 0.0

        CS0 = 0.1185185185185185185     # 16.0/135.0
        CS1 = 0.0
        CS2 = 0.51898635477582844       # 6656.0/12825.0
        CS3 = 0.50613149034201665       # 28561.0/56430.0
        CS4 = -0.18                     # -9.0/50.0
        CS5 = 0.03636363636363636       #  2.0/55.0

        A0 = 0.0
        A1 = 0.25                      # 1.0/4.0
        A2 = 0.375                     # 3.0/8.0
        A3 = 0.92307692307692313       # 12.0/13.0
        A4 = 1.0
        A5 = 0.5                       # 1.0/2.0

        B1 = array('d', (0.25, float('nan')))
             # 1.0/4.0, and to make it indexable...
        B2 = array('d', (0.09375, 0.28125))
             # (3.0/32.0, 9.0/32.0)
        B3 = array('d', (0.87938097405553028, -3.2771961766044608, \
                         3.3208921256258535))
             # (1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0)
        B4 = array('d', (2.0324074074074074, -8.0, 7.1734892787524362, \
                        -0.20589668615984405))
             # (439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0)
        B5 = array('d', (-0.2962962962962963, 2.0, -1.3816764132553607, \
                          0.45297270955165692, -0.275))
             # (-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0)


        # Initiate auxiliary dicts/arrays:
        if self._mode == 'dict':
            k0  = {}; k1  = {}; k2  = {}; k3  = {}; k4  = {}; k5  = {}
            yk  = {}
            s   = {}; ss  = {}
        else:
            NN  = float('nan')
            k0  = array('d', self._neqs*[NN]); k1  = array('d', self._neqs*[NN])
            k2  = array('d', self._neqs*[NN]); k3  = array('d', self._neqs*[NN])
            k4  = array('d', self._neqs*[NN]); k5  = array('d', self._neqs*[NN])
            yk  = array('d', self._neqs*[NN])
            s   = array('d', self._neqs*[NN]); ss  = array('d', self._neqs*[NN])


        # OK:
        h     = tnext - t
        tout  = tnext
        niter = 0
        while True:
            niter = niter + 1
            if niter > maxniter:
                wtext1 = "step size control did not converge in rkf45 for time "
                wtext2 = "= " + str(t) + ". Try changing tolerance or maxniter"
                warn(wtext1+wtext2)
                break

            deriv = self._model(t+A0*h, self._state)
            for n in self._sequence:
                k0[n]  = h * deriv[n]
                summ   = B1[0]*k0[n]
                #yk[n]  = self._state[n]
                yk[n]  = self._state[n] + summ

            deriv = self._model(t+A1*h, yk)
            for n in self._sequence:
                k1[n]  = h * deriv[n]
                summ   = B2[0]*k0[n] + B2[1]*k1[n]
                yk[n]  = self._state[n] + summ

            deriv = self._model(t+A2*h, yk)
            for n in self._sequence: 
                k2[n]  = h * deriv[n]
                summ   = B3[0]*k0[n] + B3[1]*k1[n] + B3[2]*k2[n]
                yk[n]  = self._state[n] + summ
            
            deriv = self._model(t+A3*h, yk)
            for n in self._sequence: 
                k3[n]  = h * deriv[n]
                summ   = B4[0]*k0[n] + B4[1]*k1[n] + B4[2]*k2[n] + B4[3]*k3[n]
                yk[n]  = self._state[n] + summ

            #deriv = self._model(t+A4*h, yk)
            deriv = self._model(tout, yk)    # Better for the tiptoe function
            for n in self._sequence: 
                k4[n]  = h * deriv[n]
                summ   = B5[0]*k0[n] + B5[1]*k1[n] + B5[2]*k2[n] + \
                                       B5[3]*k3[n] + B5[4]*k4[n]
                yk[n]  = self._state[n] + summ

            deriv = self._model(t+A5*h, yk)
            for n in self._sequence: 
                k5[n]  = h * deriv[n]

            # Test the solution
            converged = True
            if not individ:
                ass  = 0.0
                asss = 0.0
            for n in self._sequence:
                s[n]    = C0*k0[n]  + C1*k1[n]  + C2*k2[n]  + \
                          C3*k3[n]  + C4*k4[n]  + C5*k5[n]
                s[n]   += self._state[n]
                ss[n]   = CS0*k0[n] + CS1*k1[n] + CS2*k2[n] + \
                          CS3*k3[n] + CS4*k4[n] + CS5*k5[n]
                ss[n]  += self._state[n]
                if individ:
                    if abs(s[n]-ss[n]) > tolf*abs(ss[n]): converged = False
                else:
                    ass  += abs(s[n])
                    asss += abs(ss[n])
            if not individ:
                if abs(ass-asss) > tolf*asss: converged = False
            if not converged:
                h    = factor * h
                tout = t + h
            else:
                break
    
        # Final solution and advancement of time
        self._state = deepcopy(ss)

        self._restart = False

        return tout, self._state

    # end of rkf45

# ------------------------------------------------------------------------------

    def rkck45(self, t, tnext, individ=False, tolf=(0.5**12)*SQRTMACHEPS, \
                               factor=0.25,   maxniter=10):
        """
        This solver uses the Runge-Kutta-Cash-&-Karp algorithm (cf. 
        http://en.wikipedia.org/wiki/Cash-Karp): fourth and fifth order 
        Runge-Kutta with Cash & Karps's coefficients is used in an adaptive 
        step size scheme that uses the difference between fourth and fifth 
        order solutions to control the stepping. Six computations of the 
        model/derivative function are used in each iteration. Very 
        efficient, accurate and also stable for most problems.
        
        When individ==False, 'tolf' is the maximum allowed fractional difference 
        between the sum of the absolute values of the state vector variables 
        for the fourth and fifth order solutions. When individ==True, the 
        comparison between the fourth and fifth order solutions is made for 
        each of the stage variables. 'factor' is the factor by which the 
        present time step is multiplied when the fractional difference is 
        greater than tolf. 'maxniter' is the maximum number of iterations made. 
        The computation is NOT stopped if convergence is not reached, but a 
        warning will be issued to stdout.
        
        NB. This method uses adaptive step size control which means that it may 
        compute the model/derivative function for times prior to what was just 
        used in the same single call. This might cause problems if there are 
        setups in the model that uses values of the derivatives or states from 
        prior calls. Short time steps might make this a minor problem, but 
        "ten cuidado"...
        """        
        
        assert tnext > t, "time step must be positive in rkck45!"

        if tolf < MACHEPS:
            tolf = MACHEPS
            wtext  = "tolerance smaller than machine epsilon is not recommended"
            wtext += "  in rkck45. Machine epsilon is used instead"
            warn(wtext)

        assert is_posinteger(maxniter), \
                            "maxniter must be a positive integer in rkck45!"

    
        C0  = 0.097883597883597878      # 37.0/378.0
        C1  = 0.0
        C2  = 0.40257648953301128       # 250.0/621.0
        C3  = 0.21043771043771045       # 125.0/594.0
        C4  = 0.0
        C5  = 0.28910220214568039       # 512.0/1771.0

        CS0 = 0.10217737268518519       # 2825.0/27648.0
        CS1 = 0.0
        CS2 = 0.38390790343915343       # 18575.0/48384.0
        CS3 = 0.24459273726851852       # 13525.0/55296.0
        CS4 = 0.019321986607142856      # 277.0/14336.0
        CS5 = 0.25                      # 1.0/4.0

        A0 = 0.0
        A1 = 0.2                       # 1.0/5.0
        A2 = 0.3                       # 3.0/10.0
        A3 = 0.6                       # 3.0/5.0
        A4 = 1.0
        A5 = 0.875                     # 7.0/8.0

        B1 = array('d', (0.2, float('nan')))
             # 1.0/5.0, and to make it indexable...
        B2 = array('d', (0.075, 0.225))
             # (3.0/40.0, 9.0/40.0)
        B3 = array('d', (0.3, -0.9, 1.2))
             # (3.0/10.0, -9.0/10.0, 6.0/5.0)
        B4 = array('d', (-0.20370370370370370, 2.5, -2.5925925925925926, 
                          1.2962962962962963))
             # (-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0)
        B5 = array('d', (0.029495804398148147, 0.341796875, \
                         0.041594328703703706, 0.40034541377314814, \
                         0.061767578125))
             # (1631.0/55296.0, 175.0/512.0, 575.0/13824.0, \
             #                44275.0/110592.0, 253.0/4096.0)


        # Initiate auxiliary dicts/arrays:
        if self._mode == 'dict':
            k0  = {}; k1  = {}; k2  = {}; k3  = {}; k4  = {}; k5  = {}
            yk  = {}
            s   = {}; ss  = {}
        else:
            NN  = float('nan')
            k0  = array('d', self._neqs*[NN]); k1  = array('d', self._neqs*[NN])
            k2  = array('d', self._neqs*[NN]); k3  = array('d', self._neqs*[NN])
            k4  = array('d', self._neqs*[NN]); k5  = array('d', self._neqs*[NN])
            yk  = array('d', self._neqs*[NN])
            s   = array('d', self._neqs*[NN]); ss  = array('d', self._neqs*[NN])


        h     = tnext - t
        tout  = tnext
        niter = 0
        while True:
            niter = niter + 1
            if niter > maxniter:
                wtext1 = "step size control did not converge in rkck45 for time"
                wtext2 = " = " + str(t) + ". Try changing tolerance or maxniter"
                warn(wtext1+wtext2)
                break

            tau   = t + A0*h
            deriv = self._model(tau, self._state)
            for n in self._sequence:
                k0[n]  = h * deriv[n]
                summ   = B1[0]*k0[n]
                #yk[n]  = self._state[n]
                yk[n]  = self._state[n] + summ

            tau   = t + A1*h
            deriv = self._model(tau, yk)
            for n in self._sequence:
                k1[n]  = h * deriv[n]
                summ   = B2[0]*k0[n] + B2[1]*k1[n]
                yk[n]  = self._state[n] + summ

            tau   = t + A2*h
            deriv = self._model(tau, yk)
            for n in self._sequence: 
                k2[n]  = h * deriv[n]
                summ   = B3[0]*k0[n] + B3[1]*k1[n] + B3[2]*k2[n]
                yk[n]  = self._state[n] + summ

            tau   = t + A3*h
            deriv = self._model(tau, yk)
            for n in self._sequence: 
                k3[n]  = h * deriv[n]
                summ   = B4[0]*k0[n] + B4[1]*k1[n] + B4[2]*k2[n] + B4[3]*k3[n]
                yk[n]  = self._state[n] + summ

            #deriv = self._model(t+A4*h, yk)
            deriv = self._model(tout, yk)   # Better for the tiptoe function
            for n in self._sequence: 
                k4[n]  = h * deriv[n]
                summ   = B5[0]*k0[n] + B5[1]*k1[n] + B5[2]*k2[n] + \
                                       B5[3]*k3[n] + B5[4]*k4[n]
                yk[n]  = self._state[n] + summ

            tau   = t + A5*h
            deriv = self._model(tau, yk)
            for n in self._sequence: 
                k5[n]  = h * deriv[n]

            # Test the solution
            converged = True
            if not individ:
                ass  = 0.0
                asss = 0.0
            for n in self._sequence:
                s[n]    = C0*k0[n]  + C1*k1[n]  + C2*k2[n]  + \
                          C3*k3[n]  + C4*k4[n]  + C5*k5[n]
                s[n]   += self._state[n]
                ss[n]   = CS0*k0[n] + CS1*k1[n] + CS2*k2[n] + \
                          CS3*k3[n] + CS4*k4[n] + CS5*k5[n]
                ss[n]  += self._state[n]
                if individ:
                    if abs(s[n]-ss[n]) > tolf*abs(ss[n]): converged = False
                else:
                    ass  += abs(s[n])
                    asss += abs(ss[n])
            if not individ:
                if abs(ass-asss) > tolf*asss: converged = False
            if not converged:
                h    = factor * h
                tout = t + h
            else:
                break
    
        # Final solution and advancement of time
        self._state = deepcopy(ss)

        self._restart = False

        return tout, self._state

    # end of rkck45

# ------------------------------------------------------------------------------

    def modmidpoint(self, t, tnext, nsteps=4):
        """
        W.B. Gragg's modified midpoint method. It uses nsteps computations of 
        the model/derivative function for each time step. 
        """        
        
        assert tnext > t, "time step must be positive in modmidpoint!"
        assert is_eveninteger(nsteps) and nsteps >= 2, \
                "Number of steps must be an even integer >= 2 in modmidpoint!"


        # Initiate auxiliary dicts/arrays
        if self._mode == 'dict':
            yn = {}; ynp1 = {}
        else:
            NN = float('nan')
            yn = array('d', self._neqs*[NN]); ynp1 = array('d', self._neqs*[NN])

        deltat = tnext - t
        h      = deltat / float(nsteps)
        twoh   = 2.0 * h

        ynm1 = deepcopy(self._state)                   # y0

        deriv = self._model(t, ynm1)
        for n in self._sequence:
            yn[n]  =  ynm1[n] + h*deriv[n]             # y1

        for k in range(1, nsteps):
            tau    = t + k*h
            tau    = min(tau, tnext)
            deriv = self._model(t+k*h, yn)
            for n in self._sequence:
                ynp1[n]  =  ynm1[n] + twoh*deriv[n]    # y2 etc
            ynm1 = deepcopy(yn)
            yn   = deepcopy(ynp1)

        deriv = self._model(tnext, yn)
        for n in self._sequence:
            self._state[n]  =  0.5 * (yn[n] + ynm1[n] + h*deriv[n])


        self._restart = False
    
        return t+deltat, self._state

    # end of modmidpoint

# ------------------------------------------------------------------------------

    def modmidpoint_ex(self, t, tnext, nsteps=2):
        """
        The modified midpoint method using one Richardson extrapolation 
        step, so it really uses nsteps + 2*nsteps computations of the 
        model/derivative function for each time step! 
        """        
        
        assert tnext > t, "time step must be positive in modmidpoint_ex!"
        assert is_eveninteger(nsteps) and nsteps >= 2, \
              "Number of steps must be an even integer >= 2 in modmidpoint_ex!"


        deltat = tnext - t

        # -------------------------------------------------------
        def _mmp(nstps):
            h    = deltat / float(nstps)
            twoh = 2.0 * h
            # Initiate auxiliary dicts/arrays
            if self._mode == 'dict':
                yn   = {}
                ynp1 = {}
            else:
                yn   = array('d', self._neqs*[float('nan')])
                ynp1 = array('d', self._neqs*[float('nan')])
            ynm1 = deepcopy(self._state)                   # y0

            deriv = self._model(t, ynm1)
            for n in self._sequence:
                yn[n]  =  ynm1[n] + h*deriv[n]             # y1

            for k in range(1, nstps):
                tau    = t + k*h
                tau    = min(tau, tnext)
                deriv = self._model(t+k*h, yn)
                for n in self._sequence:
                    ynp1[n]  =  ynm1[n] + twoh*deriv[n]    # y2 etc
                ynm1 = deepcopy(yn)
                yn   = deepcopy(ynp1)

            deriv = self._model(tnext, yn)
            for n in self._sequence:
                yn[n]  =  0.5 * (yn[n] + ynm1[n] + h*deriv[n])

            return yn
        # -------------------------------------------------------

        ynfull       = _mmp(nsteps)      # Full step size
        self._state  = _mmp(2*nsteps)    # Half step size

        # One Richardson extrapolation step (second order/h^2-dependent)
        for n in self._sequence:
            self._state[n]  =  (4.0*self._state[n] - ynfull[n]) / 3.0

        self._restart = False
    
        return t+deltat, self._state

    # end of modmidpoint_ex

# ------------------------------------------------------------------------------

    def trapezoid(self, t, tnext, tolf=SQRTMACHEPS, maxniter=4):
        """
        The implicit trapezoidal method:
              ynp1 - yn  -  0.5*h*(f(tn, yn) + f(tnp1, ynp1))  =  0 
        used in a predictor-corrector scheme using the simple Euler 
        scheme for the predictor. 

        'tolf' is the maximum allowed fractional difference between the sum of 
        the absolute values of the state vector variables for a prediction and 
        a correction. 'maxniter' is the maximum number of iterations made. 
        The computation is NOT stopped if convergence is not reached, but a 
        warning will be issued to stdout.
        """        
        
        assert is_nonneginteger(maxniter), \
         "max number of iterations must be a non-negative integer in trapezoid!"
        assert tnext > t, "time step must be positive in trapezoid!"
        if tolf < MACHEPS:
            tolf = MACHEPS
            wtext  = "tolerance smaller than machine epsilon is not recommended"
            wtext += " in trapezoid. Machine epsilon is used instead"
            warn(wtext)


        # Initiate auxiliary dicts/arrays
        if self._mode == 'dict':
            s = {}; derivc = {}
        else:
            NN = float('nan')
            s =array('d', self._neqs*[NN]); derivc =array('d', self._neqs*[NN])

        deltat =  tnext - t
        h      =  0.5 * deltat

        asump = 0.0
        iter  = False
        deriv = self._model(t, self._state)
        for n in self._sequence:
            s[n]   = self._state[n] + deltat*deriv[n]
            asump  +=  abs(s[n])
        for k in range(0, maxniter):
            derivc = self._model(tnext, s)
            asumc   = 0.0
            for n in self._sequence:
                s[n]  = self._state[n] + h*(deriv[n]+derivc[n])
                asumc += abs(s[n])
            if abs(asumc-asump) <= tolf*asumc:
                iter = True
                break
            else:
                asump = asumc

        if not iter:
            wtext1 = "iterations did not converge in trapezoid for time = "
            wtext2 = str(t) + ". Try changing tolerance or maxniter"
            warn(wtext1+wtext2)

        self._state = deepcopy(s)
        del s

        self._restart = False
   
        return t+deltat, self._state

    # end of trapezoid

# ------------------------------------------------------------------------------

    def abm2(self, t, tnext, tolf=SQRTMACHEPS, maxniter=4):
        """
        Adams-Bashforth-Moulton 2:nd order, a predictor-corrector method. The 
        algorithm is claimed to be more accurate for a reasonably large number 
        of iterations but is also claimed to be more stable for a smaller number 
        of iterations. Different orders of Adams-Bashforth-Moulton may have 
        different accuracy and stability properties - this is the reason for 
        three abmX methods being here. abm2 uses one computation of the 
        model/derivative function per iteration.
        
        'tolf' is the maximum allowed fractional difference between the sum of 
        the absolute values of the state vector variables for a prediction and 
        a correction. 'maxniter' is the maximum number of iterations made. 
        The computation is NOT stopped if convergence is not reached, but a 
        warning will be issued to stdout.

        This method uses the first time step twice to build its history at 
        the very outset. After that the historic points will differ from 
        one another.
        """

        assert is_nonneginteger(maxniter), \
             "max number of iterations must be a non-negative integer in abm2!"
        assert tnext > t, "time step must be positive in abm2!"
        if tolf < MACHEPS:
            tolf = MACHEPS
            wtext1 = "tolerance smaller than machine epsilon is not"
            wtext2 = " recommended in abm2. Machine epsilon is used instead"
            warn(wtext1+wtext2)


        deltat =  tnext - t
        h  =  0.5 * deltat
        a  =  array('d', (3.0, -1.0))
        b  =  array('d', (1.0,  1.0))

        if len(self._derivp) == 0:
            deriv0       = self._model(t, self._state)
            self._derivp = 2*[deriv0]

        # Initiate auxiliary dicts/arrays
        if self._mode == 'dict':
            s = {}; derivc = {}
        else:
            NN=float('nan')
            s =array('d', self._neqs*[NN]); derivc =array('d', self._neqs*[NN])

        asump = 0.0
        iter  = False
        for n in self._sequence:
            s[n] = self._state[n] + h * (a[0]*self._derivp[-1][n] + \
                                         a[1]*self._derivp[-2][n])
            asump += abs(s[n])
        for k in range(0, maxniter):
            derivc = self._model(tnext, s)
            asumc  = 0.0
            for n in self._sequence:
                s[n] = self._state[n] + h * (b[0]*derivc[n] + \
                                             b[1]*self._derivp[-1][n])
                asumc += abs(s[n])
            if abs(asumc-asump) <= tolf*asumc:
                iter  = True
                break
            else:
                asump = asumc

        if not iter:
            wtext1 = "iterations did not converge in abm2 for time = "
            wtext2 = str(t) + ". Try changing tolerance or maxniter"
            wtext  = wtext1 + wtext2
            warn(wtext)

        derivc      = self._model(tnext, s)
        self._state = deepcopy(s)
        del s

        if not self._restart:
            del self._derivp[0]
            self._derivp.append(derivc)
        else:
            self._derivp = []

        self._restart = False
   
        return t+deltat, self._state

    # end of abm2

# ------------------------------------------------------------------------------

    def abm3(self, t, tnext, tolf=SQRTMACHEPS, maxniter=4):
        """
        Adams-Bashforth-Moulton 3:rd order, a predictor-corrector method. The 
        algorithm is claimed to be more accurate for a reasonably large number 
        of iterations but is also claimed to be more stable for a smaller number 
        of iterations. Different orders of Adams-Bashforth-Moulton may have 
        different accuracy and stability properties - this is the reason for 
        three abmX methods being here. abm3 uses one computation of the 
        model/derivative function per iteration.
        
        'tolf' is the maximum allowed fractional difference between the sum of 
        the absolute values of the state vector variables for a prediction and 
        a correction. 'maxniter' is the maximum number of iterations made. 
        The computation is NOT stopped if convergence is not reached, but a 
        warning will be issued to stdout.

        This method uses abm2 to build up its history before it has created 
        its own history. Cf. abm2 for details.
        """

        assert is_nonneginteger(maxniter), \
             "max number of iterations must be a non-negative integer in abm3!"
        assert tnext > t, "time step must be positive in abm3!"
        if tolf < MACHEPS:
            tolf = MACHEPS
            wtext1 = "tolerance smaller than machine epsilon is not"
            wtext2 = " recommended in abm3. Machine epsilon is used instead"
            warn(wtext1+wtext2)


        deltat =  tnext - t
        h  =  Dynamics.__ONE12TH * deltat
        a  =  array('d', (23.0, -16.0,  5.0))
        b  =  array('d', ( 5.0,   8.0, -1.0))

        if len(self._derivp) < 3:
            deriv0 = self._model(t, self._state)
            tnew, self._state = self.abm2(t, tnext, tolf, maxniter)
            self._derivp.insert(0, deriv0)
            return tnew, self._state

        # Initiate auxiliary dicts/arrays
        if self._mode == 'dict':
            s = {}; derivc = {}
        else:
            NN = float('nan')
            s =array('d', self._neqs*[NN]); derivc =array('d', self._neqs*[NN])

        asump = 0.0
        iter  = False
        for n in self._sequence:
            s[n] = self._state[n] + h * (a[0]*self._derivp[-1][n] + \
                                         a[1]*self._derivp[-2][n] + \
                                         a[2]*self._derivp[-3][n])
            asump += abs(s[n])
        for k in range(0, maxniter):
            derivc = self._model(tnext, s)
            asumc  = 0.0
            for n in self._sequence:
                s[n] = self._state[n] + h * (b[0]*derivc[n] + \
                                             b[1]*self._derivp[-1][n] + \
                                             b[2]*self._derivp[-2][n])
                asumc += abs(s[n])
            if abs(asumc-asump) <= tolf*asumc:
                iter  = True
                break
            else:
                asump = asumc

        if not iter:
            wtext1 = "iterations did not converge in abm3 for time = "
            wtext2 = str(t) + ". Try changing tolerance or maxniter"
            wtext  = wtext1 + wtext2
            warn(wtext)

        derivc      = self._model(tnext, s)
        self._state = deepcopy(s)
        del s

        if not self._restart:
            del self._derivp[0]
            self._derivp.append(derivc)
        else:
            self._derivp = []

        self._restart = False
   
        return t+deltat, self._state

    # end of abm3

# ------------------------------------------------------------------------------

    def abm4(self, t, tnext, tolf=SQRTMACHEPS, maxniter=4):
        """
        Adams-Bashforth-Moulton 4:th order, a predictor-corrector method. The 
        algorithm is claimed to be more accurate for a reasonably large number 
        of iterations but is also claimed to be more stable for a smaller number 
        of iterations. Different orders of Adams-Bashforth-Moulton may have 
        different accuracy and stability properties - this is the reason for 
        three abmX methods being here. abm4 uses one computation of the 
        model/derivative function per iteration.
        
        'tolf' is the maximum allowed fractional difference between the sum of 
        the absolute values of the state vector variables for a prediction and 
        a correction. 'maxniter' is the maximum number of iterations made. 
        The computation is NOT stopped if convergence is not reached, but a 
        warning will be issued to stdout.

        This method uses abm3 to build up its history before it has created 
        its own history. Cf. abm3 for details.
        """

        assert is_nonneginteger(maxniter), \
             "max number of iterations must be a non-negative integer in abm4!"
        assert tnext > t, "time step must be positive in abm4!"
        if tolf < MACHEPS:
            tolf = MACHEPS
            wtext1 = "tolerance smaller than machine epsilon is not"
            wtext2 = " recommended in abm4. Machine epsilon is used instead"
            warn(wtext1+wtext2)


        deltat =  tnext - t
        h  =  Dynamics.__ONE24TH * deltat
        a  =  array('d', (55.0, -59.0, 37.0, -9.0))
        b  =  array('d', ( 9.0,  19.0, -5.0,  1.0))

        if len(self._derivp) < 4:
            deriv0 = self._model(t, self._state)
            tnew, self._state = self.abm3(t, tnext, tolf, maxniter)
            self._derivp.insert(0, deriv0)
            return tnew, self._state

        # Initiate auxiliary dicts/arrays
        if self._mode == 'dict':
            s = {}; derivc = {}
        else:
            NN = float('nan')
            s =array('d', self._neqs*[NN]); derivc =array('d', self._neqs*[NN])

        asump = 0.0
        iter  = False
        for n in self._sequence:
            s[n] = self._state[n] + h * (a[0]*self._derivp[-1][n] + \
                                         a[1]*self._derivp[-2][n] + \
                                         a[2]*self._derivp[-3][n] + \
                                         a[3]*self._derivp[-4][n])
            asump += abs(s[n])
        for k in range(0, maxniter):
            derivc = self._model(tnext, s)
            asumc  = 0.0
            for n in self._sequence:
                s[n] = self._state[n] + h * (b[0]*derivc[n] + \
                                             b[1]*self._derivp[-1][n] + \
                                             b[2]*self._derivp[-2][n] + \
                                             b[3]*self._derivp[-3][n])
                asumc += abs(s[n])
            if abs(asumc-asump) <= tolf*asumc:
                iter  = True
                break
            else:
                asump = asumc

        if not iter:
            wtext1 = "iterations did not converge in abm4 for time = "
            wtext2 = str(t) + ". Try changing tolerance or maxniter"
            wtext  = wtext1 + wtext2
            warn(wtext)

        derivc      = self._model(tnext, s)
        self._state = deepcopy(s)
        del s

        if not self._restart:
            del self._derivp[0]
            self._derivp.append(derivc)
        else:
            self._derivp = []

        self._restart = False
   
        return t+deltat, self._state

    # end of abm4

# ------------------------------------------------------------------------------
# The matrix exponential method!!!
# ------------------------------------------------------------------------------

    def matrixexp(self, t, tnext, tolf=TWOMACHEPS, maxterms=128):
        """
        Solver using the matrix exponential method. 
        
        This solver only works for systems of odes which are linear in the 
        state vector. The state vector must be entered as a matrix column 
        vector and coeffmatrix as a matrix containing the present (possibly 
        time dependent) coefficients, both objects belonging to the 
        misclib.Matrix class. The differential equation is:  dY/deltat = C + M*Y
        where Y is the state column vector, C is a matrix column vector of 
        constants (constant between t and t+deltat), and M is a matrix of 
        constant coefficients (constant between t and t+deltat).
        
        The solution is Y(t+deltat) = C*deltat + exp(M*Y(t)*deltat), where 
        the exponential is expanded in a McLaurin series based on t as the 
        zero point in time, using 'tolf' as the fractional tolerance (based 
        on the fractional difference between the present sum and the previous) 
        and 'maxterms' as the absolute maximum number of terms.
        
        AND: matrixexp handles stiff systems well and with good accuracy !!!! 
        """        
        
        assert tnext > t, "time step must be a positive float in matrixexp!"
        if maxterms < 3:
            maxterms = 3
            wtext  = "maximum number of terms in series must not be"
            wtext += " less than 3 in matrixexp (3 will be used)"
            warn(wtext)

        deltat               = tnext - t
        vconst, coeffmatrix  = self._model(t, self._state)
        vconstdeltat         = scaled(vconst, deltat)
        coeffdeltat          = scaled(coeffmatrix, deltat)
        # Vector and matrix arithmetics from the Matrix class used throughout!
        term         = coeffdeltat * self._state
        self._state  = self._state + term
        compare      = flattened(self._state)
        sumc         = sum(abs(c) for c in compare)
        converged    = False
        for k in range(2, maxterms):
            term    = coeffdeltat * term
            const   = 1.0 / float(k)
            term.scale(const)
            self._state  = self._state + term   # += does not work here!
            compare      = flattened(self._state)
            sumo         = sumc
            sumc         = sum(abs(c) for c in compare)
            if abs(sumc-sumo) <= sumc*tolf:
                converged = True
                break
        self._state = self._state + vconstdeltat
        if not converged:
            wtext1 = "sum has not converged in matrixexp for time = "
            wtext2 = str(t) + ". Try changing tolerance or maxterms"
            warn(wtext1+wtext2)

        del vconstdeltat, coeffdeltat, term

        self._restart = False
    
        return t+deltat, self._state

    # end of matrixexp

# ------------------------------------------------------------------------------
# Auxiliary function department
# ------------------------------------------------------------------------------

    def tiptoe(self, t, tstep):
        """
        Method used to get around known points of discontinuity or points at 
        which there is continuity but where the first derivative is not defined 
        in the equations defining a set of odes, such as step functions or 
        Heaviside functions for the coefficients. It avoids those points by 
        allowing the solution to progress in time to a point exactly on the 
        singularity and sets off the continued progress from a point just after 
        the discontinuity, using he solution at the point of the discontinuity 
        as a new starting point just after the discontinuity. "Just after" is 
        expressed in terms of machine epsilon. The method is used like this:
        
            t, tnext = solution.tiptoe(t, tstep)
            t, state = solution.method(t, tnext)
        
        For tiptoe to be used, breakpoints must be given to the instance object 
        punched out from Dynamics. breakpoints may be a single float, a list or 
        a tuple. The points will be deleted from the list when they are passed.
        """        
        
        assert tstep > 0.0, "time step must be positive in tiptoe!"

        tnext = t + tstep

        for point in self._breakpoints:
            if t < point and tnext >= point:
                tnext         = point
                self._restart = True
                break
            elif t == point:
                if point == 0.0: t += TINY
                else:            t *= Dynamics.__ONEP2MACHEPS
                del self._breakpoints[0]
                break

        return t, tnext

    # end of tiptoe

# ------------------------------------------------------------------------------

# end of Dynamics

# ------------------------------------------------------------------------------

# stiffdyn.py
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

from dynamics         import Dynamics
from misclib.matrix   import Matrix
from numlib.matrixops import ludcmp_crout_piv, lusubs, lusubs_imp
from numlib.miscnum   import krond
from misclib.numbers  import is_posinteger, is_nonneginteger
from machdep.machnum  import MACHEPS, TWOMACHEPS, SQRTSQRTME
from misclib.errwarn  import warn

# ------------------------------------------------------------------------------

class StiffDynamics(Dynamics):
    """
    StiffDynamics inherits from the Dynamics class and contains a number of  
    implicit solvers particularly suited for stiff systems of ODEs. All the 
    explicit solvers including the matrix exponential are inherited from the 
    Dynamics class, as well as the tiptoe method/function.

    The reason for placing the implicit solvers in a separate class has nothing 
    to do with program logic: the reason is that a file containing all solvers 
    would be voluminous!

    NB. The methods added by this class only support state vectors which 
    are either SimElement stacks, lists or 'd' arrays - dicts and Matrix 
    class column vectors are NOT supported!!!!!!!!!!!  Dicts may be used 
    in the program using StiffDynamics if iters2dict together with 
    dict2iternlist or dict2iterndarray conversion is also used, however, 
    cf. the misclib/iterables.py module!!!!

    If the state vector is the result put out by the program using the implicit 
    methods added in this class, the corresponding time to put out is "tnext" - 
    if derivatives are the result put out, then "tnext" is correct as well, 
    since the last model/derivative values computed are the ones for tnext.

    StiffDynamics offers three different types of implicit solvers: 
    
    * the Adams-Moulton type uses the model/derivative history so that 
    yn+1 = yn + const*f(tn+1, yn+1) + a linear combination of the most
    recent and the previous model/derivative values;
    
    2nd order Adams-Moulton is in fact the implicit trapezoidal;
    * the implicit midpoint method uses the midpoint between the new 
    and the most recent state values and time values so that 
    yn+1 = yn + stepsize*(f((tn + tn+1)/2, (yn + yn+1)/2));
    
    * the BDF type due to William Gear uses the state history so that 
    yn+1 = yn + const*f(tn+1, yn+1) + a linear combination of the most 
    recent and the previous state values (1st order is actually the 
    backward Euler). Up to 6th order is stable theoretically, but 'lore' 
    has it that 6th order is a bit shaky in practice, so this class 
    offers Gear's method up to 5th order only (even 5th order may appear 
    less reliable at times...).

    A matrix version of Newton's method is used for solving the implicit 
    equations. The procedure is controlled by the parameters (all equations
    are expressed as F = 0):

    tolf        desired fractional accuracy of the sum of the absolute 
                values of the states (a combination of absolute and 
                fractional will actually be used: tolf*abs(sum) + tola)

    tola        desired max absolute difference of the absolute value of 
                the sum of the F:s from zero 
                AND
                desired absolute accuracy of the sum of the absolute 
                values of the states (a combination of absolute and 
                fractional will actually be used: tolf*abs(sum) + tola)

    maxitn      maximum number of iterations

    if imprv is set True then polishing of the results will be carried out 
    (see numlib/matrixops.py for details on matrix equations).

    hf and ha are the increments used for numerically computing the 
    derivatives used by the Newton algorithm - hf is the fractional 
    increment and ha is the absolute increment used.

    It must be kept in mind that implicit methods (at least those in this 
    class) are notoriously slow and should be used only when needed. Some of 
    the methods in the Dynamics base class are capable of handling problems 
    that are relatively stiff.
    """

    # Class variables used in methods
    __ONE12TH  =  0.08333333333333333333333333   # 1.0 / 12.0
    __ONE24TH  =  0.04166666666666666666666667   # 1.0 / 24.0

# ------------------------------------------------------------------------------

    def __init__(self, model, state, breakpoints=None):
        """
        Basic initialization for the class. Cf. the Dynamics class for an 
        explanation of the input parameters.
        """

        # First take care of what is inherited:
        Dynamics.__init__(self, model, state, breakpoints)

        # The explicit solvers presently only work for stacks, lists 
        # and 'd' arrays:
        assert self._mode == 'list', \
        "StiffDynamics only handles stacks, lists and 'd' arrays at the moment!"

        # History array for the gearX solvers:
        self.__prev  = []      # For the gearX solvers

    # end of __init__

# ------------------------------------------------------------------------------

    def implicit_trapez(self, t, tnext, \
                              tolf=SQRTSQRTME, tola=TWOMACHEPS, maxitn=16, \
                              imprv=False, hf=0.5**20, ha=0.5**40):
        """
        The implicit trapezoidal method:
            yn+1 - yn  -  0.5*h*(f(tn+1, yn+1) + f(tn, yn))  =  0

        Solved for yn+1 using Newton's algorithm, i. e. the yn+1 satisfying 
        F(yn+1) = 0, which requires that all derivatives F'(yn+1) with respect 
        to all yn+1 be formed.

        The implicit trapezoidal is in fact the 2nd order Adams-Moulton method!
        Cf. Dahlquist-Bjorck-Anderson for the theory.

        implicit_trapez is less stable than the gearX methods but possibly 
        more accurate. 
        """

        self._checkinput('implicit_trapez', t, tnext, \
                          tolf, tola, maxitn, imprv)

        h   = tnext - t
        s   = deepcopy(self._state)
        ft  = self._model(t, s)
        ha5 = 0.5 * h
    
        # -----------------------------------------
        def _fidfi(y):
            # fi first
            f    = self._model(tnext, y)
            fi   = deepcopy(f)
            beta = deepcopy(fi)
            for n in self._sequence:
                fi[n]   =  y[n] - s[n] - ha5*(ft[n]+f[n])
                beta[n] = - fi[n]
            # then fid = dfi[n]/dy[m] = dy[n]/dy[m] - 0.5*h*df[n]/dy[m], or
            # - 0.5*h*df[n]/dy[m]; n != m, and
            # 1.0 - 0.5*h*df[n]/dy[m]; n == m
            jacob = self._jake3(tnext, y, hf, ha)
            fid   = Matrix()
            for n in self._sequence:
                derivs = array('d', [])
                for m in self._sequence:
                    deriv = 1.0*krond(n, m) - ha5*jacob[n][m]
                    derivs.append(deriv)
                fid.append(derivs)
            alpha = Matrix(fid)
            return alpha, beta
        # ---------------------------------------------

        y = deepcopy(s)
        converged = False
        for k in range(0, maxitn):
            alpha, beta = _fidfi(y)
            errf = 0.0
            for b in beta:
                errf += abs(b)
            if errf < tola:
                converged = True
                break
            delta = self._delta(alpha, beta, imprv)
            erry   = 0.0
            sumay  = 0.0
            for n in self._sequence:
                y[n]  += delta[n]
                sumay += abs(y[n])
                erry  += abs(delta[n])
            if erry <= tolf*sumay + tola:
                converged = True
                break
        if not converged: self._iterwarn('implicit_trapez', t)

        self._state  = deepcopy(y)

        self._restart = False
    
        return t+h, self._state

    # end of implicit_trapez

# ------------------------------------------------------------------------------

    def adams_moulton3(self, t, tnext, \
                                tolf=SQRTSQRTME, tola=TWOMACHEPS, maxitn=16, \
                                imprv=False, hf=0.5**20, ha=0.5**40):
        """
        Adams-Moulton 3rd order:
        yn+1 - yn  -  h*(5*f(tn+1, yn+1) + 8*f(tn, yn) - f(tn-1, yn-1))/12  =  0

        Solved for yn+1 using Newton's algorithm, i. e. the yn+1 satisfying 
        F(yn+1) = 0, which requires that all derivatives F'(yn+1) with respect 
        to all yn+1 be formed.

        Cf. Dahlquist-Bjorck-Anderson for the theory.

        In order to handle the lack of model/derivative history at the 
        beginning of a series of time points a procedure reusing the 
        first few points is used.

        adams_moulton3 is less stable than the gearX methods but possibly 
        more accurate. 
        """

        self._checkinput('adams_moulton3', t, tnext, \
                          tolf, tola, maxitn, imprv)

        if len(self._derivp) == 0:
            deriv0       = self._model(t, self._state)
            self._derivp = 2*[deriv0]

        # Initiate auxiliary array:
        deriv = array('d', self._neqs*[float('nan')])

        h   = tnext - t
        s   = deepcopy(self._state)
        ft  = self._model(t, s)
        fp  = self._derivp[0]
        ht  = StiffDynamics.__ONE12TH * h
        ht5 = 5.0 * ht
    
        # -----------------------------------------
        def _fidfi(y):
            # fi first
            f    = self._model(tnext, y)
            fi   = deepcopy(f)
            beta = deepcopy(fi)
            for n in self._sequence:
                fi[n]   = y[n] - s[n] - ht*(5.0*f[n]+8.0*ft[n]-fp[n])
                beta[n] = - fi[n]
            # then fid = dfi[n]/dy[m] = dy[n]/dy[m] - 5.0*ht*df[n]/dy[m], or
            # - 5.0*ht*df[n]/dy[m]; n != m, and
            # 1.0 - 5.0*ht*df[n]/dy[m]; n == m
            jacob = self._jake3(tnext, y, hf, ha)
            fid   = Matrix()
            for n in self._sequence:
                derivs = array('d', [])
                for m in self._sequence:
                    deriv = 1.0*krond(n, m) - ht5*jacob[n][m]
                    derivs.append(deriv)
                fid.append(derivs)
            alpha = Matrix(fid)
            return alpha, beta
        # ---------------------------------------------

        y = deepcopy(s)
        converged = False
        for k in range(0, maxitn):
            alpha, beta = _fidfi(y)
            errf = 0.0
            for b in beta:
                errf += abs(b)
            if errf < tola:
                converged = True
                break
            delta = self._delta(alpha, beta, imprv)
            erry   = 0.0
            sumay  = 0.0
            for n in self._sequence:
                y[n]  += delta[n]
                sumay += abs(y[n])
                erry  += abs(delta[n])
            if erry <= tolf*sumay + tola:
                converged = True
                break
        if not converged: self._iterwarn('adams_moulton3', t)

        self._state  = deepcopy(y)

        tnew  = t + h
        deriv = self._model(tnew, self._state)

        if not self._restart:
            del self._derivp[0]
            self._derivp.append(deriv)
        else:
            self._derivp = array('d', [])

        self._restart = False
   
        return tnew, self._state

    # end of adams_moulton3

# ------------------------------------------------------------------------------

    def adams_moulton4(self, t, tnext, \
                                tolf=SQRTSQRTME, tola=TWOMACHEPS, maxitn=16, \
                                imprv=False, hf=0.5**20, ha=0.5**40):
        """
        Adams-Moulton 4th order:
        yn+1 - yn  -  h*(9*f(tn+1, yn+1) + 19*f(tn, yn) - 
                       - 5*f(tn-1, yn-1) + f(tn-2, yn-2))/24   =   0

        Solved for yn+1 using Newton's algorithm, i. e. the yn+1 satisfying 
        F(yn+1) = 0, which requires that all derivatives F'(yn+1) with respect 
        to all yn+1 be formed.
        
        Cf. Dahlquist-Bjorck-Anderson for the theory.

        In order to handle the lack of model/derivative history at the 
        beginning of a series of time points a procedure reusing the 
        first few points is used.

        adams_moulton4 is less stable than the gearX methods but possibly 
        more accurate. 
        """

        self._checkinput('adams_moulton4', t, tnext, \
                          tolf, tola, maxitn, imprv)

        if len(self._derivp) == 0:
            deriv0       = self._model(t, self._state)
            self._derivp = 3*[deriv0]

        # Initiate auxiliary array:
        deriv = array('d', self._neqs*[float('nan')])

        h   = tnext - t
        s   = deepcopy(self._state)
        ft  = self._model(t, s)
        fp1 = self._derivp[-1]
        fp2 = self._derivp[-2]
        ht  = StiffDynamics.__ONE24TH * h
        ht9 = 9.0 * ht
    
        # -----------------------------------------
        def _fidfi(y):
            # fi first
            f    = self._model(tnext, y)
            fi   = deepcopy(f)
            beta = deepcopy(fi)
            for n in self._sequence:
                fi[n] = y[n] - s[n] - ht*(9.0*f[n]+19.0*ft[n]-5.0*fp1[n]+fp2[n])
                beta[n] = - fi[n]
            # then fid = dfi[n]/dy[m] = dy[n]/dy[m] - 9.0*ht*df[n]/dy[m], or
            # - 9.0*ht*df[n]/dy[m]; n != m, and
            # 1.0 - 9.0*ht*df[n]/dy[m]; n == m
            jacob = self._jake3(tnext, y, hf, ha)
            fid   = Matrix()
            for n in self._sequence:
                derivs = array('d', [])
                for m in self._sequence:
                    deriv = 1.0*krond(n, m) - ht9*jacob[n][m]
                    derivs.append(deriv)
                fid.append(derivs)
            alpha = Matrix(fid)
            return alpha, beta
        # ---------------------------------------------

        y = deepcopy(s)
        converged = False
        for k in range(0, maxitn):
            alpha, beta = _fidfi(y)
            errf = 0.0
            for b in beta:
                errf += abs(b)
            if errf < tola:
                converged = True
                break
            delta = self._delta(alpha, beta, imprv)
            erry   = 0.0
            sumay  = 0.0
            for n in self._sequence:
                y[n]  += delta[n]
                sumay += abs(y[n])
                erry  += abs(delta[n])
            if erry <= tolf*sumay + tola:
                converged = True
                break
        if not converged: self._iterwarn('adams_moulton4', t)

        self._state  = deepcopy(y)

        tnew  = t + h
        deriv = self._model(tnew, self._state)

        if not self._restart:
            del self._derivp[0]
            self._derivp.append(deriv)
        else:
            self._derivp = array('d', [])

        self._restart = False
   
        return tnew, self._state

    # end of adams_moulton4

# ------------------------------------------------------------------------------

    def implicit_midpoint(self, t, tnext, \
                                tolf=SQRTSQRTME, tola=TWOMACHEPS, maxitn=16, \
                                imprv=False, hf=0.5**20, ha=0.5**40):
        """
        The implicit midpoint method:
            yn+1 - yn  -  h*(f((tn + tn+1)/2, (yn + yn+1)/2))  =  0

        Solved for yn+1 using Newton's algorithm, i. e. the yn+1 satisfying 
        F(yn+1) = 0, which requires that all derivatives F'(yn+1) with respect 
        to all yn+1 be formed.

        implicit_midpoint is less stable than the gearX methods but possibly 
        more accurate. 
        """

        self._checkinput('implicit_midpoint', t, tnext, \
                          tolf, tola, maxitn, imprv)

        h   = tnext - t
        s   = deepcopy(self._state)
        tav = 0.5 * (t+tnext)
        ha5 = 0.5 * h
        yav = deepcopy(self._state)
    
        # -----------------------------------------
        def _fidfi(y):
            # fi first
            for n in self._sequence:
                yav[n]  = 0.5 * (s[n]+y[n])
            f    = self._model(tav, yav)
            fi   = deepcopy(f)
            beta = deepcopy(fi)
            for n in self._sequence:
                fi[n]   = y[n] - s[n] - h*f[n]
                beta[n] = - fi[n]
            # then fid = dfi[n]/dy[m] = dy[n]/dy[m] - 0.5*h*df[n]/dy[m], or
            # - 0.5*h*df[n]/dy[m]; n != m, and
            # 1.0 - 0.5*h*f[n]*df[n]/dy[m]; n == m
            jacob = self._jake3(tav, yav)
            fid   = Matrix()
            for n in self._sequence:
                derivs = array('d', [])
                for m in self._sequence:
                    deriv = 1.0*krond(n, m) - ha5*jacob[n][m]
                    derivs.append(deriv)
                fid.append(derivs)
            alpha = Matrix(fid)
            return alpha, beta
        # ---------------------------------------------

        y = deepcopy(s)
        converged = False
        for k in range(0, maxitn):
            alpha, beta = _fidfi(y)
            errf = 0.0
            for b in beta:
                errf += abs(b)
            if errf < tola:
                converged = True
                break
            delta = self._delta(alpha, beta, imprv)
            erry   = 0.0
            sumay  = 0.0
            for n in self._sequence:
                y[n]  += delta[n]
                sumay += abs(y[n])
                erry  += abs(delta[n])
            if erry <= tolf*sumay + tola:
                converged = True
                break
        if not converged: self._iterwarn('implicit_midpoint', t)

        self._state  = deepcopy(y)

        self._restart = False
    
        return t+h, self._state

    # end of implicit_midpoint

# ------------------------------------------------------------------------------

    def gear1(self, t, tnext, \
                    tolf=SQRTSQRTME, tola=TWOMACHEPS, maxitn=16, \
                    imprv=False, hf=0.5**20, ha=0.5**40):
        """
        Gear 1st order:  yn+1 - yn  -  h*f(tn+1, yn+1)  =  0
        
        Solved for yn+1 using Newton's algorithm, i. e. the yn+1 satisfying 
        F(yn+1) = 0, which requires that all derivatives F'(yn+1) with respect 
        to all yn+1 be formed.

        gear1 is more stable than the adams_moultonX methods but possibly 
        less accurate. 
        """

        self._checkinput('gear1', t, tnext, \
                          tolf, tola, maxitn, imprv)

        h  = tnext - t
        s  = deepcopy(self._state)

        # -----------------------------------------
        def _fidfi(y):
            # fi first
            f    = self._model(tnext, y)
            fi   = deepcopy(f)
            beta = deepcopy(fi)
            for n in self._sequence:
                fi[n]   = y[n] - s[n] - h*f[n]
                beta[n] = - fi[n]
            # then fid = dfi[n]/dy[m] = dy[n]/dy[m] - h*df[n]/dy[m], or
            # - h*df[n]/dy[m]; n != m, and
            # 1.0 - h*df[n]/dy[m]; n == m
            jacob = self._jake3(tnext, y, hf, ha)
            fid   = Matrix()
            for n in self._sequence:
                derivs = array('d', [])
                for m in self._sequence:
                    deriv = 1.0*krond(n, m) - h*jacob[n][m]
                    derivs.append(deriv)
                fid.append(derivs)
            alpha = Matrix(fid)
            return alpha, beta
        # ---------------------------------------------

        y = deepcopy(s)
        converged = False
        for k in range(0, maxitn):
            alpha, beta = _fidfi(y)
            errf = 0.0
            for b in beta:
                errf += abs(b)
            if errf < tola:
                converged = True
                break
            delta = self._delta(alpha, beta, imprv)
            erry   = 0.0
            sumay  = 0.0
            for n in self._sequence:
                y[n]  += delta[n]
                sumay += abs(y[n])
                erry  += abs(delta[n])
            if erry <= tolf*sumay + tola:
                converged = True
                break
        if not converged: self._iterwarn('gear1', t)

        self._state  = deepcopy(y)

        self._restart = False
    
        return t+h, self._state

    # end of gear1

# ------------------------------------------------------------------------------

    def gear2(self, t, tnext, \
                    tolf=SQRTSQRTME, tola=TWOMACHEPS, maxitn=16, \
                    imprv=False, hf=0.5**20, ha=0.5**40):
        """
        Gear 2nd order:  3*yn+1 - 4*yn + yn-1  -  2*h*f(tn+1, yn+1)  =  0
        
        Solved for yn+1 using Newton's algorithm, i. e. the yn+1 satisfying 
        F(yn+1) = 0, which requires that all derivatives F'(yn+1) with respect 
        to all yn+1 be formed.

        In order to handle the lack of state history at the beginning of a 
        series of time points gear1 is called for the first point.

        gear2 is more stable than the adams_moultonX methods but possibly 
        less accurate. 
        """

        self._checkinput('gear2', t, tnext, \
                          tolf, tola, maxitn, imprv)

        h  = tnext - t
        s  = deepcopy(self._state)

        lenprev = len(self.__prev)
        if lenprev < 1:  # gear1 must be used as long as previously computed 
                         # self._state backward are not sufficient
            tdum, self._state = self.gear1(t, tnext, tolf, tola, maxitn, \
                                                           imprv, hf, ha)
            # (we don't want to return the wrong time point...)
            self.__prev.insert(0, s)

        else:
            # -------------------------------------------------------
            def _fidfi(y):
                # fi first
                f    = self._model(tnext, y)
                fi   = deepcopy(f)
                beta = deepcopy(fi)
                for n in self._sequence:
                    fi[n]   = 3.0*y[n] - 4.0*s[n] + self.__prev[0][n] - \
                                                                    2.0*h*f[n]
                    beta[n] = - fi[n]
                # then fid = dfi[n]/dy[m] = 3*dy[n]/dy[m] - 2*h*df[n]/dy[m], or
                # - 2*h*df[n]/dy[m]; n != m, and
                # 3 - 2*h*df[n]/dy[m]; n == m
                jacob = self._jake3(tnext, y, hf, ha)
                fid   = Matrix()
                for n in self._sequence:
                    derivs = array('d', [])
                    for m in self._sequence:
                        deriv = 3.0*krond(n, m) - 2.0*h*jacob[n][m]
                        derivs.append(deriv)
                    fid.append(derivs)
                alpha = Matrix(fid)
                return alpha, beta
            # -------------------------------------------------------

            y = deepcopy(s)
            converged = False
            for k in range(0, maxitn):
                alpha, beta = _fidfi(y)
                errf = 0.0
                for b in beta:
                    errf += abs(b)
                if errf < tola:
                    converged = True
                    break
                delta = self._delta(alpha, beta, imprv)
                erry  = 0.0
                sumay = 0.0
                for n in self._sequence:
                    y[n]  += delta[n]
                    sumay += abs(y[n])
                    erry  += abs(delta[n])
                if erry <= tolf*sumay + tola:
                    converged = True
                    break
            if not converged: self._iterwarn('gear2', t)

            self._state = deepcopy(y)

            if not self._restart: self.__prev[0] = deepcopy(s)
            else:                 self.__prev    = []

        self._restart  = False

        return t+h, self._state

    # end of gear2

# ------------------------------------------------------------------------------

    def gear3(self, t, tnext, \
                    tolf=SQRTSQRTME, tola=TWOMACHEPS, maxitn=16, \
                    imprv=False, hf=0.5**20, ha=0.5**40):
        """
        Gear 3rd order:
              11*yn+1 - 18*yn + 9*yn-1 - 2*yn-1  -  6*h*f(tn+1, yn+1)  =  0

        Solved for yn+1 using Newton's algorithm, i. e. the yn+1 satisfying 
        F(yn+1) = 0, which requires that all derivatives F'(yn+1) with respect 
        to all yn+1 be formed.

        In order to handle the lack of state history at the beginning of a 
        series of time points gear2 is called for the two first points.

        gear3 is more stable than the adams_moultonX methods but possibly 
        less accurate. 
        """

        self._checkinput('gear3', t, tnext, \
                          tolf, tola, maxitn, imprv)

        h  = tnext - t
        s  = deepcopy(self._state)
        yP = deepcopy(self.__prev)

        lenprev = len(self.__prev)
        if lenprev < 2:  # gear2 must be used as long as previously computed 
                         # self._state backward are not sufficient
            tdum, self._state = self.gear2(t, tnext, tolf, tola, maxitn, \
                                                           imprv, hf, ha)
            # (we don't want to return the wrong time point...)
            if lenprev == 1: self.__prev.insert(0, yP[0])

        else:
            # -------------------------------------------------------
            def _fidfi(y):
                # fi first
                f    = self._model(tnext, y)
                fi   = deepcopy(f)
                beta = deepcopy(fi)
                for n in self._sequence:
                    fi[n]   = 11.0*y[n] - 18.0*s[n] + 9.0*self.__prev[1][n] - \
                                           2.0*self.__prev[0][n] - 6.0*h*f[n]
                    beta[n] = - fi[n]
                # then fid = dfi[n]/dy[m] = 11*dy[n]/dy[m] - 6*h*df[n]/dy[m], or
                # - 6*h*df[n]/dy[m]; n != m, and
                # 11 - 6*h*df[n]/dy[m]; n == m
                jacob = self._jake3(tnext, y, hf, ha)
                fid   = Matrix()
                for n in self._sequence:
                    derivs = array('d', [])
                    for m in self._sequence:
                        deriv = 11.0*krond(n, m) - 6.0*h*jacob[n][m]
                        derivs.append(deriv)
                    fid.append(derivs)
                alpha = Matrix(fid)
                return alpha, beta
            # -------------------------------------------------------

            y = deepcopy(s)
            converged = False
            for k in range(0, maxitn):
                alpha, beta = _fidfi(y)
                errf = 0.0
                for b in beta:
                    errf += abs(b)
                if errf < tola:
                    converged = True
                    break
                delta = self._delta(alpha, beta, imprv)
                erry  = 0.0
                sumay = 0.0
                for n in self._sequence:
                    y[n]  += delta[n]
                    sumay += abs(y[n])
                    erry  += abs(delta[n])
                if erry <= tolf*sumay + tola:
                    converged = True
                    break
            if not converged: self._iterwarn('gear3', t)

            self._state = deepcopy(y)

            if not self._restart:
                self.__prev[0] = deepcopy(self.__prev[1])
                self.__prev[1] = deepcopy(s)
            else:
                self.__prev    = []

        self._restart  = False

        return t+h, self._state

    # end of gear3

# ------------------------------------------------------------------------------

    def gear4(self, t, tnext, \
                    tolf=SQRTSQRTME, tola=TWOMACHEPS, maxitn=16, \
                    imprv=False, hf=0.5**20, ha=0.5**40):
        """
        Gear 4th order:
                  25*yn+1 - 48*yn + 36*yn-1 - 16*yn-2 + 3*yn-3  -
                                                 -  12*h*f(tn+1, yn+1)  =  0
        
        Solved for yn+1 using Newton's algorithm, i. e. the yn+1 satisfying 
        F(yn+1) = 0, which requires that all derivatives F'(yn+1) with respect 
        to all yn+1 be formed.

        In order to handle the lack of state history at the beginning of a 
        series of time points gear3 is called for the first three points.

        gear4 is more stable than the adams_moultonX methods but possibly 
        less accurate. 
        """

        self._checkinput('gear4', t, tnext, \
                          tolf, tola, maxitn, imprv)

        h  = tnext - t
        s  = deepcopy(self._state)
        yP = deepcopy(self.__prev)

        lenprev = len(self.__prev)

        if lenprev < 3:  # gear3 must be used as long as previously computed 
                         # self._state backward are not sufficient
            tdum, self._state = self.gear3(t, tnext, tolf, tola, maxitn, \
                                                           imprv, hf, ha)
            # (we don't want to return the wrong time point...)
            if lenprev == 2: self.__prev.insert(0, yP[0])

        else:
            # -------------------------------------------------------
            def _fidfi(y):
                # fi first
                f    = self._model(tnext, y)
                fi   = deepcopy(f)
                beta = deepcopy(fi)
                for n in self._sequence:
                    fi[n]   = 25.0*y[n] - 48.0*s[n] + 36.0*self.__prev[2][n] - \
                              16.0*self.__prev[1][n] + \
                               3.0*self.__prev[0][n] - 12.0*h*f[n]
                    beta[n] = - fi[n]
                # then fid = dfi[n]/dy[m] = 3*dy[n]/dy[m] - 2*h*df[n]/dy[m], or
                # - 12*h*df[n]/dy[m]; n != m, and
                # 25 - 12*h*df[n]/dy[m]; n == m
                jacob = self._jake3(tnext, y, hf, ha)
                fid   = Matrix()
                for n in self._sequence:
                    derivs = array('d', [])
                    for m in self._sequence:
                        deriv = 25.0*krond(n, m) - 12.0*h*jacob[n][m]
                        derivs.append(deriv)
                    fid.append(derivs)
                alpha = Matrix(fid)
                return alpha, beta
            # -------------------------------------------------------

            y = deepcopy(s)
            converged = False
            for k in range(0, maxitn):
                alpha, beta = _fidfi(y)
                errf = 0.0
                for b in beta:
                    errf += abs(b)
                if errf < tola:
                    converged = True
                    break
                delta = self._delta(alpha, beta, imprv)
                erry  = 0.0
                sumay = 0.0
                for n in self._sequence:
                    y[n]  += delta[n]
                    sumay += abs(y[n])
                    erry  += abs(delta[n])
                if erry <= tolf*sumay + tola:
                    converged = True
                    break
            if not converged: self._iterwarn('gear4', t)

            self._state = deepcopy(y)

            if not self._restart:
                self.__prev[0] = deepcopy(self.__prev[1])
                self.__prev[1] = deepcopy(self.__prev[2])
                self.__prev[2] = deepcopy(s)
            else:
                self.__prev    = []

        self._restart  = False
   
        return t+h, self._state

    # end of gear4

# ------------------------------------------------------------------------------

    def gear5(self, t, tnext, \
                    tolf=SQRTSQRTME, tola=TWOMACHEPS, maxitn=16, \
                    imprv=False, hf=0.5**20, ha=0.5**40):
        """
        Gear 5th order:
          137*yn+1 - 300*yn + 300*yn-1 - 200*yn-2 + 75*yn-3 - 12*yn-4  - 
                                                 -  60*h*f(tn+1, yn+1)  =  0

        Solved for yn+1 using Newton's algorithm, i. e. the yn+1 satisfying 
        F(yn+1) = 0, which requires that all derivatives F'(yn+1) with respect 
        to all yn+1 be formed.

        In order to handle the lack of state history at the beginning of a 
        series of time points gear4 is called for the first four points.
        """

        self._checkinput('gear5', t, tnext, \
                          tolf, tola, maxitn, imprv)

        h  = tnext - t
        s  = deepcopy(self._state)
        yP = deepcopy(self.__prev)

        lenprev = len(self.__prev)
        if lenprev < 4:  # gear4 must be used as long as previously computed 
                         # self._state backward are not sufficient
            tdum, self._state = self.gear4(t, tnext, tolf, tola, maxitn, \
                                                           imprv, hf, ha)
            # (we don't want to return the wrong time point...)
            if lenprev == 3: self.__prev.insert(0, yP[0])

        else:
            # -------------------------------------------------------
            def _fidfi(y):
                # fi first
                f    = self._model(tnext, y)
                fi   = deepcopy(f)
                beta = deepcopy(fi)
                for n in self._sequence:
                    fi[n]   = 137.0*y[n] - 300.0*s[n] + \
                              300.0*self.__prev[3][n] - \
                              200.0*self.__prev[2][n] + \
                               75.0*self.__prev[1][n] - \
                               12.0*self.__prev[0][n] -  60.0*h*f[n]
                    beta[n] = - fi[n]
                # then fid = dfi[n]/dy[m] = 137*dy[n]/dy[m] - 60*h*df[n]/dy[m], 
                # or  - 60*h*df[n]/dy[m]; n != m, and
                # 137 - 60*h*df[n]/dy[m]; n == m
                jacob = self._jake3(tnext, y, hf, ha)
                fid   = Matrix()
                for n in self._sequence:
                    derivs = array('d', [])
                    for m in self._sequence:
                        deriv = 137.0*krond(n, m) - 60.0*h*jacob[n][m]
                        derivs.append(deriv)
                    fid.append(derivs)
                alpha = Matrix(fid)
                return alpha, beta
            # -------------------------------------------------------

            y = deepcopy(s)
            converged = False
            for k in range(0, maxitn):
                alpha, beta = _fidfi(y)
                errf = 0.0
                for b in beta:
                    errf += abs(b)
                if errf < tola:
                    converged = True
                    break
                delta = self._delta(alpha, beta, imprv)
                erry  = 0.0
                sumay = 0.0
                for n in self._sequence:
                    y[n]  += delta[n]
                    sumay += abs(y[n])
                    erry  += abs(delta[n])
                if erry <= tolf*sumay + tola:
                    converged = True
                    break
            if not converged: self._iterwarn('gear5', t)

            self._state = deepcopy(y)

            if not self._restart:
                self.__prev[0] = deepcopy(self.__prev[1])
                self.__prev[1] = deepcopy(self.__prev[2])
                self.__prev[2] = deepcopy(self.__prev[3])
                self.__prev[3] = deepcopy(s)
            else:
                self.__prev    = []

        self._restart  = False

        return t+h, self._state

    # end of gear5

# ------------------------------------------------------------------------------

    def _jake3(self, t, y, hf=0.5**20, ha=0.5**40):
        """
        Auxiliary function/method used by implicit methods to compute 
        the Jacobian. Cannot be used for dicts!!!!!
        """

        yplus  = array('d', [])
        yminus = array('d', [])
        for n in self._sequence:
            why = y[n]
            yplus.append( (1.0 + hf) * why + ha)
            yminus.append((1.0 - hf) * why - ha)

        jacob = Matrix()
        for n in self._sequence:
            derivs = array('d', [])
            for m in self._sequence:
                yn    = deepcopy(y)
                yp    = yplus[m]
                yn[m] = yp
                fp    = self._model(t, yn)[n]
                ym    = yminus[m]
                yn[m] = ym
                fm    = self._model(t, yn)[n]
                deriv = (fp-fm) / (yp-ym)
                derivs.append(deriv)
            jacob.append(derivs)

        return jacob

    # end of _jake3

# ------------------------------------------------------------------------------

    def _delta(self, alpha, beta, imprv):
        """
        Auxiliary function/method to compute the residual in the 
        iterative procedure of the implicit solvers.
        """

        lower, upper, permlist = ludcmp_crout_piv(alpha)[0:3]
        delta = lusubs(lower, upper, beta, permlist)
        if imprv:
            delta = lusubs_imp(alpha, lower, upper, beta, permlist, delta)

        return delta

    # end of _delta

# ------------------------------------------------------------------------------

    def _checkinput(self, cll, t, tnext, tolf, tola, maxitn, imprv):
        """
        Used to check the values of the input parameters to the solver methods. 
        cll is the name of the present method (a string).
        """

        assert tnext > t, "time step must be positive in " + cll + "!"

        wtext1 = "tolerances smaller than machine epsilon are not recommended "
        wtext2 = "in " + cll + ". Machine epsilon is used instead"
        wtext  = wtext1 + wtext2
        if tolf < MACHEPS:
            tolf = MACHEPS
            warn(wtext)
        if tola < MACHEPS:
            tola = MACHEPS
            warn(wtext)
        assert is_posinteger(maxitn), \
                     "maxitn must be a positive integer in " + cll + "!"

        if not is_nonneginteger(imprv):
            imprv = 0
            wtext1  = "imprv must be a non-negative integer in "
            wtext2  = cll + "! imprv=0 is used instead"
            wtext   = wtext1 + wtext2
            warn(wtext)

    # end of _checkinput

# ------------------------------------------------------------------------------

    def _iterwarn(self, caller, t):

        wtext1 = "Newton iteration did not converge in " + caller + " for time"
        wtext2 = " = " + str(t) + ". Try changing tolerances or maxitn"
        warn(wtext1+wtext2)

    # end of _iterwarn

# ------------------------------------------------------------------------------

# end of StiffDynamics

# ------------------------------------------------------------------------------
# abcrand.py
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

from abc    import ABCMeta, abstractmethod
from random import Random
from math   import sqrt, acos

from misclib.stack     import Stack
from statlib.invcdf    import ichistogram, ichistogram_int
from statlib.invcdf    import iemp_exp, iexpo_gen, ilaplace, icauchy
from statlib.invcdf    import iextreme_I, iextreme_gen, ilogistic
from statlib.invcdf    import irayleigh, ikodlin, ipareto_zero
from statlib.cdf       import cemp_exp, cexpo_gen, claplace, ccauchy
from statlib.cdf       import cextreme_I, cextreme_gen, clogistic
from statlib.cdf       import crayleigh, ckodlin, cpareto_zero
from numlib.miscnum    import safelog
from misclib.numbers   import is_posinteger, kept_within
from misclib.mathconst import PIINV

# ------------------------------------------------------------------------------

class ABCRand(metaclass=ABCMeta):
    """
    This class contains everything that is common to the GeneralRandomStream 
    and InverseRandomStream classes. Since this is also an abstract base 
    class, it cannot be used in a standalone fashion. Its methods and 
    attributes can only be reached through its subclasses GeneralRandomStream 
    and InverseRandomStream, which inherit from this class.

    ABCRand imports (and uses) some of the methods from Python's built-in 
    Random class including the "Mersenne Twister". This makes the Mersenne 
    Twister the basic rng of ABCRand and its heirs. All methods in ABCRand 
    that are not taken from Random are inverse-based, but the methods from 
    Random are generally not inverse-based. It may be noted that the Mersenne 
    Twister is a very reputable random number generator having a period of 
    2**19937-1.

    The following methods from Python's own Random class are inheritable from 
    ABCRand: randrange, randint, choice, shuffle, sample, vonmisesvariate, 
    paretovariate and weibullvariate.

    All the methods in ABCRand are inherited by GeneralRandomStream including 
    the ones imported from Random. The methods added by GeneralRandomStream do 
    NOT return the inverse of the [0.0, 1.0] random numbers from the basic rng.
    
    InverseRandomStream inherits the methods in ABCRand with the EXCEPTION 
    of the methods from Random (the Mersenne Twister is still there, though), 
    making all the methods in InverseRandomStream inverse-based, including 
    the methods added in the latter.
    
    The docstring documentation of Random, GeneralRandomStream and 
    InverseRandomStream must always be consulted before using the methods 
    inherited from ABCRand!

    NB  Some methods may return float('inf') or float('-inf') !
    """
# ------------------------------------------------------------------------------

    @abstractmethod
    def __init__(self, nseed=2147483647, heir=None):
        """
        Initiates the random stream using the input seed 'nseed' and Python's 
        __init__ constructor method. Unless...
        ...the input seed 'nseed' happens to be a list or tuple of numbers 
        in [0.0, 1.0], in which case this external feed will be used as the 
        basis of all random variate generation for the instance and will be 
        used in place of consecutively sampled numbers from Python's built-in 
        "random" method! 
        """

        if isinstance(nseed, int):
            assert is_posinteger(nseed), \
               "The seed (if not a feed) must be a positive integer in ABCRand!"
            rstream      = Random(nseed)
            self._feed   = False
            self.runif01 = rstream.random
            if heir != "InverseRandomStream":
                self.randrange       = rstream.randrange
                self.randint         = rstream.randint
                self.vonmisesvariate = rstream.vonmisesvariate
                # Random.paretovariate and Random.weibullvariate
                # are used by methods in GeneralRandomStream
                self._paretovariate  = rstream.paretovariate
                self._weibullvariate = rstream.weibullvariate

        else:  # nseed is a list or tuple
            # Check to see beforehand that no numbers 
            # from the feed is outside [0.0, 1.0]
            for x in nseed:
                assert 0.0 <= x <= 1.0, \
                    "number from feed is outside of [0.0, 1.0] in ABCRand!"
            self._feed   = Stack(nseed)  # Creates a Stack object
            self.runif01 = self.__rfeed01

    # end of __init__

# ------------------------------------------------------------------------------

    def __rfeed01(self):
        """
        Will be used as the "getter" of numbers in [0.0, 1.0] 
        when the input to the class is a feed rather than a 
        positive integer seed!
        """
                                    # Removes one number at a time 
        return self._feed.shift()   # from the stack

    # end of __rfeed01

# ------------------------------------------------------------------------------

    def runif_int0N(self, number):
        """
        Generator of uniformly distributed integers in [0, number) (also the 
        basis of some other procedures for generating random variates). 
        Numbers returned are 0 through number-1. NB!!!!!!!
        """

        assert is_posinteger(number)

        return int(number*self.runif01())

    # end of runif_int0N

# ------------------------------------------------------------------------------

    def rsign(self):
        """
        Returns -1.0 or 1.0 with probability 0.5 for each. 
        """

        x = self.runif01()
        if x <= 0.5: return -1.0
        else:        return  1.0

    # end of rsign

# ------------------------------------------------------------------------------

    def runifab(self, left, right): 
        """
        Generator of uniformly distributed floats between 'left' and 'right'. 
        """

        assert right >= left, "support span must not be negative in runifab!"

        x  =  left + (right-left)*self.runif01()

        x  =  kept_within(left, x, right)

        return x

    # end of runifab

# ------------------------------------------------------------------------------

    def rchistogram(self, values, qumul):
        """
        Generates random variates from an input CUMULATIVE histogram.
        'values' is a list/tuple with FLOATS in ascending order - A MUST! 
        These values represent bin end points and must be one more than 
        the number of cumulative frequencies, and where...
        ...'qumul' are the corresponding CUMULATIVE FREQUENCIES such that 
        qumul[k] = P(x<=values[k+1]).
        
        The cumulative frequencies must of course obey qumul[k+1] >= qumul[k],
        otherwise an exception will be raised!
        
        The values of the random variate are assumed to be uniformly 
        distributed within each bin.
        """

        p = self.runif01()

        x = ichistogram(p, values, qumul)

        return x

    # end of rchistogram

# ------------------------------------------------------------------------------

    def rchistogram_int(self, values, qumul):
        """
        Generates random variates from an input CUMULATIVE histogram.
        'values' is a list/tuple with INTEGERS in ascending order - A MUST! 
        These values represent bin end points and must be one more than 
        the number of cumulative frequencies, and where...
        ...'qumul' are the corresponding CUMULATIVE FREQUENCIES such that 
        qumul[k] = P(x<=values[k+1]).

        NB The first element of the values list is will never be returned!
        The first integer to be returned is values[0] + 1   !!!!

        The cumulative frequencies must of course obey qumul[k+1] >= qumul[k],
        otherwise an exception will be raised!
        
        The integer values of the random variate are assumed to be uniformly 
        distributed within each bin.
        """

        p = self.runif01()

        x = ichistogram_int(p, values, qumul)

        return x

    # end of rchistogram_int

# ------------------------------------------------------------------------------

    def rtriang(self, left, mode, right):
        """
        Generator of triangularly distributed random numbers on [left, right] 
        with the peak of the pdf at mode. 
        """

        assert left <= mode and mode <= right, \
                                  "mode out of support range in rtriang!"

        p  =  self.runif01()

        span    =  right - left
        spanlo  =  mode  - left
        spanhi  =  right - mode
        #height  =  2.0 / span
        #surf1   =  0.5 * spanlo * height
        #surf1   =  spanlo/float(span)

        #if p <= surf1:
        if p <= spanlo/float(span):
            #x  =  sqrt(2.0*spanlo*p/height)
            x  =  sqrt(spanlo*span*p)
        else:
            #x  =  span - sqrt(2.0*spanhi*(1.0-p)/height)
            x  =  span - sqrt(spanhi*span*(1.0-p))
        x += left

        x  = kept_within(left, x, right)

        return x

    # end of rtriang

# ------------------------------------------------------------------------------

    def rtri_unif_tri(self, a, b, c, d):
        """
        Triangular-uniform-triangular distribution with support on [a, d] and 
        with breakpoints in b and c
                      ------
        pdf:        /        \
                   /           \
            ------               -------                                         
        """

        # Input check -----------------------
        assert a <= b and b <= c and c <= d, \
                                "break points scrambled in rtri_unif_tri!"
        # -----------------------------------


        if d == a: return a


        dcba   =  d + c - b - a
        h      =  2.0 / dcba
        first  =  0.5 * h * (b-a)
        p      =  self.runif01()
        poh    =  0.5 * p * dcba

        if p <= first:
            x  =  sqrt(2.0*(b-a)*poh) + a
        elif first < p <= first + h*(c-b):
            x  =  (c-b)*(poh-0.5*(b-a)) + b
        else:
            x  =  d - sqrt((d-c)*dcba*(1.0-p))

        x  = kept_within(a, x, d)

        return x

    # end of rtri_unif_tri

# ------------------------------------------------------------------------------

    def rkumaraswamy(self, a, b, x1, x2):
        """
        The Kumaraswamy distribution f = a*b*x**(a-1) * (1-x**a)**(b-1)
                                     F = 1 - (1-x**a)**b
                                     a, b >= 0; 0 <= x <= 1
        The Kumaraswamy is very similar to the beta distribution !!!
        
        x2 >= x1 !!!! 
        """

        assert a  >  0.0, "shape parameters in rkumaraswamy must be positive!"
        assert b  >  0.0, "shape parameters in rkumaraswamy must be positive!"
        assert x2 >= x1,  "support range in rkumaraswamy must not be negative!"

        y  =  (1.0 - (1.0-self.runif01())**(1.0/b)) ** (1.0/a)

        x  =  y*(x2-x1) + x1

        x  =  kept_within(x1, x, x2)

        return x

    # end of rkumaraswamy

# ------------------------------------------------------------------------------

    def rsinus(self, left, right):
        """
        The "sinus distribution". 
        """

        assert right >= left, "support range must not be negative in rsinus!"

        #x  =  left  +  (right-left) * acos(1.0-2.0*self.runif01()) / PI
        x  =  left  +  (right-left) * PIINV*acos(1.0-2.0*self.runif01())

        x  =  kept_within(left, x, right)

        return x

    # end of rsinus

# ------------------------------------------------------------------------------
 
    def rgeometric(self, phi):
        """
        The geometric distribution with p(K=k) = phi * (1-phi)**(k-1)  and 
        P(K>=k) = sum phi * (1-phi)**k = 1 - q**k, where q = 1 - phi and  
        0 < phi <= 1 is the success frequency or "Bernoulli probability" 
        and K >= 1 is the number of  trials to the first success in a series 
        of Bernoulli trials. It is easy to prove that P(k) = 1 - (1-phi)**k: 
        let q = 1 - phi. p(k) = (1-q) * q**(k-1) = q**(k-1) - q**k. Then P(1) = 
        p(1) = 1 - q. P(2) = p(1) + p(2) = 1 - q + q - q**2 = 1 - q**2. 
        Induction can be used to show that P(k) = 1 - q**k = 1 - (1-phi)**k 
        """

        assert 0.0 <= phi and phi <= 1.0, \
                      "success frequency must be in [0.0, 1.0] in rgeometric!"

        if phi == 1.0: return 1   # Obvious...

        p  = self.runif01()

        q  =  1.0 - phi

        if phi < 0.25:            # Use the direct inversion formula
            lnq   = - safelog(q)
            ln1mp = - safelog(1.0 - p)
            kg    =  1 + int(ln1mp/lnq)

        else:             # Looking for the passing point is more efficient for 
            kg = 1        # phi >= 0.25 (it's still inversion)
            u  = p
            a  = phi
            while True:
                u = u - a
                if u > 0.0:
                    kg += 1
                    a  *= q
                else:
                    break

        return kg

    # end of rgeometric

# ------------------------------------------------------------------------------

    def remp_exp(self, values, npexp=0, ordered=False, \
                                        xmax=float('inf'), pmax=1.0): 
        """
        The mixed expirical/exponential distribution from Bratley, Fox and 
        Schrage. A polygon (piecewise linearly interpolated cdf) is used 
        together with a (shifted) exponential for the tail. The procedure 
        is designed so as to preserve the mean of the input sample.
        
        The input is a set of observed points (vector) and an integer 
        representing the npexp largest points that will be used to formulate 
        the exponential tail.
        
        NB it is assumed that x is in [0, inf) (with the usual cutoff 
        provisions)  !!!!!
        
        The function may also be used for a piecewise linear cdf without the 
        exponential tail - corrections are made to preserve the mean in this 
        case as well !!! 
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in remp_exp!"
        self._checkpmax(pmax, 'remp_exp')

        pmx = pmax
        #if xmax < float('inf'):
            #pmx = min(pmax, cemp_exp(values, npexp, ordered, xmax))

        p  =  pmx * self.runif01()
        x  =  iemp_exp(p, values, npexp, ordered)

        return x

    # end of remp_exp

# ------------------------------------------------------------------------------

    def rexpo_gen(self, a, b, c, xmin=float('-inf'), xmax=float('inf'), \
                                 pmin=0.0, pmax=1.0):   
        """
        The generalized continuous double-sided exponential 
        distribution (x in R):
        x <= c: f  =  [a*b/(a+b)] * exp(+a*[x-c])
                F  =   [b/(a+b)]  * exp(+a*[x-c])
        x >= c: f  =  [a*b/(a+b)] * exp(-b*[x-c])
                F  =  1 - [a/(a+b)]*exp(-b*[x-c])
        a > 0, b > 0
        
        NB The symmetrical double-sided exponential sits in rlaplace!
        """

        self._checkminmax(xmin, xmax, pmin, pmax, 'rexpo_gen')

        pmn = pmin
        pmx = pmax
        if xmin > float('-inf'): pmn = max(pmin, cexpo_gen(a, b, c, xmin))
        if xmax < float('inf'):  pmx = min(pmax, cexpo_gen(a, b, c, xmax))

        p  =  pmn + (pmx-pmn)*self.runif01()
        x  =  iexpo_gen(p, a, b, c)

        return x

    # end of rexpo_gen

# ------------------------------------------------------------------------------

    def rlaplace(self, loc, scale, xmin=float('-inf'), xmax=float('inf'), \
                                   pmin=0.0, pmax=1.0):
        """
        The Laplace aka the symmetrical double-sided exponential distribution 
        f = ((1/2)/s)) * exp(-abs([x-l]/s))
        F = (1/2)*exp([x-l]/s)  {x <= 0},  F = 1 - (1/2)*exp(-[x-l]/s)  {x >= 0}
        s >= 0  
        """

        self._checkminmax(xmin, xmax, pmin, pmax, 'rlaplace')

        pmn = pmin
        pmx = pmax
        if xmin > float('-inf'): pmn = max(pmin, claplace(shift, scale, xmin))
        if xmax < float('inf'):  pmx = min(pmax, claplace(shift, scale, xmax))

        p  =  pmn + (pmx-pmn)*self.runif01()
        x  =  ilaplace(p, loc, scale)

        return x

    # end of rlaplace

# ------------------------------------------------------------------------------

    def rtukeylambda_gen(self, lam1, lam2, lam3, lam4, pmin=0.0, pmax=1.0):
        """
        The Friemer-Mudholkar-Kollia-Lin generalized Tukey lambda distribution.
        lam1 is a location parameter and lam2 a scale parameter. lam3 and lam4
        are associated with the shape of the distribution. 
        """

        assert lam2 > 0.0, \
          "shape parameter lam2 must be a positive float in rtukeylambda_gen!"
        assert 0.0 <= pmin < pmax, \
                             "pmin must be in [0.0, pmax) in rtukeylambda_gen!"
        assert pmin < pmax <= 1.0, \
                             "pmax must be in (pmin, 1.0] in rtukeylambda_gen!"        

        p  =  pmin + (pmax-pmin)*self.runif01()

        if lam3 == 0.0:
            q3 = safelog(p)
        else:
            q3 = (p**lam3-1.0) / lam3

        if lam4 == 0.0:
            q4 = safelog(1.0-p)
        else:
            q4 = ((1.0-p)**lam4 - 1.0) / lam4

        x  =  lam1 + (q3-q4)/lam2

        return x

    # end of rtukeylambda_gen

# ------------------------------------------------------------------------------

    def rcauchy(self, location, scale, xmin=float('-inf'), xmax=float('inf'), \
                                       pmin=0.0, pmax=1.0):
        """
        Generator of random variates from the Cauchy distribution: 
        f = 1 / [s*pi*(1 + [(x-l)/s]**2)]
        F = (1/pi)*arctan((x-l)/s) + 1/2
        (also known as the Lorentzian or Lorentz distribution)
        
        scale must be >= 0 
        """

        self._checkminmax(xmin, xmax, pmin, pmax, 'rcauchy')

        pmn = pmin
        pmx = pmax
        if xmin > float('-inf'):
            pmn = max(pmin, ccauchy(location, scale, xmin))
        if xmax < float('inf'):
            pmx = min(pmax, ccauchy(location, scale, xmax))
        
        p  =  pmn + (pmx-pmn)*self.runif01()
        x  =  icauchy(p, location, scale)

        return x

    # end of rcauchy

# ------------------------------------------------------------------------------

    def rextreme_I(self, type, mu, scale, \
                                        xmin=float('-inf'), xmax=float('inf'), \
                                        pmin=0.0, pmax=1.0):
        """
        Extreme value distribution type I (aka the Gumbel distribution or 
        Gumbel distribution type I):
        F = exp{-exp[-(x-mu)/scale]}       (max variant)
        f = exp[-(x-mu)/scale] * exp{-exp[-(x-mu)/scale]} / scale
        F = 1 - exp{-exp[+(x-mu)/scale]}   (min variant)
        f = exp[+(x-mu)/scale] * exp{-exp[+(x-mu)/scale]} / scale

        type must be 'max' or 'min'
        scale must be > 0.0
        """

        self._checkminmax(xmin, xmax, pmin, pmax, 'rextreme_I')

        pmn = pmin
        pmx = pmax
        if xmin > float('-inf'):
            pmn = max(pmin, cextreme_I(type, mu, scale, xmin))
        if xmax < float('inf'):
            pmx = min(pmax, cextreme_I(type, mu, scale, xmax))

        p  =  pmn + (pmx-pmn)*self.runif01()
        x  =  iextreme_I(p, type, mu, scale)

        return x

    # end of rextreme_I

# ------------------------------------------------------------------------------

    def rextreme_gen(self, type, shape, mu, scale, \
                                      xmin=float('-inf'), xmax=float('inf'), \
                                      pmin=0.0, pmax=1.0):
        """
        Generalized extreme value distribution:
        F = exp{-[1-shape*(x-mu)/scale]**(1/shape)}       (max version)
        f = [1-shape*(x-mu)/scale]**(1/shape-1) * 
                                 exp{-[1-shape*(x-mu)/scale]**(1/shape)} / scale
        F = 1 - exp{-[1+shape*(x-mu)/scale]**(1/shape)}   (min version)
        f = [1+shape*(x-mu)/scale]**(1/shape-1) * 
                                 exp{-[1+shape*(x-mu)/scale]**(1/shape)} / scale
        shape  < 0 => Type II
        shape  > 0 => Type III
        shape -> 0 => Type I - Gumbel

        type must be 'max' or 'min'
        scale must be > 0.0

        A REASONABLE SCHEME SEEMS TO BE mu = scale WHICH SEEMS TO LIMIT THE
        DISTRIBUTION TO EITHER SIDE OF THE Y-AXIS!
        """

        self._checkminmax(xmin, xmax, pmin, pmax, 'rextreme_gen')

        pmn = pmin
        pmx = pmax
        if xmin > float('-inf'):
            pmn = max(pmin, cextreme_gen(type, shape, mu, scale, xmin))
        if xmax < float('inf'):
            pmx = min(pmax, cextreme_gen(type, shape, mu, scale, xmax))

        p  =  pmn + (pmx-pmn)*self.runif01()
        x  =  iextreme_gen(p, type, shape, mu, scale)

        return x

    # end of rextreme_gen

# ------------------------------------------------------------------------------

    def rlogistic(self, mu, scale, xmin=float('-inf'), xmax=float('inf'), \
                                   pmin=0.0, pmax=1.0):  
        """
        The logistic distribution: F = 1 / {1 + exp[-(x-m)/s]}; x on R
        m is the mean and mode, and s is a scale parameter (s >= 0) 
        """

        self._checkminmax(xmin, xmax, pmin, pmax, 'rlogistic')

        pmn = pmin
        pmx = pmax
        if xmin > float('-inf'): pmn = max(pmin, clogistic(mu, scale, xmin))
        if xmax < float('inf'):  pmx = min(pmax, clogistic(mu, scale, xmax))

        p  =  pmn + (pmx-pmn)*self.runif01()
        x  =  ilogistic(p, mu, scale)

        return x

    # end of rlogistic

# ------------------------------------------------------------------------------

    def rrayleigh(self, sigma, xmax=float('inf'), pmax=1.0):  
        """
        The Rayleigh distribution:
        f = (x/s**2) * exp[-x**2/(2*s**2)]
        F = 1 - exp[-x**2/(2*s**2)]
        x, s >= 0 
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rrayleigh!"
        self._checkpmax(pmax, 'rrayleigh')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, crayleigh(sigma, xmax))

        p  =  pmx * self.runif01()
        x  =  irayleigh(p, sigma)

        return x

    # end of rrayleigh

# ------------------------------------------------------------------------------

    def rpareto_zero(self, lam, xm, xmax=float('inf'), pmax=1.0):   
        """
        The Pareto distribution with the support shifted to [0, inf):
        f = lam * xm**lam / (x+xm)**(lam+1)
        F = 1 - [xm/(x+xm)]**lam
        x in [0, inf)
        lam > 0
        For lam < 1 all moments are infinite
        For lam < 2 all moments are infinite except for the mean
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rpareto_zero!"
        self._checkpmax(pmax, 'rpareto_zeroero')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, cpareto_zero(lam, xm, xmax))

        p  =  pmx * self.runif01()
        x  =  ipareto_zero(p, lam, xm)

        return x

    # end of rpareto_zero

# ------------------------------------------------------------------------------

    def rkodlin(self, gam, eta, xmax=float('inf'), pmax=1.0):  
        """
        The Kodlin distribution, aka the linear hazard rate distribution:
        f = (gam + eta*x) * exp{-[gam*x + (1/2)*eta*x**2]},
        F = 1 - exp{-[gam*x + (1/2)*eta*x**2]};  x, gam, eta >= 0 
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rkodlin!"
        self._checkpmax(pmax, 'rkodlin')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, ckodlin(scale, xmax))

        p  =  pmx * self.runif01()
        x  =  ikodlin(p, gam, eta)

        return x

    # end of rkodlin

# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------

    def _checkpmax(self, pmax, caller='caller'):

        assert 0.0 <= pmax and pmax <= 1.0, \
                      "pmax must be in [0.0, 1.0] in" + caller + "!"

    # end of _checkpmax

# ------------------------------------------------------------------------------

    def _checkminmax(self, xmin, xmax, pmin, pmax, caller='caller'):

        assert xmax >= xmin,        \
                     "xmax must be >= xmin in " + caller + "!"
        assert  0.0 <= pmin <= pmax, \
                     "pmin must be in [0.0, pmax] in " + caller + "!"
        assert pmin <= pmax <= 1.0,\
                     "pmax must be in [pmin, 1.0] in " + caller + "!"

    # end of _checkminmax

# ------------------------------------------------------------------------------

# end of ABCRand

# ------------------------------------------------------------------------------
# genrandstrm.py
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

from random import Random
from math   import exp, sin, cos, tan, atan, sqrt, modf
from bisect import bisect

from abcrand           import ABCRand
from statlib.pdf       import dnormal
from numlib.specfunc   import lngamma
from numlib.miscnum    import safelog, safediv
from misclib.numbers   import is_posinteger, kept_within
from machdep.machnum   import TWOMACHEPS, ONEMMACHEPS, MAXFLOAT, TINY, HUGE
from misclib.errwarn   import Error, warn
from misclib.mathconst import PI, PIHALF, PIINV, E

# ------------------------------------------------------------------------------

class GeneralRandomStream(ABCRand):
    """
    GENERAL CLASS FOR INITIATING RANDOM NUMBER STREAMS AND CREATING RANDOM 
    VARIATES FROM DIFFERENT PROBABILITY DISTRIBUTIONS 

    NB. All methods return a single random number on each call. 

    The class inherits from the abstract base class ABCRand. Some the methods 
    of Python's built-in Random class are also available. Please consult the 
    docstrings of the methods of the ABCRand and Random classes for information 
    on how to use the inherited methods.
    
    The GeneralRandomStream class is used like this:
        rstream = GeneralRandomStream()
        meanx   = 2.5
        x       = rstream.rexpo(meanx)
        muy     = -1.5
        sigmay  = 3.0
        y       = rstream.rnormal(muy, sigmay)  # or whatever....

    If another seed than the default seed is desired, then the class can be 
    instantiated using rstream = GeneralRandomStream(nseed) where nseed is 
    an integer (preferably a large one).

    A generating method of this class is in many cases much faster than the 
    corresponding inverse-based method provided by the InverseRandomStream 
    class. The Latin Hypercube and antithetic variance reduction methods 
    provided by he RandomStructure class only work with the methods in 
    InverseRandomStream, on the other hand. So speed is based on the generating 
    methods as well as the possibility of using variance reduction. Tests 
    should be made when speed is crucial.

    Synchronized random number sequences are NOT attainable with the methods 
    in this class, so 'common random numbers' will not work (for instance).

    The possibility of of using correlations between parameters is restricted 
    in GeneralRandomStream to pairwise correlation between two normally 
    distributed random variates (chains of correlations may be used, of 
    course).

    It might be desirable at times to limit the otherwise unbound output from 
    a variate generator for practical or physical reasons, for instance (some 
    output values may be unrealistic). A number of the methods of this class 
    allow that. The actual bounds (xmin and xmax) can be given. When the method 
    encountered an undesirable result, this will be rejected and its starts 
    over to create a new result. The reason for using a generator not based on 
    inversion is that it is often faster than its corresponding inversion-based 
    generator, as noted above. Limiting the range of the output variate may 
    alter this however, particularly for tight bounds, so "ten mucho cuidado!". 
    AND: prescribing bounds alters the distribution so the outputs will not 
    truly adhere to the theoretical one. The other input parameters are, 
    however, the parameters for the corresponding full-span theoretical 
    distribution! 

    NB. Methods may return float('inf') or float('-inf') !!!!!
    """

    # Class variables used in methods
    __LN4 = 1.3862943611198906188344642
    # log(4.0, e) i.e. ln(4.0); according to Abramowitz & Stegun
    __ERLANG2GAMMA  = 17   # Must be > __GAMMA2ERLANG
    __GAMMA2ERLANG  =  3.0

# ------------------------------------------------------------------------------

    def __init__(self, nseed=2147483647):
        """
        Initiates the object and sets the seed.
        """

        errtxt  = "The seed must be a positive integer in GeneralRandomStream\n"
        errtxt += "\t(external feeds cannot be used)"
        assert is_posinteger(nseed), errtxt

        ABCRand.__init__(self, nseed)
        self._feed = False

    # end of __init__

# ------------------------------------------------------------------------------

    def rconst(self, const):
        """
        Returns the input constant. 
        """

        return const

    # end of rconst

# ------------------------------------------------------------------------------

    def rbootstrap(self, sequence):
        """
        Picks elements from the input sequence (list, tuple etc) at random 
        (could be any sequence). 
        """

        index = self.runif_int0N(len(sequence))
        return sequence[index]

    # end of rbootstrap

# ------------------------------------------------------------------------------

    def rdiscrete(self, values, qumul):
        """
        Generates random variates from a user-defined discrete cdf. 

        'values' is a list/tuple with numbers, and 'qumul' are the corresponding 
        CUMULATIVE FREQUENCIES such that qumul[k] = P(x<=values[k]). The number 
        of values must be equal to the number of cumulative frequencies.
        
        The cumulative frequencies must of course obey qumul[k+1] >= qumul[k],
        otherwise an exception will be raised!
        """

        # Input check ------------
        errortext1 = "Input vectors are of unequal length in rdiscrete!"
        assert len(values) == len(qumul), errortext1
        assert qumul[-1]  == 1.0
        assert 0.0 < qumul[0] and qumul[0] <= 1.0
        # ---
        nvalues = len(values)
        errortext2 = "qumul list is not in order in rdiscrete!"
        for k in range (1, nvalues):
            assert qumul[k] >= qumul[k-1], errortext2
            pass
        # ---

        p = self.runif01()

        #k = binSearch(qcumul, p)[0]  # Only the first of two outputs is needed
        k = bisect(qumul, p)
        x = values[k]

        return x

    # end of rdiscrete

# ------------------------------------------------------------------------------

    def rbeta(self, a, b, x1, x2):
        """
        The beta distribution f = x**(a-1) * (1-x)**(b-1) / beta(a, b)
        The cdf is the integral = the incomplete beta or the incomplete 
        beta/complete beta depending on how the incomplete beta function 
        is defined.
        x, a, b >= 0; x2 > x1 
        
        The algorithm is due to Berman (1970)/Jonck (1964) (for a and b < 1.0) 
        and Cheng (1978) as described in Bratley, Fox and Schrage.
        """

        assert a  > 0.0, \
                        "shape parameters a and b must both be > 0.0 in rbeta!"
        assert b  > 0.0, \
                        "shape parameters a and b must both be > 0.0 in rbeta!"
        assert x2 >= x1, \
                         "support span must not be negative in rbeta!"

        if x2 == x1: return x1

        if a == 1.0 and b == 1.0:
            y = self.runif01()

        elif is_posinteger(a) and is_posinteger(b):
            nstop = int(a + b - 1.0)
            r = []
            for k in range(0, nstop):
                r.append(self.runif01())
            r.sort()
            y = r[int(a-1.0)]

        elif a < 1.0 and b < 1.0:
            a1 = 1.0 / a
            b1 = 1.0 / b
            while True:
                u = pow(self.runif01(), a1)
                v = pow(self.runif01(), b1)
                if u + v <= 1.0:
                    y = u / (u+v)
                    break

        else:
            alpha = a + b
            if min(a, b) <= 1.0: beta = 1.0 / min(a, b)
            else:                beta = sqrt((alpha-2.0)/(2.0*a*b-alpha))
            gamma = a + 1.0/beta
            while True:
                u1    =  self.runif01()
                u2    =  self.runif01()
                u1    =  kept_within(TINY, u1, ONEMMACHEPS)
                u2    =  kept_within(TINY, u2, HUGE)
                comp1 =  safelog(u1*u1*u2)
                v     =  beta * safelog(safediv(u1, 1.0-u1))
                w     =  a * exp(v)
                comp2 =  alpha*safelog(alpha/(b+w)) + gamma*v - \
                                                     GeneralRandomStream.__LN4
                if comp2 >= comp1:
                    y = w / (b+w)
                    break

        x = y*(x2-x1) + x1
        x = kept_within(x1, x, x2)

        return x

    # end of rbeta

# ------------------------------------------------------------------------------

    def rbinomial(self, n, phi):
        """
        The binomial distribution: p(N=k) = bincoeff * phi**k * (1-phi)**(n-k); 
        n >= 1;  k = 0, 1,...., n  where phi is the frequency or "Bernoulli 
        probability".
        
        Algorithm taken from ORNL-RSIC-38, Vol II (1973). 
        """

        assert is_posinteger(n), \
                          "n must be a positive integer in rbinomial!"
        assert 0.0 < phi and phi < 1.0, \
                   "frequency parameter is out of range in rbinomial!"

        normconst = 10.0
        onemphi   = 1.0 - phi
        if phi < 0.5: w = int(round(normconst * onemphi / phi))
        else:         w = int(round(normconst * phi / onemphi))

        if n > w:
            #-------------------------------------------------------
            k = int(round(self.rnormal(n*phi, sqrt(n*phi*onemphi))))

        else:
            #-------------------------------------------------------
            if phi < 0.25:
                k   = -1
                m   =  0
                phi = - safelog(onemphi)
                while m < n:
                    r  = self.rexpo(1.0)
                    j  = 1 + int(r/phi)
                    m += j
                    k += 1
                if m == n:
                    k += 1

            elif phi > 0.75:
                k   =  n + 1
                m   =  0
                phi = - safelog(phi)
                while m < n:
                    r  = self.rexpo(1.0)
                    j  = 1 + int(r/phi)
                    m += j
                    k -= 1
                if m == n:
                    k -= 1

            else: # if 0.25 <= phi and phi <= 0.75:
                k = 0
                m = 0
                while m < n:
                    r  = self.runif01()
                    if r < phi: k += 1
                    m += 1

        k = kept_within(0, k, n)

        return k

    # end of rbinomial

# ------------------------------------------------------------------------------

    def rpoisson(self, lam, tspan, nmax=False):
        """
        The Poisson distribution: p(N=n) = exp(-lam*tspan) * (lam*tspan)**n / n!
        n = 0, 1, ...., infinity 
        
        A maximum number for the output may be given in nmax - then it must be 
        a positive integer.
        """

        assert  lam  >= 0.0, "Poisson rate must not be negative in rpoisson!"
        assert tspan >= 0.0, "time span must not be negative in rpoisson!"

        if is_posinteger(nmax): nmaxflag = True
        else:                   nmaxflag = False

        lamtau = lam*tspan

        if lamtau < 64.0:
            while True:
                p = self.runif01()
                n = 0
                r = exp(-lamtau)
                c = r
                f = float(lamtau)
                while c <= p:
                    n  = n + 1
                    r *= f/n
                    c += r
                if not nmaxflag:         break
                if nmaxflag and n<=nmax: break

            n = max(0, n)
            return n

        else:
            while True:
                p = self.runif01()
                n = ipoisson(p, lam, tspan)  # Faster than rej'n vs. the Cauchy
                if not nmaxflag:         break
                if nmaxflag and n<=nmax: break

    # end of rpoisson

# ------------------------------------------------------------------------------

    def rexpo(self, mean, xmax=float('inf')):
        """
        Generator of exponentially distributed random variates with 
        mean = 1.0/lambda:
        f = (1/mean) * exp(-x/mean)
        F = 1 - exp(-x/mean).
        mean >= 0.0
        """

        assert mean >= 0.0, "mean must be a non-negative float in rexpo!"
        assert xmax >= 0.0, "variate max must be a non-negative float in rexpo!"

        while True:
            x = - mean * safelog(self.runif01())
            if x <= xmax: break

        return x

    # end of rexpo

# ------------------------------------------------------------------------------

    def rhyperexpo(self, means, qumul, xmax=float('inf')):
        """
        Generates a random number from the hyperexponential distribution 
        f = sumk pk * exp(x/mk) / mk), F = sumk pk * (1-exp(x/mk)) 
        NB Input to the function is the list of CUMULATIVE FREQUENCIES ! 
        """

        assert xmax >= 0.0, "variate max must be non-negative in rhyperexpo!"

        while True:
            mean  =  self.rdiscrete(means, qumul)
            x     =  self.rexpo(mean)
            if x <= xmax: break

        return x

    # end of rhyperexpo

# ------------------------------------------------------------------------------

    def rcoxian(self, means, probs, xmax=float('inf')):
        """
        Generates a random number from the Coxian phased distribution, which 
        is based on the exponential.
        probs is a list of probabilities for GOING ON TO THE NEXT PHASE rather 
        than reaching the absorbing state prematurely. The number of means in 
        the means list must (of course) be one more than the number of 
        probabilities!
        
        NB Use rNexpo when all probs=1.0 ! 
        """

        assert xmax >= 0.0, \
                       "variate max must be a non-negative float in rcoxian!"

        lm = len(means)
        try:
            lp     = len(probs)
            probsl = probs
        except TypeError:   #  probs is not provided as a list
            probsl = [probs]
            lp     = len(probsl)

        assert lp == lm - 1, \
                         "lengths of input lists are not matched in rcoxian!"

        while True:
            x = 0.0
            for k in range(0, lm):
                mean = means[k]
                assert mean >= 0.0, "means must not be negative in rcoxian!"
                try:
                    x += self.rexpo(mean)
                except OverflowError:
                    x  = float('inf')
                if k < lp:
                    p    = self.runif01()
                    prob = probsl[k]
                    assert 0.0 <= prob and prob <= 1.0, \
                                     "probability out of range in rcoxian!"
                    if p >= prob:
                        break
            if x <= xmax: break

        return x

    # end of rcoxian

# ------------------------------------------------------------------------------

    def rNexpo(self, means, xmax=float('inf')):
        """
        Generator of random variates from a distribution of the sum of 
        exponentially distributed random variables. means[k] = 1.0/lambda[k].
        N.B. means is a list or tuple with non-negative numbers.
        If all means[k] are equal the distribution is Erlang. 
        """

        assert xmax >= 0.0, "variate max must be non-negative in rNexpo!"

        while True:
            x = 0.0
            for m in means:
                x += self.rexpo(m)
            if x <= xmax: break

        x = kept_within(0.0, x)

        return x

    # end of rNexpo

# ------------------------------------------------------------------------------

    def rerlang(self, nshape, phasemean, xmax=float('inf')):
        """
        Generator of Erlang-distributed random variates.
        Represents the sum of nshape exponentially distributed random variables, 
        each having the same mean value = phasemean. For nshape = 1 it works as 
        a generator of exponentially distributed random numbers.
        """

        assert is_posinteger(nshape), \
                       "shape parameter must be a positive integer in rerlang!"
        assert phasemean >= 0.0,   "phasemean must not be negative in rerlang!"
        assert xmax >= 0.0,      "variate max must be non-negative in rerlang!"


        if nshape < GeneralRandomStream.__ERLANG2GAMMA:
            while True:
                x  =  1.0
                for k in range(0, nshape):
                    x *= self.runif01() # Might turn out to be zero...
                x  = - phasemean * safelog(x)
                if x <= xmax: break

        else:   # Gamma is OK
            while True:
                x  =  phasemean * self.rgamma(float(nshape), 1.0)
                if x <= xmax: break

        x  =  kept_within(0.0, x)

        return x

    # end of rerlang

# ------------------------------------------------------------------------------

    def rerlang_gen(self, nshapes, qumul, phasemean, xmax=float('inf')):
        """
        The generalized Erlang distribution - the Erlang equivalent of the 
        rhyperexpo generator 
        f = sumk pk * ferlang(m, nk), F = sumk pk * Ferlang(m, nk), the same
        mean for all phases.
        NB Input to the function is the list of CUMULATIVE FREQUENCIES ! 
        """

        assert xmax >= 0.0, "variate max must be non-negative in rerlang_gen!"

        while True:
            nshape = self.rdiscrete(nshapes, qumul)
            x      = self.rerlang(nshape, phasemean)
            if x <= xmax: break

        return x

    # end of rerlang_gen

# ------------------------------------------------------------------------------

    def rexppower(self, loc, scale, alpha, \
                                    xmin=float('-inf'), xmax=float('inf')):
        """
        The exponential power distribution 
        f  =  (a/s) * exp(-abs([x-l]/s)**a) / [2*gamma(1/a)]
        F  =  1/2 * [1 + sgn(x-l) * Fgamma(1/a, abs([x-l]/s)**a)],  x in R
        s, a > 0
        where Fgamma is the gamma distribution cdf.
        
        A modified version of the rejection technique proposed in 
        P.R. Tadikamalla;
        "Random Sampling From the Exponential Power Distribution",
        J. Am. Statistical Association 75(371), 1980, pp 683-686 
        is used (the gamma is used for alpha > 2.0 in place of the rejection 
        procedure proposed by Tadikamalla - for purposes of speed).
        """

        assert xmax > xmin, "xmax must be > xmin in rexppower!"
        assert scale > 0.0, \
                   "scale parameter must be a positive float in rexppower!"
        assert alpha > 0.0, \
                "shape parameter alpha must be a positive float in rexppower!"

        sinv = 1.0/scale
        ainv = 1.0/alpha

        while True:
            if alpha < 1.0:      # The gamma distribution
                rgam =  self.rgamma(ainv, sinv)
                sign =  self.rsign()
                x    =  loc + sign*rgam**ainv

            elif alpha == 1.0:   # The Laplace distribution is used
                x  =  self.rlaplace(d, scale)

            elif 1.0 < alpha < 2.0:   # Tadikamalla's rejection procedure
                ayay  = ainv**ainv
                ayayi = 1.0/ayay
                while True:
                    u1   = self.runif01()
                    if u1 > 0.5: x = - ayay * safelog(2.0*(1.0-u1))
                    else:        x =   ayay * safelog(2.0*u1)
                    ax   = abs(x)
                    lnu2 = safelog(self.runif01())
                    if lnu2  <=  - ax**alpha + ayayi*ax - 1.0 + ainv:
                        break
                x  =  loc + sinv*x
                
            elif alpha == 2.0:   # The normal (Gaussian) distribution
                x  =  self.rnormal(d, scale)

            else:
                '''while True:   # Tadikamalla is slower than the gamma!!!
                    ayay   = ainv**ainv
                    ayayi2 = 1.0/(ayay*ayay)
                    x      = self.rnormal(0.0, ayay)
                    ax     = abs(x)
                    lnu    = safelog(self.runif01())
                    if lnu  <=  - ax**alpha + 0.5*ayayi2*x*x + ainv - 0.5:
                        break'''
                rgam =  self.rgamma(ainv, sinv)   # The gamma distribution
                sign =  self.rsign()
                x    =  loc + sign*rgam**ainv

            if xmin <= x <= xmax: break

        return x

    # end of rexppower

# ------------------------------------------------------------------------------

    def rgamma(self, alpha, lam, xmax=float('inf')):
        """
        The gamma distribution:
        f = lam * exp(-lam*x) * (lam*x)**(alpha-1) / gamma(alpha)
        The cdf is the integral = the incomplete gamma ratio.
        x, alpha >= 0; lam > 0.0
         
        The generator is a slight modification of Python's 
        built-in "gammavariate". 
        """

        assert alpha >= 0.0, "alpha must be non-negative in rgamma!"
        assert  lam  >  0.0, "lambda must be a positive float in rgamma!"
        assert xmax  >= 0.0, \
                    "variate max must be a non-negative float in rgamma!"

        f, i = modf(alpha)
        if f == 0.0:
            if 1.0 <= i and i <= GeneralRandomStream.__GAMMA2ERLANG:
                return self.rerlang(int(i), 1.0/lam, xmax)

        if alpha < 1.0:
            # Uses ALGORITHM GS of Statistical Computing - Kennedy & Gentle
            # (according to Python's "gammavariate")
            alphainv = 1.0 / alpha
            alpham1  = alpha - 1.0
            while True:
                while True:
                    u = self.runif01()
                    b = (E+alpha) / E
                    p = b * u
                    if p <= 1.0:
                        w = p ** alphainv
                    else:
                        w = -safelog((b-p)*alphainv)
                    u1 = self.runif01()
                    if p > 1.0:
                        if u1 <= w ** (alpham1):
                            break
                    elif u1 <= exp(-w):
                        break
                x = w / lam
                if x <= xmax: break

        else:   # elif alpha > 1.0:
            # Uses R.C.H. Cheng, "The generation of Gamma
            # variables with non-integral shape parameters",
            # Applied Statistics, (1977), 26, No. 1, p71-74
            # (according to Python's "gammavariate")

            ainv = sqrt(2.0*alpha - 1.0)
            beta = 1.0 / ainv
            bbb  = alpha - GeneralRandomStream.__LN4
            ccc  = alpha + ainv

            while True:
                while True:
                    u1 = self.runif01()
                    u2 = self.runif01()
                    v  = beta * safelog(safediv(u1, 1.0-u1))
                    w  = alpha * exp(v)
                    c1 = u1 * u1 * u2
                    r  = bbb + ccc*v - w
                    c2 = r + 2.5040773967762742 - 4.5*c1
                                        # 2.5040773967762742 = 1.0 + log(4.5)
                    if c2 >= 0.0 or r >= safelog(c1):
                        break
                x = w / lam
                if x <= xmax: break

        x = kept_within(0.0, x)
        return x

    # end of rgamma

# ------------------------------------------------------------------------------

    def rnormal(self, mu, sigma, xmin=float('-inf'), xmax=float('inf')):
        """
        Generator of normally distributed random variates. Based on the 
        Kinderman-Ramage algorithm as modified by Tirler, Dalgaard, Hoermann & 
        Leydold. 
        
        sigma >= 0.0
        """

        assert sigma >= 0.0, \
                 "standard deviation must be a non-negative float in rnormal!"
        assert xmax > xmin, "xmax must be > xmin in rnormal!"

        ksi   = 2.2160358671
        ksi2h = 0.5*ksi*ksi
        p18   = 0.180025191068563
        p48   = 0.479727404222441

        while True:

            u = self.runif01()

            if u < 0.884070402298758:
                v = self.runif01()
                x = ksi * (1.131131635444180*u + v - 1.0)

            elif u >= 0.973310954173898:
                while True:
                    v = self.runif01()
                    w = self.runif01()
                    t = ksi2h - safelog(w)
                    if v*v*t <= ksi2h:
                        break
                if u < 0.986655477086949:
                    x =  sqrt(2.0*t)
                else:
                    x = -sqrt(2.0*t)

            elif u >= 0.958720824790463:
                while True:
                    v = self.runif01()
                    w = self.runif01()
                    z = v - w
                    t = ksi - 0.630834801921960*min(v, w)
                    if max(v, w) <= 0.755591531667601:
                        if z < 0.0: x =  t
                        else:       x = -t
                        break
                    p = dnormal(0.0, 1.0, t)
                    f = p - p18*max(ksi-abs(t), 0.0)
                    if 0.034240503750111*abs(z) <= f:
                        if z < 0.0: x =  t
                        else:       x = -t
                        break

            elif u >= 0.911312780288703:
                while True:
                    v = self.runif01()
                    w = self.runif01()
                    z = v - w
                    t = p48 + 1.105473661022070*min(v, w)
                    if max(v, w) <= 0.872834976671790:
                        if z < 0.0: x =  t
                        else:       x = -t
                        break
                    p = dnormal(0.0, 1.0, t)
                    f = p - p18*max(ksi-abs(t), 0.0)
                    if 0.049264496373128*abs(z) <= f:
                        if z < 0.0: x =  t
                        else:       x = -t
                        break

            else:
                while True:
                    v = self.runif01()
                    w = self.runif01()
                    z = v - w
                    t = p48 - 0.595507138015940*min(v, w)
                    if t >= 0.0:
                        if max(v, w) <= 0.805577924423817:
                            if z < 0.0: x =  t
                            else:       x = -t
                            break
                        p = dnormal(0.0, 1.0, t)
                        f = p - p18*max(ksi-abs(t), 0.0)
                        if 0.053377549506886*abs(z) <= f:
                            if z < 0.0: x =  t
                            else:       x = -t
                            break

            x = sigma*x + mu
            if xmin <= x <= xmax: break

        return x

    # end of rnormal

# ------------------------------------------------------------------------------

    def rcorr_normal(self, rho, muy, sigmay, mux, sigmax, x, \
                                        ymin=float('-inf'), ymax=float('inf')):
        """
        Generator of a normally distrubuted random variate that is correlated 
        with another normally distributed random variate.
        Takes the normally distributed random number of the "leader distri-
        bution" as an input (besides mu and sigma for the two distributions). 
        A number can also be generated from scratch if None is input for the 
        leader (hard to say when this would be useful, though...)
        From Kleijnen, J P C; "Statistical Techniques in Simulation, Part 1",
        Marcel Dekker, N Y 1974.
        """

        assert abs(rho) <= 1.0, \
                    "corr. coeff. must be between -1.0 and 1.0 in rcorr_normal!"
        assert sigmay   >  0.0, \
                         "standard deviations must be positive in rcorr_normal!"
        assert sigmax   >  0.0, \
                         "standard deviations must be positive in rcorr_normal!"
        assert  ymax    > ymin, "ymax must be > ymin in corr_normal!"

        while True:
            if x == None:
                y = self.rnormal(muy, sigmay)

            else:
                aux1  =  rho * sigmay / float(sigmax)
                aux2  =  sqrt(1.0-rho**2) * sigmay**2
                z     =  self.rnormal(0.0, 1.0)
                y     =  muy + aux1*(x-mux) + aux2*z

            if ymin <= y and y <= ymax: break

        return y

    # end of rcorr_normal

# ------------------------------------------------------------------------------

    def rwiener(self, tau, wmin=float('-inf'), wmax=float('inf')):
        """
        Generates random numbers corresponding to a Wiener process 
        (the integral of white noise (Langevin's function)). 
        The Wiener process is W(t+tau) - W (t) = N(0, sqrt(tau)) 
        where tau is the time increment and N(0, sqrt(tau)) is a  
        normally distributed random variable having zero mean and 
        sqrt(tau) as its standard deviation.

        This method returns W(t+tau) - W (t) given tau and allows 
        tau to be negative; abs(tau) is used.
        """

        assert wmax > wmin, "wmax must be > wmin in rwiener!"

        c = sqrt(abs(tau))        

        while True:
            w = self.rnormal(0.0, c)
            if wmin <= w and w <= wmax: break

        return w

    # end of rwiener

# ------------------------------------------------------------------------------

    def rlognormal(self, mulg, sigmalg, xmax=float('inf')):
        """
        Generator of lognormally distributed random variates.
        The log10-converted form is assumed for mulg and sigmalg:  
        mulg is the mean of the log10 (and the log10 of the median) of 
        the random variate, NOT the log10 of the mean of the non-logged 
        variate!, and sigmalg is the standard deviation of the log10 of 
        the random variate, NOT the log10 of the standard deviation of 
        the non-logged variate!!
        
        sigmalg > 0.0
        """

        assert xmax  >= 0.0, "variate max must be non-negative in rlognormal!"
        # sigmalg is checked in rnormal

        while True:
            x = self.rnormal(mulg, sigmalg)
            x = pow(10.0, x)
            x = kept_within(0.0, x)
            if x <= xmax: break

        return x

    # end of rlognormal

# ------------------------------------------------------------------------------

    def rfoldednormal(self, muunfold, sigmaunfold, xmax=float('inf')):
        """
        The distribution of a random variable that is the absolute value
        of a variate drawn from the normal distribution (i. e. the distribution 
        of a variate that is the absolute value of a normal variate, the latter 
        having muunfold as its mean and sigmaunfold as its standard deviation). 
        
        sigmaunfold >= 0.0
        """

        assert xmax  >= 0.0, \
                  "variate max must be a non-negative float in rfoldednormal!"

        while True:
            x  =  abs(self.rnormal(muunfold, sigmaunfold))
            if x <= xmax: break

        return x

    # end of rfoldednormal

# ------------------------------------------------------------------------------

    def rweibull(self, c, scale, xmax=float('inf')):
        """
        Generator of random variates from the Weibull distribution.
        F = 1 - exp[-(x/s)**c]
        s >= 0.0, c >= 1.0
        For c = 1.0 it is equivalent to a generator of exponential variates 
        with mean = scale.

        The method uses Python's built-in method "weibullvariate", which is 
        actually an inverse method, but it does not provide for fractile span 
        scaling. The reason for it being used here is that it is normally 
        slighly faster than the inverse method of the corresponding 
        InverseRandomStream method (ca. 20 %). It may, on the other hand, 
        use more than one basic uniform variate per call, particularly for 
        small xmax.
        """
        
        assert   c   >= 1.0, \
                      "shape parameter must not be less than 1.0 in rweibull!"
        assert scale >= 0.0, \
                           "scale parameter must not be negative in rweibull!"
        assert xmax  >  0.0, \
                       "variate max must be a non-negative float in rweibull!"

        if c == 1.0:
           while True:
               x = self.rexpo(scale)
               if x <= xmax: break

        else:
            while True:
                try:
                    x = self._weibullvariate(scale, c)
                except ValueError:
                    x = float('inf')
                if x <= xmax: break

        return x

    # end of rweibull

# ------------------------------------------------------------------------------

    def rpareto(self, lam, xm, xmax=float('inf')):
        """
        The Pareto distribution:
        f = lam * xm**lam / x**(lam+1)
        F = 1 - (xm/x)**lam
        x in [xm, inf) ; lam > 0
        For lam < 1 all moments are infinite
        For lam < 2 all moments are infinite except for the mean
        """
        # Uses Python's built-in method "paretovariate" which is actually an 
        # inverse method, but it does not provide for fractile span scaling. 
        # The reason for it being available here is that it is normally slightly 
        # faster than the inverse (ca. 25 %). It may, on the other hand, use 
        # more than one basic uniform variate per call, particularly for small 
        # xmax.
        
        assert lam  >  0.0, \
                           "shape parameter lambda must be positive in rpareto!"
        assert xm   >= 0.0, \
              "left support limit parameter xm must not be negative in rpareto!"
        assert xmax >  0.0, \
                          "variate max must be a non-negative float in rpareto!"

        while True:
            try:
                x  =  xm * self._paretovariate(lam)
            except OverflowError:
                x  =  float('inf')
            if x <= xmax: break

        return x

    # end of rpareto

# ------------------------------------------------------------------------------

    def rstable(self, alpha, beta, location, scale, \
                                    xmin=float('-inf'), xmax=float('inf')):
        """
        rstable generates random variates from any distribution of the stable 
        family. 0 < alpha <= 2 is the tail index (the smaller alpha, the wider 
        the distribution). -1 <= beta <= 1 is the skewness parameter (the 
        greater absolute value, the more skewed the distribution, negative = 
        skewed to the left, positive = skewed to the right). 
        
        For beta = 0 and alpha = 2 the distribution is a Gaussian distribution -
        but scaled with sqrt(2) (the variance of the stable is twice that of the
        Gaussian). For beta = 0 and alpha = 1 the distribution is the Cauchy. 
        For abs(beta) = 1 and alpha = 0.5 the distribution is the Levy.
        
        For alpha < 2.0 the variance and higher moments are infinite, and for 
        alpha <= 1 all moments are infinite. A scale parameter and a location 
        parameter are also applicable so that Y = scale*X + location is still 
        a stable variate with the same alpha as that of X. E{Y} = location for 
        alpha > 1.
        
        The algorithm is taken from Weron but the original version seems to be 
        Chambers, Mallows and Stuck: 
        Weron R. (1996): 'On the Chambers-Mallows-Stuck method for simulating 
        skewed stable random variates', Statist. Probab. Lett. 28, 165-171. 
        Chambers, J.M., Mallows, C.L. and Stuck, B.W. (1976): 'A Method for 
        simulating stable random variables', J. Amer. Statist. Assoc. 71, 
        340-344.
        Weron, R. (1996): 'Correction to: On the Chambers-Mallows-Stuck Method 
        for Simulating Skewed Stable Random Variables', Research Report 
        HSC/96/1, Wroclaw University of Technology. 
        """

        assert  0.0 < alpha and alpha <= 2.0, \
                                      "alpha must be in (0.0, 2.0] in rstable!"
        assert -1.0 <= beta and  beta <= 1.0, \
                                      "beta must be in [-1.0, 1.0] in rstable!"
        assert  xmax > xmin,                  "xmax must be > xmin in rstable!"

        while True:
            v = self.runifab(-PIHALF, PIHALF)
            w = self.rexpo(1.0)

            if beta == 0 or beta == 0.0:
                oneoa =  1.0 / float(alpha)
                av    =  alpha * v
                x1    =  sin(av) / cos(v)**oneoa
                x2    =  (cos(v-av)/w) ** (oneoa-1.0)
                x     =  x1 * x2
                x     =  scale*x + location

            elif alpha == 1 or alpha == 1.0:
                pihpbv = PIHALF + beta*v
                x1     = pihpbv * tan(v)
                x2     = beta * safelog(PIHALF*w*cos(v)/pihpbv)
                #x      = (x1-x2) / PIHALF
                #x  = scale*x + (1.0/PIHALF)*beta*scale*safelog(scale) + location
                x      = 2.0*PIINV * (x1-x2) / PIHALF
                x   = scale*x + (2.0*PIINV)*beta*scale*safelog(scale) + location

            else:
                tpiha =  tan(PIHALF*alpha)
                oneoa =  1.0 / float(alpha)
                bab   =  atan(beta*tan(PIHALF*alpha))
                sab   =  (1.0 + beta*beta*tpiha*tpiha) ** (0.5*oneoa)
                vpbab =  v + bab
                av    =  alpha * vpbab
                x1    =  sin(av) / cos(v)**oneoa
                x2    =  (cos(v-av) / w) ** (oneoa-1.0)
                x     =  sab * x1 * x2
                x     =  scale*x + location

            if xmin <= x and x <= xmax: break

        return x

    # end of rstable

# ------------------------------------------------------------------------------

    def rstable_sym(self, alpha, location, scale, \
                                 xmin=float('-inf'), xmax=float('inf')):  
        """
        The SYMMETRICAL stable distribution where alpha is the tail exponent 
        This is the general stable distribution with the skewness parameter 
        (beta) = 0.0
        """

        return self.rstable(alpha, 0.0, location, scale, xmin, xmax)

    # end of rstable_sym

# ------------------------------------------------------------------------------

    def rlevy(self, scale, xmax=float('inf')):
        """
        The Levy distribution: f = sqrt(s/2pi) * (1/x)**(3/2) * exp(-s/2x)
                               F = erfc(sqrt(s/2x)); x >= 0
        (stable distribution with alpha = 1/2, and beta = 1, aka the Cournot 
        distribution or the right-skewed Levy).
        """

        assert scale >= 0.0, \
                       "scale parameter must be a non-negative float in rlevy!"
        assert xmax  >= 0.0, \
                           "variate max must be a non-negative float in rlevy!"

        while True:
            x = self.rnormal(0.0, 1.0)
            x = scale / (x*x)
            if x <= xmax: break

        return x

    # end of rlevy

# ------------------------------------------------------------------------------

    def ruser_defined(self, ifunc, *args, \
                            xmin=float('-inf'), xmax=float('inf')):
        """
        Random deviate generation based on a user-defined inverse cdf placed in 
        a function ('ifunc') with arguments p in [0.0, 1.0] and sequence *args.
        
        NB 1  *args is optional in that it is OK for 'ifunc' to have 
        zero arguments.

        NB 2  THE VALUES OF xmin AND xmax MUST - AND CAN ONLY - BE PASSED 
        BY KEYWORD: xmin=??, xmax=?? MUST ALWAYS BE WRITTEN OUT EXPLICITLY 
        UNLESS DEFAULTS ARE USED ('xmin' may be entered without explicit 
        keywording of xmax if the default is accepted for xmax).
        """

        assert xmax > xmin, "xmax must be > xmin in ruser_defined!"

        while True:
            x  =  ifunc(self.runif01(), *args)
            if xmin <= x <= xmax: break

        return x

    # end of ruser_defined

# ------------------------------------------------------------------------------

# end of GeneralRandomStream

# ------------------------------------------------------------------------------
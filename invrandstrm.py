# invrandstrm.py
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

from abcrand         import ABCRand
from statlib.invcdf  import idiscrete, ibeta, ipoisson, iexpo, ihyperexpo
from statlib.invcdf  import iNexpo, iNexpo2, ierlang, ierlang_gen, icoxian
from statlib.invcdf  import icoxian2, ipareto, igamma, iweibull, inormal
from statlib.invcdf  import ifoldednormal, istable_sym, ilevy, iexppower
from statlib.cdf     import cexpo, chyperexpo, cNexpo, cNexpo2, cerlang
from statlib.cdf     import cerlang_gen, ccoxian, ccoxian2, cpareto, cgamma
from statlib.cdf     import cpoisson, cweibull, cexppower
from statlib.cdf     import cnormal, cfoldednormal, cstable_sym, clevy
from misclib.numbers import is_posinteger
from machdep.machnum import MAXFLOAT, FOURMACHEPS

# ------------------------------------------------------------------------------

class InverseRandomStream(ABCRand):
    """
    GENERAL CLASS FOR INITIATING RANDOM NUMBER STREAMS AND CREATING RANDOM 
    VARIATES FROM DIFFERENT PROBABILITY DISTRIBUTIONS 

    NB. All methods return a single random number on each call. 

    The class inherits the methods explicitly available from the abstract base 
    class ABCRand. The methods of Python's built-in Random class inherited via 
    ABCRand can not be guaranteed to be inverse-based. They are killed off 
    here and NOT available in InverseRandomStream! Please consult the docstrings
    of the methods of the ABCRand class for information on how to use the 
    methods that ARE inherited.
    
    The class is normally used like this:
        rstream = InverseRandomStream()
        meanx   = 2.5
        x       = rstream.rexpo(meanx)
        muy     = -1.5
        sigmay  = 3.0
        y       = rstream.rnormal(muy, sigmay)  # or whatever....

    If another seed than the default seed is desired, then the class can be 
    instantiated using rstream = InverseRandomStream(nseed) where nseed is 
    an integer (preferably a large one).

    A generating method of this class is in many cases much slower than the 
    corresponding method provided by the GeneralRandomStream class. The Latin 
    Hypercube and antithetic variance reduction methods provided by the 
    RandomStructure class only work with the methods in InverseRandomStream, 
    on the other hand. So speed is based on the generating methods as well as 
    the possibility of using variance reduction. Tests should be made when 
    speed is crucial.

    InverseRandomStream makes it possible to use the RandomStructure class 
    to enter rank correlations between random parameters regardless of 
    distribution type, as well as correlated multinormal distributions.

    It might be desirable at times to bound the output from a variate 
    generator for practical or physical reasons, for instance (some output 
    values might be unrealistic). A majority of the methods in this class 
    allow that. The actual bounds can be given (xmin and xmax), or the 
    corresponding cdf values of the unbound distribution can be specified 
    (pmin and pmax), a faster alternative since the method will otherwise 
    compute the corresponding cdf values every time the method is used. 
    Prior cdf value bounds may be computed by calling the corresponding 
    function in the statlib.cdf module.
    BUT: 
    prescribing limits alters the distribution so the outputs will not truly 
    adhere to the theoretical one. The input parameters (mean, standard 
    deviation etc) are, anyhow, for the corresponding full-span theoretical 
    distribution! 

    NB The reason for using a generator not based on inversion is that a 
    particular generator not using inversion is faster. Bounding the range 
    of the output variate may alter this however, particularly for tight 
    bounds, so "ten mucho cuidado!". AND prescribing limits alters the 
    distribution so the outputs will not truly adhere to the theoretical 
    one. The input parameters (mean, std dev etc) are, however, for the
    corresponding full-span theoretical distribution! 

    An externally generated input [0.0, 1.0] stream or "feed" can also be 
    used instead of letting the object instance pick the random variates 
    on its own. Cf. the docstrings of the ABCRand class for details!
    
    NB. Some methods may return float('inf') or float('-inf') !
    """

    # For all methods calling functions from an inverse library it is assumed 
    # that the checking of input parameters is made in the respective inverse 
    # function. 

# ------------------------------------------------------------------------------

    def __init__(self, nseed=2147483647):
        """
        The seed 'nseed' must be a positive integer or a feed (a list or 
        a tuple) of numbers in [0.0, 1.0]!
        """

        if isinstance(nseed, int):
            errtxt  = "The seed (if not a feed) must be a positive\n"
            errtxt += "\tinteger in InverseRandomStream!"
            assert is_posinteger(nseed), errtxt

        ABCRand.__init__(self, nseed, 'InverseRandomStream')

    # end of __init__

# ------------------------------------------------------------------------------

    def rconst(self, const):
        """ 
        Returns the input constant. Use it to keep up the synchronization 
        even when a parameter has no spread (a dummy random number is sampled 
        each time the method is called)!
        """

        self.runif01(); return const

    # end of rconst

# ------------------------------------------------------------------------------

    def rbootstrap(self, values):
        """
        Picks elements from the input sequence (list, tuple etc) at random 
        (could be any sequence). The input sequence is sorted to assure 
        inverse properties. 
        """
   
        index = self.runif_int0N(len(sequence))
        return sorted(sequence)[index]

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
    
        The 'values' sequence is sorted to assure inverse properties. 
        """

        p  =  self.runif01()
        x  =  idiscrete(p, values, qumul)

        return x

    # end of rdiscrete

# ------------------------------------------------------------------------------


    def rbeta(self, a, b, x1, x2, betaab=False):
        """
        The beta distribution f = x**(a-1) * (1-x)**(b-1) / beta(a, b)
        The cdf is the integral = the incomplete beta or the incomplete 
        beta/complete beta depending on how the incomplete beta function 
        is defined.
        
        x, a, b >= 0; x2 > x1 
        
        NB It is possible to provide the value of the complete beta 
        function beta(a, b) as a pre-computed input (may be computed 
        using numlib.specfunc.beta) instead of the default "False", a 
        feature that will make rbeta 30 % faster!
        """

        p  =  self.runif01()
        x  =  ibeta(p, a, b, x1, x2, betaab)

        return x

    # end of rbeta

# ------------------------------------------------------------------------------

    def rpoisson(self, lam, tspan, nmax=False, pmax=1.0):
        """
        The Poisson distribution: p(N=n) = exp(-lam*tspan) * (lam*tspan)**n / n!
        n = 0, 1, ...., infinity

        A maximum number for the output may be given in nmax - then it must be 
        a positive integer.
        """

        pmx = pmax
        if is_posinteger(nmax): pmx = cpoisson(lam, tspan, nmax)

        p  =  pmx * self.runif01()
        n  =  ipoisson(p, lam, tspan)

        return n

    # end of rpoisson

# ------------------------------------------------------------------------------

    def rexpo(self, mean, xmax=float('inf'), pmax=1.0):
        """
        Generator of exponentially distributed random variates with 
        mean = 1.0/lambda
        F = 1 - exp(-x/mean) 
        mean >= 0.0
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rexpo!"
        self._checkpmax(pmax, 'rexpo')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, cexpo(mean, xmax))

        p  =  pmx * self.runif01()
        x  =  iexpo(p, mean)

        return x

    # end of rexpo

# ------------------------------------------------------------------------------

    def rhyperexpo(self, means, qumul, xmax=float('inf'), pmax=1.0):
        """
        Generates a random number from the hyperexponential distribution 
        f = sumk pk * exp(x/mk) / mk, F = sumk pk * (1-exp(x/mk)) 
        
        NB Input to the function is the list of CUMULATIVE FREQUENCIES ! 
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rhyperexpo!"
        self._checkpmax(pmax, 'rhyperexpo')

        pmx = pmax
        if xmax < float('inf'):
            pmx = min(pmax, chyperexpo(means, qumul, xmax))

        p  =  pmx * self.runif01()
        x  =  ihyperexpo(p, means, qumul)

        return x

    # end of rhyperexpo

# ------------------------------------------------------------------------------

    def rNexpo(self, means, xmax=float('inf'), pmax=1.0):
        """
        Generator of random variates from a distribution of the sum of 
        exponentially distributed random variables. means[k] = 1.0/lambda[k].
        NB Numbers in the means vector can be exactly equal! This generator 
        is slow, however... 
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rNexpo!"
        self._checkpmax(pmax, 'rNexpo')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, cNexpo(means, xmax))

        p  =  pmx * self.runif01()
        x = iNexpo(p, means)

        return x

    # end of rNexpo

# ------------------------------------------------------------------------------

    def rNexpo2(self, means, xmax=float('inf'), pmax=1.0):
        """
        Generator of random variates from a distribution of the sum of 
        exponentially distributed random variables. means[k] = 1.0/lambda[k].
        NB No two numbers in the means vector can be exactly equal! If this 
        is desired, use rNexpo ! 
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rNexpo2!"
        self._checkpmax(pmax, 'rNexpo2')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, cNexpo2(means, xmax))

        p  =  pmx * self.runif01()
        x = iNexpo2(p, means)

        return x

    # end of rNexpo2

# ------------------------------------------------------------------------------

    def rerlang(self, nshape, phasemean, xmax=float('inf'), pmax=1.0):  
        """
        Generator of Erlang-distributed random variates.
        Represents the sum of nshape exponentially distributed random variables, 
        each having the same mean value = phasemean. For nshape = 1 it works as 
        a generator of exponentially distributed random numbers.
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rerlang!"
        self._checkpmax(pmax, 'rerlang')

        pmx = pmax
        if xmax < float('inf'):
            pmx = min(pmax, cerlang(nshape, phasemean, xmax))

        p  =  pmx * self.runif01()
        x = ierlang(p, nshape, phasemean)

        return x

    # end of rerlang

# ------------------------------------------------------------------------------

    def rerlang_gen(self, nshapes, qumul, phasemean, \
                          xmax=float('inf'), pmax=1.0):
        """
        The generalized Erlang distribution - the Erlang equivalent of the 
        rhyperexpo generator 
        f = sumk pk * ferlang(m, nk), F = sumk pk * Ferlang(m, nk), the same
        mean for all phases.
        
        NB Input to the function is the list of CUMULATIVE FREQUENCIES ! 
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rerlang_gen!"
        self._checkpmax(pmax, 'rerlang_gen')

        pmx = pmax
        if xmax < float('inf'): \
               pmx = min(pmax, cerlang_gen(nshapes, qumul, phasemean, xmax))

        p  =  pmx * self.runif01()
        x  =  ierlang_gen(p, nshapes, qumul, phasemean)

        return x

    # end of rerlang_gen

# ------------------------------------------------------------------------------

    def rcoxian(self, means, probs, xmax=float('inf'), pmax=1.0):
        """
        Generates a random number from the Coxian phased distribution, which is 
        based on the exponential.
        probs is a list of probabilities for GOING ON TO THE NEXT PHASE rather 
        than reaching the absorbing state prematurely. The number of means must 
        (of course) be one more than the number of probabilities!
        
        NB means are allowed to be equal!
        
        NB It is better to use rNexpo when all probs=1.0 ! 
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rcoxian!"
        self._checkpmax(pmax, 'rcoxian')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, ccoxian(means, probs, xmax))

        p  =  pmx * self.runif01()
        x  =  icoxian(p, means, probs)

        return x

    # end of rcoxian

# ------------------------------------------------------------------------------

    def rcoxian2(self, means, probs, xmax=float('inf'), pmax=1.0):
        """
        Generates a random number from the Coxian phased distribution, which is 
        based on the exponential.
        probs is a list of probabilities for GOING ON TO THE NEXT PHASE rather 
        than 
        reaching the absorbing state prematurely. The number of means must (of 
        course) be one more than the number of probabilities!
        
        NB No two means must be equal - if this is desired, use rcoxian instead! 
        
        NB It is better to use rNexpo2 when all probs=1.0 ! 

        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rcoxian2!"
        self._checkpmax(pmax, 'rcoxian2')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, ccoxian2(means, probs, xmax))

        p  =  pmx * self.runif01()
        x  =  icoxian2(p, means, probs)

        return x

    # end of rcoxian2

# ------------------------------------------------------------------------------

    def rpareto(self, lam, xm, xmax=float('inf'), pmax=1.0):  
        """
        The Pareto distribution:
        f = lam * xm**lam / x**(lam+1)
        F = 1 - (xm/x)**lam 
        x >= xm,  lam > 0 
        For lam < 1 all moments are infinite
        For lam < 2 all moments are infinite except for the mean
        """

        assert xmax >= xm, "xmax must be >= xm in rpareto!"
        self._checkpmax(pmax, 'rpareto')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, cpareto(lam, xm, xmax))

        p  =  pmx * self.runif01()
        x  =  ipareto(p, lam, xm)

        return x

    # end of rpareto

# ------------------------------------------------------------------------------

    def rexppower(self, loc, scale, alpha, lngam1oalpha=False, \
                  xmin=float('-inf'), xmax=float('inf'), pmin=0.0, pmax=1.0, \
                  tolf=FOURMACHEPS, itmax=128):
        """
        The exponential power distribution 
        f  =  (a/s) * exp(-abs([x-l]/s)**a) / [2*gamma(1/a)]
        F  =  1/2 * [1 + sgn(x-l) * Fgamma(1/a, abs([x-l]/s)**a)],  x in R
        s, a > 0
        where Fgamma is the gamma distribution cdf.

        NB It is possible to gain efficiency by providing the value of the 
        natural logarithm of the complete gamma function ln(gamma(1.0/alpha)) 
        as a pre-computed input (may be computed using numlib.specfunc.lngamma) 
        instead of the default 'False'.
    
        tolf and itmax are the numerical control parameters of iexppower 
        and cexppower.
        """

        self._checkminmax(xmin, xmax, pmin, pmax, 'rexppower')

        if alpha==1.0: return self.rlaplace(loc, scale, xmin, xmax, pmin, pmax)

        pmn = pmin
        pmx = pmax
        if xmin > float('-inf'): pmn = max(pmin, cexppower(loc, scale, alpha, \
                                            xmin, lngam1oalpha, tolf, itmax))
        if xmax < float('inf'):  pmx = min(pmax, cexppower(loc, scale, alpha, \
                                            xmax, lngam1oalpha, tolf, itmax))

        assert scale > 0.0, \
                   "scale parameter must be a positive float in rexppower!"
        assert alpha > 0.0, \
                "shape parameter alpha must be a positive float in rexppower!"

        p  =  pmn + (pmx-pmn)*self.runif01()
        x  =  iexppower(p, loc, scale, alpha, lngam1oalpha, tolf, itmax)

        return x

    # end of rexppower

# ------------------------------------------------------------------------------

    def rgamma(self, alpha, lam, lngamalpha=False, \
                                 xmax=float('inf'), pmax=1.0, \
                                 tolf=FOURMACHEPS, itmax=128):
        """
        The gamma distribution
        f = lam * exp(-lam*x) * (lam*x)**(alpha-1) / gamma(alpha)
        The cdf is the integral = the incomplete gamma ratio.
        x, lam, alpha >= 0 

        NB It is possible to provide the value of the natural logarithm of the 
        complete gamma function lngamma(alpha) as a pre-computed input (may be 
        computed using numlib.specfunc.lngamma) instead of the default "False", 
        a feature that will make rgamma 50 % faster! 
        
        tolf and itmax are the numerical control parameters of igamma 
        and cgamma.
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rgamma!"
        self._checkpmax(pmax, 'rgamma')

        pmx = pmax
        if xmax < float('inf'): \
             pmx = min(pmax, cgamma(alpha, lam, xmax, lngamalpha, tolf, itmax))

        p  =  pmx * self.runif01()
        x  =  igamma(p, alpha, lam, lngamalpha, tolf, itmax)

        return x

    # end of rgamma

# ------------------------------------------------------------------------------

    def rweibull(self, c, scale, xmax=float('inf'), pmax=1.0):
        """
        Generator of random variates from the Weibull distribution.
        F = 1 - exp[-(x/s)**c]
        s >= 0.0, c >= 1.0
        For c = 1.0 it is equivalent to a generator of exponential variates 
        with mean = scale 
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rweibull!"
        self._checkpmax(pmax, 'rweibull')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, cweibull(c, scale, xmax))

        p  =  pmx * self.runif01()
        x  =  iweibull(p, c, scale)

        return x

    # end of rweibull

# ------------------------------------------------------------------------------

    def rnormal(self, mu, sigma, xmin=float('-inf'), xmax=float('inf'), \
                                 pmin=0.0, pmax=1.0):
        """
        Generator of normally distributed random variates using an inverse 
        method, claimed to give a maximum relative error less than 2.6e-9 
        (cf. statlib.invcdf.inormal for details). 
        """

        self._checkminmax(xmin, xmax, pmin, pmax, 'rnormal')

        pmn = pmin
        pmx = pmax
        if xmin > float('-inf'): pmn = max(pmin, cnormal(mu, sigma, xmin))
        if xmax < float('inf'):  pmx = min(pmax, cnormal(mu, sigma, xmax))

        p  =  pmn + (pmx-pmn)*self.runif01()
        x  =  inormal(p, mu, sigma)

        return x

    # end of rnormal

# ------------------------------------------------------------------------------

    def rwiener(self, tau, xmin=float('-inf'), xmax=float('inf'), \
                           pmin=0.0, pmax=1.0):
        """
        Generates random numbers corrsponding to a Wiener process (the 
        intergral of white noise (Langevin's function)), inverse variant. 
        The Wiener process is W(t+tau) - W (t) = N(0, sqrt(tau)) 
        where tau is the time increment and N(0, sqrt(tau)) is a  
        normally distributed random variable having zero mean and 
        sqrt(tau) as its standard deviation.
        
        This method returns W(t+tau) - W(t) given tau and allows 
        tau to be negative.
        """

        self._checkminmax(xmin, xmax, pmin, pmax, 'rwiener')

        mu    = 0.0
        sigma = sqrt(abs(tau))   # abs(tau) is used

        pmn = pmin
        pmx = pmax
        if xmin > float('-inf'): pmn = max(pmin, cnormal(mu, sigma, xmin))
        if xmax < float('inf'):  pmx = min(pmax, cnormal(mu, sigma, xmax))

        p  =  pmn + (pmx-pmn)*self.runif01()
        w  =  inormal(p, mu, sigma)

        return w

    # end of rwiener

# ------------------------------------------------------------------------------

    def rlognormal(self, mulg, sigmalg, xmax=float('inf'), pmax=1.0):
        """
        Generator of lognormally distributed random variates, inverse variant.
        The log10-converted form is assumed for mulg and sigmalg: 
        mulg is the mean of the log10 (and the log10 of the median) of 
        the random variate, NOT the log10 of the mean of the non-logged 
        variate!, and sigmalg is the standard deviation of the log10 of 
        the random variate, NOT the log10 of the standard deviation of 
        the non-logged variate!!
        
        sigmalg > 0.0
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rlognormal!"
        # sigmalg is checked in rnormal
        self._checkpmax(pmax, 'rlognormal')

        if xmax < float('inf'): \
                   pmx = min(pmax, pow(10.0, cnormal(mulg, sigmalg, xmax)))
        pmn = 0.0
        x    = pow(10.0, self.rnormal(mulg, sigmalg, pmn, pmx))

        return x

    # end of rlognormal

# ------------------------------------------------------------------------------

    def rfoldednormal(self, muunfold, sigmaunfold, xmax=float('inf'), pmax=1.0):
        """
        The distribution of a random variable that is the absolute value
        of a variate drawn from the normal distribution (i. e. the distribution 
        of a variate that is the absolute value of a normal variate, the latter 
        having muunfold as its mean and sigmaunfold as its standard deviation). 
        
        sigmaunfold >= 0.0
        """

        assert xmax >= 0.0, \
                       "xmax must be a non-negative float in rfoldednormal!"
        self._checkpmax(pmax, 'rfoldednormal')

        pmx = pmax
        if xmax < float('inf'): \
                 pmx = min(pmax, cfoldednormal(muunfold, sigmaunfold, xmax))

        p  =  pmx * self.runif01()
        x  =  ifoldednormal(p, muunfold, sigmaunfold)

        return x

    # end of rfoldednormal

# ------------------------------------------------------------------------------

    def rstable_sym(self, alpha, location, scale, \
                                 xmin=float('-inf'), xmax=float('inf'), \
                                 pmin=0.0, pmax=1.0):
        """
        The SYMMETRICAL stable distribution where alpha is the tail exponent. 
        For numerical reasons alpha is restricted to [0.25, 0.9] and 
        [1.125, 1.9] - but alpha = 1.0 (the Cauchy) and alpha = 2.0 (scaled 
        normal) are also allowed!

        Numerics are somewhat crude but the fractional error is mostly < 0.001 -
        sometimes much less - and the absolute error is almost always < 0.001 - 
        sometimes much less... 

        NB This generator is slow - particularly for small alpha  !!!!!
        """

        self._checkminmax(xmin, xmax, pmin, pmax, 'rstable_sym')

        pmn = pmin
        pmx = pmax
        if xmin > float('-inf'):\
                  pmn = max(pmin, cstable_sym(alpha, location, scale, xmin))
        if xmax < float('inf'): \
                  pmx = min(pmax, cstable_sym(alpha, location, scale, xmax))

        p  =  pmn + (pmx-pmn)*self.runif01()
        x  =  istable_sym(p, alpha, location, scale)

        return x

    # end of rstable_sym

# ------------------------------------------------------------------------------

    def rlevy(self, scale, xmax=float('inf'), pmax=1.0): 
        """
        The Levy distribution: f = sqrt(s/2pi) * (1/x)**(3/2) * exp(-s/2x)
                               F = erfc(sqrt(s/2x)); x >= 0
        (stable distribution with alpha = 1/2, and beta = 1, aka the Cournot 
        distribution or the right-skewed Levy).
        """

        assert xmax >= 0.0, "xmax must be a non-negative float in rlevy!"
        self._checkpmax(pmax, 'rlevy')

        pmx = pmax
        if xmax < float('inf'): pmx = min(pmax, clevy(scale, xmax))

        p  =  pmx * self.runif01()
        x  =  ilevy(p, scale)

        return x

    # end of rlevy

# ------------------------------------------------------------------------------

    def ruser_defined(self, ifunc, *args, pmin=0.0, pmax=1.0):
        """
        Random deviate generation based on a user-defined inverse cdf placed in 
        a function ('ifunc') with arguments p in [0.0, 1.0] and sequence *args.
        
        NB 1  *args is optional in that it is OK for ifunc to have no arguments.

        NB 2  THE VALUES OF pmin AND pmax MUST - AND CAN ONLY - BE PASSED 
        BY KEYWORD: pmin=??, pmax=?? MUST ALWAYS BE WRITTEN OUT EXPLICITLY 
        UNLESS DEFAULTS ARE USED ('pmin' may be entered without explicit 
        keywording of pmax if the default is accepted for pmax).
        """

        assert  0.0 <= pmin <= pmax, \
                     "pmin must be in [0.0, pmax] in ruser_defined!"
        assert pmin <= pmax <= 1.0,\
                     "pmax must be in [pmin, 1.0] in ruser_defined!"

        p  =  pmin + (pmax-pmin)*self.runif01()
        x  =  ifunc(p, *args)

        return x

    # end of ruser_defined

# ------------------------------------------------------------------------------

# end of InverseRandomStream

# ------------------------------------------------------------------------------
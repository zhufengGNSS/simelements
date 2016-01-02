# statlib/stats.py
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
MODULE WITH A NUMBER OF BASIC STATISTICAL FUNCTIONS. 
                        #  | Parameter used in bootstrapping procedures so that
_BOOTCARDINAL = 2**35   # < no. of samples = int(sqrt(_BOOTCARDINAL/length)) + 1
                        #  | where "length" is the length of the input vector(s)           
"""
# ------------------------------------------------------------------------------

from copy  import deepcopy
from math  import sqrt, exp, fsum

from genrandstrm       import GeneralRandomStream
from statlib.invcdf    import icauchy, inormal
from misclib.matrix    import Matrix
from misclib.numbers   import kept_within, is_posinteger
from misclib.errwarn   import warn
from misclib.mathconst import PIHALF
from machdep.machnum   import TINY

                        #  | Parameter used in bootstrapping procedures so that
_BOOTCARDINAL = 2**35   # < no. of samples = int(sqrt(_BOOTCARDINAL/length)) + 1
                        #  | where "length" is the length of the input vector(s)

# ------------------------------------------------------------------------------

def aritmean(vector, confdeg=0.90):
    """
    aritmean computes the arithmetic mean of a set of data placed in a list 
    and returns the mean, the standard error of the estimate and - if 'confdeg' 
    > 0.0 - the symmetric confidence interval for confidence degree 'confdeg' 
    around the estimate using Student's 't' distribution. 
    """

    assert 0.0 <= confdeg and confdeg <= 1.0, \
          "Confidence degree must be in [0.0, 1.0] in aritmean!"

    length = len(vector)

    # Compute the arithmetic mean:
    amean = fsum(vector) / length  # Will be a float...

    serror = None
    if length > 1:    # Compute the standard deviation and the standard error
        sums   = fsum((vk-amean)**2 for vk in vector)
        serror = sqrt(sums/((length-1)*length))  # Will be a float...

    confint = None
    if confdeg > 0.0 and length > 1:   # Compute the confidence interval
        stud    =  _studQ(length-1, confdeg)
        confint =  stud * serror

    return amean, serror, confint

# end of aritmean

# ------------------------------------------------------------------------------

def boot_aritmean(vector, confdeg=0.90, nsamples=None, printns=False):
    """
    boot_aritmean computes symmetric confidence limits around an arithmetic 
    mean estimate for confidence degree 'confdeg' using bootstrapping. Lower 
    and upper limit are returned. When the number of bootstrap samples 
    nsamples=None, a default number int(sqrt(_BOOTCARDINAL/length)) + 1 will 
    be used. For printns=True the number of bootstrap samples used are printed 
    to stdout.
    
    For more general bootstrap sampling needs the methods 'bootvector' and 
    'bootindexvector' of the RandomStructure class may be used as a basis.
    """

    assert 0.0 <= confdeg and confdeg <= 1.0, \
          "Confidence degree must be in [0.0, 1.0] in boot_aritmean!"

    length  = len(vector)

    if nsamples == None: nsamples = int(sqrt(_BOOTCARDINAL/length)) + 1

    if nsamples < 101:
        nsamples = 101
        wtxt1 = "Number of samples in the bootstrap in boot_aritmean should\n"
        wtxt2 = "be at least 101 (101 samples will be used)"
        warn(wtxt1+wtxt2)

    ratio   = []
    rstream = GeneralRandomStream()
    for k in range(0, nsamples):
        bootv = [vector[rstream.runif_int0N(length)] for v in vector]
        rate  = fsum(bootv) / length   # Will be a float...
        ratio.append(rate)

    median, conflow, confupp  =  mednfrac(ratio, confdeg)

    if length <= 1: confupp = conflow = None

    if printns and nsamples != None:
        print("(number of bootstrap samples used in boot_aritmean = " +\
                                                    str(nsamples) + ")")

    return conflow, confupp

# end of boot_aritmean

# ------------------------------------------------------------------------------

def mednfrac(vector, centwin=0.90):
    """
    mednfrac computes the median and two fractiles for the numbers in a list, 
    given a symmetric fractile interval (e. g. 0.05 and 0.95 for centwin=0.90).
    The algorithm is based on linear interpolation. 
    """

    assert 0.0 <= centwin and centwin <= 1.0, \
          "Fractile interval must be in [0.0, 1.0] in mednfrac!"

    length = len(vector)

    if length <= 1:
        fracupp = fraclow = median = None

    else:
        auxvect = list(vector)  # Don't sort the original - may cause problems!
        auxvect.sort()

        # First the median (integer division is used):
        if int(0.5*length) == 0.5*length:
            median  = 0.5 * (auxvect[length//2 - 1] + auxvect[length//2])
        else:
            median = auxvect[length//2]

        # Then the fractiles:
        nm1     =  length - 1
        low     =  0.5 * (1.0-centwin) * nm1
        upp     =  0.5 * (1.0+centwin) * nm1
        ilow    =  int(low)
        iupp    =  min(int(upp), nm1)
        ilp1    =  ilow + 1
        iup1    =  iupp + 1
        qlow    =  low - float(ilow)
        qupp    =  upp - float(iupp)
        difflow =  auxvect[ilp1] - auxvect[ilow]
        diffupp =  auxvect[iup1] - auxvect[iupp]
        fraclow =  auxvect[ilow] + qlow*difflow
        fracupp =  auxvect[iupp] + qupp*diffupp
# In case existing data points are preferred instead of points from interpol'n..
        #fraclow = auxvect[ilow]
        #fracupp = auxvect[iup1]


    return median, fraclow, fracupp

# end of mednfrac

# ------------------------------------------------------------------------------

def boot_mednfrac(vector, centwin=0.90, confdeg=0.90, nsamples=None, \
                                            printns=False):
    """
    boot_mednfrac computes symmetric confidence limits for a median-fractile 
    estimate for confidence degree 'confdeg' using mednfrac and bootstrapping. 
    Lower and upper limit of 1) the median, 2) the lower fractile and 3) the 
    upper fractile for confidence degree 'confdeg'. When the number of bootstrap 
    samples nsamples=None, a default number int(sqrt(_BOOTCARDINAL/length)) + 1
    will be used. For printns=True the number of bootstrap samples used are 
    printed to stdout.
        
    For more general bootstrap sampling needs the methods 'bootvector' and 
    'bootindexvector' of the RandomStructure class may be used as a basis.
    """

    assert 0.0 <= centwin and centwin <= 1.0, \
          "Fractile interval must be in [0.0, 1.0] in boot_mednfrac!"
    assert 0.0 <= confdeg and confdeg <= 1.0, \
          "Confidence degree must be in [0.0, 1.0] in boot_mednfrac!"

    length  = len(vector)

    if nsamples == None: nsamples = int(sqrt(_BOOTCARDINAL/length)) + 1

    if nsamples < 101:
        nsamples = 101
        wtxt1 = "Number of samples in the bootstrap in boot_mednfrac should\n"
        wtxt2 = "be at least 101 (101 samples will be used)"
        warn(wtxt1+wtxt2)


    median = []
    lower  = []
    upper  = []

    rstream = GeneralRandomStream()
    for k in range(0, nsamples):
        bootv = [vector[rstream.runif_int0N(length)] for v in vector]
        med, low, upp = mednfrac(bootv, centwin)
        median.append(med)
        lower.append(low)
        upper.append(upp)

    med, lowmed, uppmed  =  mednfrac(median, confdeg)
    med, lowlow, upplow  =  mednfrac(lower,  confdeg)
    med, lowupp, uppupp  =  mednfrac(upper,  confdeg)

    if printns and nsamples != None:
        print("(number of bootstrap samples used in boot_mednfrac = " + \
                                                     str(nsamples) + ")")

    return lowmed, uppmed, lowlow, upplow, lowupp, uppupp

# end of boot_mednfrac

# ------------------------------------------------------------------------------

def variance(vector, confdeg=0.90):
    """
    variance computes the variance, the standard deviation and the standard 
    error of the standard deviation estimate plus - provided that confdeg > 0.0 
    - the symmetrical confidence limits computed using Student's "t" 
    distribution for the values in the input vector (list/'d' array/tuple).
    """

    length = len(vector)
    assert length > 1, \
                 "Must be more than one element in input list to variance!"

    assert 0.0 <= confdeg and confdeg <= 1.0, \
          "Confidence degree must be in [0.0, 1.0] in variance!"

    # Compute the arithmetic mean:
    amean = fsum(vector) / length   # Will be a float...

    # Compute the variance, the standard deviation, and the 
    # standard error of the standard deviation:
    sums = fsum((vk-amean)**2 for vk in vector)

    flengthm1 =  float(length-1)
    var       =  sums / flengthm1
    sigma     =  sqrt(var)
    se        =  var / sqrt(0.5*flengthm1)

    confint = None
    if confdeg > 0.0 and length > 1:   # Compute the confidence interval
        stud    =  _studQ(length-1, confdeg)
        confint =  stud * se

    return var, sigma, se, confint

# end of variance

# ------------------------------------------------------------------------------

def statsummary(vector, centwin=0.90, confdeg=0.90):
    """
    statsummary compiles a number of essential statistics for a given input 
    vector (list/'d' array/tuple) of data: the median, lower and upper 
    fractile (using mednfrac), arithmetic mean, symmetrical confidence 
    interval around the mean (using aritmean), as well as the minimum and 
    the maximum are returned.
    """

    assert 0.0 <= centwin and centwin <= 1.0, \
          "Fractile interval must be in [0.0, 1.0] in statsummary!"
    assert 0.0 <= confdeg and confdeg <= 1.0, \
          "Confidence degree must be in [0.0, 1.0] in statsummary!"

    median, fraclow, fracupp  =  mednfrac(vector, centwin)
    amean, se, confint        =  aritmean(vector, confdeg)
    #min = vector[0]  # Not sorted?
    #max = vector[-1]

    return median, fraclow, fracupp, amean, confint, min(vector), max(vector)

# end of statsummary

# ------------------------------------------------------------------------------

def sercorr(vector, nlag=1):
    """
    sercorr returns the serial correlation coefficient for the sequence of 
    values in the input vector (list/'d' array/tuple) nlag steps apart.
    """

    assert is_posinteger(nlag), \
                      "Lag ('nlag') must be a positive integer in sercorr!"
    nlagp2 = nlag + 2
    assert len(vector) >= nlagp2, "There must be at least " + str(nlagp2) + \
                                  " elements in input sequence to sercorr!"

    vector1 = list(vector)
    vector2 = list(vector)
    for k in range(0, nlag):
        del vector1[-1]
        del vector2[0]

    amean1, amean2, var1, var2, cov12, rho12  =  covar(vector1, vector2)

    return rho12

# end of sercorr

# ------------------------------------------------------------------------------

def covar(vector1, vector2):
    """
    covar computes the covariance between the two input vectors (lists/
    'd' arrays/tuples) and returns the means and the variances of the two 
    input sequences as well as the covariance and the correlation coefficient. 
    """

    length = len(vector1)
    assert length > 1, "Must be more than one element in input lists to covar!"
    assert length == len(vector2), \
                           "Input vectors must have equal lengths in covar!"

    # Compute the arithmetic means:
    sum1 = fsum(vector1)
    sum2 = fsum(vector2)
    amean1  = sum1 / length   # Will be a float...
    amean2  = sum2 / length   # Will also be a float...

    # Compute variances, covariance and correlation coefficient:
    sums11 = 0.0
    sums22 = 0.0
    sums12 = 0.0
    for k in range(0, length):
        diff1 = vector1[k] - amean1
        diff2 = vector2[k] - amean2
        #sums11 += vector1[k]*diff1
        #sums22 += vector2[k]*diff2
        #sums12 += vector1[k]*diff2
        sums11 += diff1**2
        sums22 += diff2**2
        sums12 += diff1*diff2

    flengthm1 =  float(length-1)
    var1  =  sums11 / flengthm1
    var2  =  sums22 / flengthm1
    cov12 =  sums12 / flengthm1

    if sums11 == 0.0 or sums22 == 0.0: rho12 =  1.0
    else:                              rho12 =  sums12 / sqrt(sums11*sums22)
    rho12 = kept_within(-1.0, rho12, 1.0)

    return amean1, amean2, var1, var2, cov12, rho12

# end of covar

# ------------------------------------------------------------------------------

def corrmatrix(inputmatrix):
    """
    Computes the correlation matrix of the input matrix. Each row is assumed 
    to contain the vector for one parameter. 
    """

    ndim = len(inputmatrix)  # = the number of rows/parameters
    
    # First create unity output matrix
    corrmatrix = Matrix()
    corrmatrix.unity(ndim)

    # Then fill it with correlation coefficients
    for k in range(0, ndim):
        kp1 = k + 1
        for j in range(0, kp1):
            if j != k:
                #amk,amj,vk,vj,covkj, rhokj  = covar(inputmatrix[k], \
                #                                    inputmatrix[j])
                #corrmatrix[k][j] = corrmatrix[j][k] = rhokj
                corrmatrix[k][j] = corrmatrix[j][k] = \
                           covar(inputmatrix[k], inputmatrix[j])[5]  # = rhokj

    return corrmatrix

# end of corrmatrix

# ------------------------------------------------------------------------------

def boot_ratio(vnumer, vdenom, confdeg=0.90, nsamples=None, printns=False):
    """
    boot_ratio computes symmetric confidence limits for a ratio of a sum to 
    another sum for confidence degree 'confdeg' using bootstrapping. The 
    sums are defined by the two input sequences (lists/'d' arrays/tuples).
    Lower and upper limit are returned. When the number of bootstrap samples 
    nsamples=None, a default number int(sqrt(_BOOTCARDINAL/length)) + 1 
    will be used. For printns=True the number of bootstrap samples used 
    are printed to stdout.
        
    For more general bootstrap sampling needs the methods 'bootvector' and 
    'bootindexvector' of the RandomStructure class may be used as a basis.
    """

    length  = len(vnumer)
    assert len(vdenom) == length, \
           "lengths of numerator and denominator must be equal in boot_ratio!"

    if nsamples == None: nsamples = int(sqrt(_BOOTCARDINAL/length)) + 1

    if nsamples < 101:
        nsamples = 101
        wtxt1 = "Number of samples in the bootstrap in boot_ratio should be\n"
        wtxt2 = "at least 101 (101 samples will be used)"
        warn(wtxt1+wtxt2)

    ratio   = []
    rstream = GeneralRandomStream()
    for k in range(nsamples):
        sumd    = 0.0
        sumn    = 0.0
        indexv  = [rstream.runif_int0N(length) for n in vnumer]
        for i in indexv:
           sumn += vnumer[i]
           sumd += vdenom[i]

        rate = sumn / float(sumd)   # Safety first - input could be ranks...
        ratio.append(rate)

    median, conflow, confupp  =  mednfrac(ratio, confdeg)

    if length <= 1: confupp = conflow = None

    if printns and nsamples != None:
        print("(number of bootstrap samples used in boot_ratio = " +\
                                                 str(nsamples) + ")")

    return conflow, confupp

# end of boot_ratio

# ------------------------------------------------------------------------------

def extract_ranks(vector, bottominteger=0):
    """
    Extract the ranks from a vector/list of numbers. The bottom integer for the
    ranking may be either 0 or 1 (default is 0). A list of ranks is returned. 
    """

    errortext = "Bottom rank must be set to 0 or 1 in extract_ranks!"
    assert bottominteger == 0 or bottominteger == 1, errortext

    length  = len(vector)
    auxvec  = deepcopy(vector)
    mtable  = {}

    auxvec  = list(auxvec)
    auxvec.sort()
    for k in range(0, length):
        mtable[auxvec[k]]  =  k + bottominteger

    ranks = []
    for k in range(0, length):
        ranks.append(mtable[vector[k]])

    return ranks

# end of extract_ranks

# ------------------------------------------------------------------------------

def normalscores(ivector, scheme="van_der_Waerden", bottominteger=0):
    """
    Creates a list of Blom, Tukey or van der Waerden normal scores given a list 
    of N integers in [0, N) or [1, N] depending on whether the bottom integer 
    for the ranking is set to 0 or 1 (0 is default). scheme is either 'Blom', 
    'Tukey' or 'van_der_Waerden'. 
    """

    assert bottominteger == 0 or bottominteger == 1, \
                       "Bottom rank must be set to 0 or 1 in normalscores!"

    if   scheme == 'Blom':
        constnumer = - 0.375
        constdenom =   0.250
    elif scheme == 'Tukey':
        constnumer = - 1.0/3.0
        constdenom = - constnumer
    elif scheme == 'van_der_Waerden':
        constnumer = 0.0
        constdenom = 1.0

    length      = len(ivector)
    scorevector = []
    for integer in ivector:
        n = integer + 1 - bottominteger
        scorevector.append(inormal((n-constnumer)/(length+constdenom)))

    return scorevector

# end of normalscores

# ------------------------------------------------------------------------------

def _studQ(ndf, confdeg=0.90):
    """
    Quantiles for Student's "t" distribution (one number is returned)
    ---------
    ndf       the number of degrees of freedom
    confdeg  confidence level
    --------
    Accuracy in decimal digits of about:
      5 if ndf >= 8., machine precision if ndf = 1. or 2., 3 otherwise
    ----------
    Reference: G.W. Hill, Communications of the ACM, Vol. 13, no. 10, Oct. 1970
    """

    # Error handling
    assert 0.0 <= confdeg and confdeg <= 1.0, \
          "Confidence degree must be in [0.0, 1.0] in _studQ!"
    assert ndf > 0.0, "Number of degrees of freedom must be positive in _studQ!"


    phi = 1.0 - confdeg

    # For ndf = 1 we can use the Cauchy distribution
    if ndf == 1.0:
        prob = 1.0 - 0.5*phi
        stud = icauchy(prob)

    # Finns exakt metod aven for ndf = 2
    elif ndf == 2.0:
        if phi <= TINY: phi = TINY
        stud = sqrt(2.0 / (phi*(2.0-phi)) - 2.0)

    # Check to see if we're not too far out in the tails
    elif phi < TINY:
        t = 1.0 / TINY
        stud = t

    # General case
    else:
        a = 1.0 / (ndf-0.5)
        b = 48.0 / a**2
        c = ((20700.0*a/b - 98.0) * a - 16.0) * a + 96.36
        d = ((94.5/(b + c) - 3.0) / b + 1.0) * sqrt(a*PIHALF) * ndf
        x = d * phi
        y = x ** (2.0 / ndf)

        if y > 0.05 + a:  #     Asymptotic inverse expansion about normal
            x = inormal(0.5*phi)
            y = x**2
            if ndf < 5.0: c = c + 0.3 * (ndf-4.5) * (x+0.6)
            c = (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b + c
            y = (((((0.4*y + 6.3) * y + 36.0) * y + 94.5) / c - y  - 3.0) \
                                                              / b + 1.0) * x
            y = a * y**2
            if y > 0.002: y = exp(y) - 1.0
            else:         y = 0.5 * y**2 + y
        else:

            y = ((1.0 / (((ndf + 6.0) / (ndf * y) - 0.089 * d -  \
                0.822) * (ndf + 2.0) * 30) + 0.5 / (ndf + 4.0))  \
                * y - 1.0) * (ndf + 1.0) / (ndf + 2.0) + 1.0 / y

        stud = sqrt(ndf*y)

    return stud

# end of _studQ

# ------------------------------------------------------------------------------

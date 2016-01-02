# randstruct.py
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

from copy  import deepcopy
from array import array

from misclib.matrix    import Matrix
from numlib.matrixops  import transposed, inverted, flattened
from numlib.matrixops  import ludcmp_chol, mxdisplay
from misclib.iterables import reorder
from misclib.numbers   import is_posinteger, kept_within
from statlib.stats     import extract_ranks, normalscores, corrmatrix
from misclib.errwarn   import warn

# ------------------------------------------------------------------------------

class RandomStructure:
    """
    Class to create randomized data structures other than single random 
    variates from various distributions. It relies heavily on the the 
    random stream generator classes but do not inherit from them. It 
    needs at least one random stream as an input. This is the way to 
    trigger the RandomStructure class:
        from genrandstrm import GeneralRandomStream
        from randstruct  import RandomStructure
        rstream = GeneralRandomStream()
        rstruct = RandomStructure(rstream)
    or, if two streams are needed:
        rstream1 = GeneralRandomStream()
        rstream2 = GeneralRandomStream(different_nseed)
        rstruct  = RandomStructure(rstream1, rstream2)

    A couple of the methods of this class may be used in a way that creates 
    streams of uniformly distributed random numbers in [0.0, 1.0] that may 
    be fed into an instance object punched out with InverseRandomStream - 
    the latter may then use this stream instead of repeated sampling via 
    the basic random number generator, cf. the docstring documentation of 
    InverseRandomStream!
    """
# ------------------------------------------------------------------------------

    def __init__(self, rstream, rstream2=None):
        """
        Inputs are GeneralRandomStream instance objects. Makes it possible 
        to use two streams of random variates. 
        """

        self.rstream = rstream
        if rstream2: self.rstream2 = rstream2

    # end of __init__

# ------------------------------------------------------------------------------

    def bootvector(self, vector):
        """
        Returns a full bootstrap sample from the input vector (a list or tuple).
        """

        rstream = self.rstream

        length  = len(vector)
        bootv   = [vector[rstream.runif_int0N(length)] for v in vector]

        return bootv

    # end of bootvector

# ------------------------------------------------------------------------------

    def bootindexvector(self, vector):
        """
        Returns a list of indices of a full bootstrap sample 
        from the input vector (a list or tuple).
        """

        rstream = self.rstream

        length  = len(vector)
        indexv  = [rstream.runif_int0N(length) for v in vector]

        return indexv

    # end of bootindexvector

# ------------------------------------------------------------------------------

    def sercorrnormvector(self, n, rho, mu=0.0, sigma=1.0):
        """
        Generates a list of serially correlated normal random 
        variates. Requires that the rstream object is punched 
        out from the GeneralRandomStream class!
        """

        # Input check -----
        assert is_posinteger(n)
        # sigma and rho are checked in rstream.rnormal and rstream.rcorr_normal
        # -----------------

        vector  = []
        rstream = self.rstream

        x = rstream.rnormal(mu, sigma)
        vector.append(x)
        for k in range(1, n):
            x = rstream.rcorr_normal(rho, mu, sigma, mu, sigma, x)
            vector.append(x)

        return vector

    # end of sercorrnormvector

# ------------------------------------------------------------------------------

    def multinormalvector(self, mumx, covarlomx, flat=False):
        """
        Generates a sample from a multinomial normal distribution.
        mumx contains the means of each variate and covarlomx is assumed 
        to be a matrix belonging to the misclib.Matrix class. mumx must 
        be a column vector and covarlomx the lower triangular matrix from 
        a previous Cholesky decomposition of the covariance matrix. The 
        output vector is also on Matrix form, it is a column vector, 
        unless the user wants it flattened to a list. 
        """

        # The necessary dimensional tests are made on the Matrix objects 
        # when operations are attempted

        # Create a matrix of totally random normal variates 
        # with mu=0.0 and sigma=1.0
        
        # First a new Matrix column vector for the output:
        xmx  = deepcopy(mumx)
        ndim = len(xmx)
        
        # Then fill it with standard normal random numbers:
        rstream = self.rstream
        for k in range(0, ndim):
            xmx[k][0] = rstream.rnormal(0.0, 1.0)
        
        # Then apply the means and the covariances
        # (Matrix multiplication and addition):
        xmx = covarlomx*xmx + mumx


        # Finally return the column vector or list containing 
        # the multinormal variates:
        if flat: return flattened(xmx)
        else:    return xmx

    # end of multinormalvector

# ------------------------------------------------------------------------------

    def antithet_sample(self, nparams):
        """
        Generates a matrix having two rows, the first row being a list of 
        uniformly distributed random numbers p in [0.0, 1.0], each row 
        containing nparams elements. The second row contains the corresponding 
        antithetic sample with the complements 1-p. 
        """

        rstream = self.rstream

        antimatrix = Matrix()  # antimatrix belongs to the Matrix class
        for k in range(0, nparams):
            pvector = array('d', [])
            p1  =  rstream.runif01()
            pvector.append(p1)
            dum =  rstream.runif01()  # For synchronization only - never used
            p2  =  1.0 - p1
            p2 = kept_within(0.0, p2, 1.0) # Probabilities must be in [0.0, 1.0]
            pvector.append(p2)
            antimatrix.append(pvector)

        # Matrix must be transposed in order for each sample to occupy one row.
        # Sample vector k is in antimatrix[k], where k is 0 or 1
        antimatrix.transpose()

        return antimatrix

    # end of antithet_sample

# ------------------------------------------------------------------------------

    def lhs_sample(self, nparams, nintervals, rcorrmatrix=None, checklevel=0):

        """
        Generates a full Latin Hypercube Sample of uniformly distributed 
        random variates in [0.0, 1.0] placed in a matrix with one realization 
        in each row. A target rank correlation matrix can be given (must have 
        the dimension nsamples*nsamples).
        
        checklevel may be 0, 1 or 2 and is used to control trace printout. 
        0 produces no trace output, whereas 2 produces the most.

        NB. IN ORDER FOR LATIN HYPERCUBE SAMPLING TO BE MEANINGFUL THE OUTPUT 
        STREAM OF RANDOM VARIATES MUST BE HANDLED BY INVERSE METHODS !!!! 

        Latin Hypercube Sampling was first described by McKay, Conover & 
        Beckman in a Technometrics article 1979. The use of the LHS technique 
        to introduce rank correlations was first described by Iman & Conover 
        1982 in an issue of Communications of Statistics.
        """

        # lhs_sample uses the Matrix class to a great extent

        if nparams > nintervals:
            warn("nparams > nintervals in RandomStructure.lhs_sample")

        nsamples     = nintervals   # Just to remember
        rstreaminner = self.rstream
        rstreamouter = self.rstream2

        factor  =  1.0 / float(nintervals)

        tlhsmatrix1 = Matrix()  # tlhsmatrix1 belongs to the Matrix class
        if rcorrmatrix: tscorematrix = Matrix()
        for k in range(0, nparams):
            if rcorrmatrix:
                tnvector, tscorevector = \
                            self.scramble_range(nsamples, rstreamouter, True)
                rowk = array('d', tscorevector)
                tscorematrix.append(rowk)
            else:
                tnvector = self.scramble_range(nsamples, rstreamouter)
            pvector = array('d', [])
            for number in tnvector:
                p  =  factor * (float(number) + rstreaminner.runif01())
                p  =  max(p, 0.0) # Probabilities must be in [0.0, 1.0]
                p  =  min(p, 1.0)
                pvector.append(p)
            tlhsmatrix1.append(pvector)
                
        
        # tlhsmatrix1 (and tscorematrix) are now transposed to run with 
        # one subsample per row to fit with output as well as Iman-Conover 
        # formulation. tlhsmatrix1 and tscorematrix will be used anyway 
        # for some manipulations which are more simple when matrices run 
        # with one variable per row

        lhsmatrix1  = transposed(tlhsmatrix1)
        if rcorrmatrix: scorematrix = transposed(tscorematrix)

        if checklevel == 2:
            print("lhs_sample: Original LHS sample matrix")
            mxdisplay(lhsmatrix1)
            if rcorrmatrix: 
                print("lhs_sample: Target rank correlation matrix")
                mxdisplay(rcorrmatrix)
        if checklevel == 1 or checklevel == 2:
            print("lhs_sample: Rank correlation matrix of")
            print("            original LHS sample matrix")
            trankmatrix1 = Matrix()
            for k in range (0, nparams):
                rowk = array('d', extract_ranks(tlhsmatrix1[k]))
                trankmatrix1.append(rowk)
            mxdisplay(Matrix(corrmatrix(trankmatrix1)))

        if not rcorrmatrix:
            return lhsmatrix1

        else:
            scorecorr = Matrix(corrmatrix(tscorematrix))
            if checklevel == 2:
                print("lhs_sample: Score matrix of original LHS sample matrix")
                mxdisplay(scorematrix)
                print("lhs_sample: Correlation matrix of scores of")
                print("            original LHS sample")
                mxdisplay(scorecorr)

            slower, slowert = ludcmp_chol(scorecorr)
            slowerinverse   = inverted(slower)
            tslowerinverse  = transposed(slowerinverse)
            clower, clowert = ludcmp_chol(rcorrmatrix)
            scoresnostar    = scorematrix*tslowerinverse # Matrix multiplication
            if checklevel == 2:
                print("lhs_sample: Correlation matrix of scoresnostar")
                mxdisplay(corrmatrix(transposed(scoresnostar)))

            scoresstar  = scoresnostar*clowert    # Matrix multiplication
            tscoresstar = transposed(scoresstar)
            trankmatrix = Matrix()
            for k in range (0, nparams):
                trankmatrix.append(extract_ranks(tscoresstar[k]))
            if checklevel == 2:
                print("lhs_sample: scoresstar matrix")
                mxdisplay(scoresstar)
                print("lhs_sample: Correlation matrix of scoresstar")
                mxdisplay(corrmatrix(tscoresstar))
                print("lhs_sample: scoresstar matrix converted to rank")
                mxdisplay(transposed(trankmatrix))
                for k in range(0, nparams):
                    tlhsmatrix1[k] = array('d', sorted(list(tlhsmatrix1[k])))
                print("RandomStructure.lhs_sample: Sorted LHS sample matrix")
                mxdisplay(transposed(tlhsmatrix1))

            tlhsmatrix2 = Matrix()
            for k in range(0, nparams):
                # Sort each row in tlhsmatrix1 and reorder 
                # according to trankmatrix rows
                auxvec = reorder(tlhsmatrix1[k], trankmatrix[k], \
                                                 straighten=True)
                tlhsmatrix2.append(auxvec)
            lhsmatrix2 = transposed(tlhsmatrix2)
            if checklevel == 2:
                print("lhs_sample: Corrected/reordered LHS sample matrix")
                mxdisplay(transposed(tlhsmatrix2))

            if checklevel == 1 or checklevel == 2:
                trankmatrix2 = Matrix()
                auxmatrix2   = tlhsmatrix2
                for k in range (0, nparams):
                    trankmatrix2.append(extract_ranks(auxmatrix2[k]))
                print("lhs_sample: Rank correlation matrix of corrected/")
                print("            /reordered LHS sample matrix")
                mxdisplay(corrmatrix(trankmatrix2))


            return lhsmatrix2

    # end of lhs_sample

# ------------------------------------------------------------------------------

    def scramble_range(self, nrange, stream, scores=False):
        """
        Returns a list containing a random permutation of integers 
        "in range(nrange)", i. e. in [0, nrange-1]. Van der Waerden 
        normal scores are also returned if so requested.
        
        An input random stream is made an argument in order for 
        the method to be able to use any of the two random streams 
        defined for the class. 
        """

        # A list comprehension is used, AND
        # It is assumed that there are no ties in the list of 
        # floats generated by runif01...
        permut = extract_ranks([stream.runif01() for k in range(nrange)])

        if scores:
            scorelist = normalscores(permut)
            return permut, scorelist
        else:
            return permut

    # end of scramble_range

# ------------------------------------------------------------------------------

# end of RandomStructure

# ------------------------------------------------------------------------------

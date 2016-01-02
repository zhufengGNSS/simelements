# numlib/matrixops.py
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
Module used to make computations on matrices. Presumes that inputs are 
objects created with the Matrix class OR have the same list (or 'd' array) 
structure. All matrices returned are objects belonging to the Matrix class 
(if not otherwise explicitly stated). 

Some functions may also appear in a method version belonging to the Matrix 
class. However: all functions in this module maintain the original instance 
object intact. Methods belonging to the Matrix class are often purposely 
designed to change the object matrix in place. 
"""
# ------------------------------------------------------------------------------

from copy  import deepcopy
from array import array
from math  import sqrt, fsum

from numlib.miscnum    import fsign
from misclib.matrix    import Matrix
from misclib.iterables import reorder
from xbuiltins.xmap    import xmap
from misclib.stack     import Stack
from misclib.numbers   import is_posinteger
from machdep.machnum   import SQRTMACHEPS, TINY
from misclib.errwarn   import Error, warn

# ------------------------------------------------------------------------------

def transposed(matrix):
    """
    Transpose matrix. 
    """

    nrows, ncols = sized(matrix, 'transposed')

    newmatrix = ncols*[float('nan')]
    for k in range(0, ncols):  # List comprehension used for the innermost loop
        newmatrix[k] = array('d', [row[k] for row in matrix])
    tmatrix = Matrix(newmatrix)
    del newmatrix

    '''tmatrix = Matrix(matrix)
    tmatrix.transpose() # would be slower'''

    return tmatrix

# end of transposed

# ------------------------------------------------------------------------------

def inverted(matrix, pivoting=True):
    """
    Only square matrices can be inverted! 
    """

    ndim = squaredim(matrix, 'inverted')

    # First: LU-decompose matrix to be inverted
    if pivoting:
        lower, upper, permlist, parity = ludcmp_crout_piv(matrix)
    else:
        lower, upper, permlist, parity = ludcmp_crout(matrix)

    # Create unity matrix
    unitymatrix = Matrix()
    unitymatrix.unity(ndim)

    # Loop over the columns in unity matrix and substitute
    # (uses the fact that rows and columns are the same in a unity matrix)
    columns = Matrix()
    columns.zero(ndim, ndim)
    for k in range(0, ndim):
        columns[k] = lusubs(lower, upper, unitymatrix[k], permlist)
        # preparations below for changing lusubs to handling column vector 
        # instead of list
        #row = Matrix([unitymatrix[k]])
        #column = transpose(row)
        #columns[k] = lusubs(lower, upper, column, permlist)
        #del column

    # Transpose matrix to get inverse
    newmatrix = ndim*[float('nan')]
    for k in range(0, ndim): # List comprehension is used for the innermost loop
        newmatrix[k] = array('d', [row[k] for row in columns])
    imatrix = Matrix(newmatrix)
    del newmatrix

    return imatrix

# end of inverted

# ------------------------------------------------------------------------------

def determinant(matrix, pivoting=True):
    """
    Can only be used for square matrices. 
    """

    ndim = squaredim(matrix, 'determinant')

    # First: LU-decompose matrix
    if pivoting:
        lower, upper, permlist, parity = ludcmp_crout_piv(matrix)
    else:
        lower, upper, permlist, parity = ludcmp_crout(matrix)

    prod  = 1.0
    for k in range(0, ndim): prod = prod * upper[k][k]
    prod *= parity

    return prod

# end of determinant

# ------------------------------------------------------------------------------

def trace(matrix):
    """
    Computes the trace of the input matrix. 
    """

    ndim = squaredim(matrix, 'trace')

    summ = 0.0
    for k in range(0, ndim): summ += matrix[k][k]

    return summ

# end of trace

# ------------------------------------------------------------------------------

def flattened(matrix, stack=False):
    """
    OUTPUT WILL  N O T  BE ON Matrix OBJECT FORMAT!!!

    Places all elements in one single  l i s t,  row-by-row. 
    
    NB. flattened returns a simple Python list (or a Stack 
    if so desired), not a single-row 'Matrix' matrix! 
    """

    nrows, ncols = sized(matrix, 'flattened')

    sequence = []
    for k in range(0, nrows):
        sequence.extend(list(matrix[k]))

    if stack: sequence = Stack(sequence)

    return sequence

# end of flattened

# ------------------------------------------------------------------------------

def check(matrix, caller='caller', elementwise=False):
    """
    Checks the input matrix with respect to consistency (number of 
    elements must be equal in all columns) and returns its dimensions. 
    Also checks that all elements are assigned, if so requested. 
    """

    nrows, ncols = sized(matrix, caller)

    for k in range(1, nrows):
        assert len(matrix[k]) == ncols, \
                 "Structure of matrix is flawed in " + caller + "!"

    if elementwise:
        for k in range(0, nrows):
            for j in range(0, ncols):
                assert matrix[k][j] != float('nan'), \
                        "Element in matrix not assigned in " + caller + "!"

    return nrows, ncols

# end of check

# ------------------------------------------------------------------------------

def sized(matrix, caller='caller'):
    """
    Checks the input matrix with respect to size. 
    """

    try:
        nrows = len(matrix)
        ncols = len(matrix[0])
        return nrows, ncols

    except TypeError:
        errortext  = "What is supposed to be a matrix is not a matrix in "
        errortext += caller + "!"
        raise TypeError(errortext)

    try:
        nrows = len(matrix)
        ncols = len(matrix[0])
        return nrows, ncols

    except IndexError:
        errortext = "Structure of matrix is flawed in " + caller + "!"
        raise IndexError(errortext)


# end of sized

# ------------------------------------------------------------------------------

def mxdisplay(matrix):
    """
    Print the input matrix to the screen. 
    """

    nrows, ncols = sized(matrix, 'mxdisplay')

    print("***** Matrix display - start *****")
    if   nrows == 1:
        print("[  " + str(matrix[0]) + "  ]")
    else:
        for k in range(0, nrows):
            if   k == 0:        print("[  " + str(matrix[k]) + ",")
            elif k == nrows-1:  print("   " + str(matrix[k]) + "  ]")
            else:               print("   " + str(matrix[k]) + ",")
    print("***** Matrix display - end   *****")
    print("")

# end of mxdisplay

# ------------------------------------------------------------------------------

def scaled(matrix, scalar):
    """
    Multiply matrix by scalar. 
    """

    sized(matrix, 'scaled')

    copymx = deepcopy(matrix)

    return Matrix(xmap((lambda x: scalar*x), copymx))

# end of scaled

# ------------------------------------------------------------------------------

def ludcmp_crout(matrix):
    """
    Decomposes/factorizes square input matrix into a lower and an 
    upper matrix using Crout's algorithm WITHOUT pivoting. 
    
    NB. It only works for square matrices!!! 
    """

    ndim = squaredim(matrix, 'ludcmp_crout')

    # Copy object instance to new matrix in order for the original instance 
    # not to be destroyed.
    # Create two new square matrices of the same sized as the input matrix:
    # one unity matrix (to be the lower matrix), one zero matrix (to be 
    # the upper matrix)
    copymx   = deepcopy(matrix)
    lower    = Matrix()
    lower.unity(ndim)
    upper    = Matrix()
    upper.zero(ndim, ndim)
    permlist = list(range(0, ndim))

    # Perform the necessary manipulations:
    for j in range(0, ndim):
        iu = 0
        while iu <= j:
            k    = 0
            summ = 0.0
            while k < iu:
                summ += lower[iu][k]*upper[k][j]
                k   = k + 1
            upper[iu][j] = copymx[iu][j] - summ
            iu = iu + 1
        il = j + 1
        while il < ndim:
            k    = 0
            summ = 0.0
            while k < j:
                summ += lower[il][k]*upper[k][j]
                k = k + 1
            divisor = float(upper[j][j])
            if abs(divisor) < TINY: divisor = fsign(divisor)*TINY
            lower[il][j] = (copymx[il][j]-summ) / divisor
            il = il + 1

    parity = 1.0


    return lower, upper, permlist, parity

# end of ludcmp_crout

# ------------------------------------------------------------------------------

def ludcmp_crout_piv(matrix):
    """
    Decomposes/factorizes square input matrix into a lower 
    and an upper matrix using Crout's algorithm WITH pivoting. 
    
    NB. It only works on square matrices!!! 
    """

    ndim     = squaredim(matrix, 'ludcmp_crout_piv')
    ndm1     = ndim - 1
    vv       = array('d', ndim*[0.0])
    permlist = list(range(0, ndim))
    parity   = 1.0
    imax     = 0

    # Copy to matrix to be processed (maintains the original matrix intact)
    compactlu = deepcopy(matrix)

    for i in range(0, ndim):   # Copy and do some other stuff
        big = 0.0
        for j in range(0, ndim):
            temp = abs(compactlu[i][j])
            if temp > big: big = temp
        assert big > 0.0
        vv[i] = 1.0/big

    # Perform the necessary manipulations:
    for j in range(0, ndim):
        for i in range(0, j):
            sum = compactlu[i][j]
            for k in range(0, i): sum -= compactlu[i][k] * compactlu[k][j]
            compactlu[i][j] = sum
        big = 0.0
        for i in range(j, ndim):
            sum = compactlu[i][j]
            for k in range(0, j): sum -= compactlu[i][k] * compactlu[k][j]
            compactlu[i][j] = sum
            dum = vv[i] * abs(sum)
            if dum > big:
                big  = dum
                imax = i
        if j != imax:
            # Substitute row imax and row j
            imaxdum        = permlist[imax]   # NB in !!!!!!!!!!!!!!!!
            jdum           = permlist[j]      # NB in !!!!!!!!!!!!!!!!
            permlist[j]    = imaxdum          # NB in !!!!!!!!!!!!!!!!
            permlist[imax] = jdum             # NB in !!!!!!!!!!!!!!!!
            for k in range(0, ndim):
                dum                = compactlu[imax][k]
                compactlu[imax][k] = compactlu[j][k]
                compactlu[j][k]    = dum
            parity   = - parity
            vv[imax] = vv[j]
        #permlist[j] = imax   # NB out !!!!!!!!!!!!!!!!!!!!!
        divisor = float(compactlu[j][j])
        if abs(divisor) < TINY: divisor = fsign(divisor)*TINY
        dum = 1.0 / divisor
        if j != ndm1:
            jp1 = j + 1
            for i in range(jp1, ndim): compactlu[i][j] *= dum

    lower = Matrix()
    lower.zero(ndim, ndim)
    upper = Matrix()
    upper.zero(ndim, ndim)

    for i in range(0, ndim):
        for j in range(i, ndim): lower[j][i] = compactlu[j][i]
    for i in range(0, ndim):
        lower[i][i] = 1.0

    for i in range(0, ndim):
        for j in range(i, ndim): upper[i][j] = compactlu[i][j]

    del compactlu


    return lower, upper, permlist, parity

# end of ludcmp_crout_piv

# ------------------------------------------------------------------------------

def ludcmp_chol(matrix, test=False):
    """
    Decomposes/factorizes square, positive definite input matrix into 
    one lower and one upper matrix. The upper matrix is the transpose of 
    the lower matrix. 
    
    NB. It only works on square, symmetric, positive definite matrices!!! 
    """

    if test:
        errortext1 = "Input matrix not positive definite in ludcmp_chol!"
        assert is_posdefinite(matrix), errortext1
        errortext2 = "Input matrix not symmetric in ludcmp_chol!"
        assert is_symmetrical(matrix), errortext2

    ndim = squaredim(matrix, 'ludcmp_chol')

    # Create new square matrix of the same size as the input matrix:
    clower = Matrix()
    clower.zero(ndim, ndim)

    # Perform the necessary manipulations:
    for k in range(0, ndim):
        kp1 = k + 1
        for j in range(0, kp1):
            summ = 0.0
            for i in range(0, j): summ += clower[k][i]*clower[j][i]
            if j == k: clower[k][j] = sqrt(matrix[k][j] - summ)
            else:      clower[k][j] = (matrix[k][j]-summ) / float(clower[j][j])

    clowert = transposed(clower)

    return clower, clowert
        
# end of ludcmp_chol

# ------------------------------------------------------------------------------

def lusubs(lower, upper, bvector, permlist):
    """
    Back substitution for LU decomposition. 
    
    NB. bvector and permlist are just lists, not matrix type vectors. 
    """

    # Check input matrices and vectors/lists for inconsistencies
    ndiml = squaredim(lower, 'lusubs')
    ndimu = squaredim(upper, 'lusubs')

    errortext1 = "lower and upper have different dimensions in lusubs!"
    assert ndimu == ndiml, errortext1
    nb    = len(bvector)
    errortext2 = "inconsistent dimensions in matrices and vector in lusubs!"
    assert nb == ndiml, errortext2

    errortext3 = "inconsistent length of permutation list in lusubs!"
    assert len(permlist) == nb, errortext3
    cvector = reorder(bvector, permlist)


    # First do forward substitution with lower matrix to 
    # create intermediate vector (yvector)
    yvector = array('d', nb*[0.0])
    divisor = float(lower[0][0])
    if abs(divisor) < TINY: divisor = fsign(divisor)*TINY
    yvector[0] = cvector[0] / divisor
    for i in range(1, nb):
        summ = 0.0
        for j in range(0, i):  summ += lower[i][j]*yvector[j]
        divisor = float(lower[i][i])
        if abs(divisor) < TINY: divisor = fsign(divisor)*TINY
        yvector[i] = (cvector[i]-summ) / divisor

    # Then do backward substitution using upper matrix and intermediate 
    # vector to acheive final result
    xvector = array('d', nb*[0.0])
    nbm1 = nb - 1
    divisor = float(upper[nbm1][nbm1])
    if abs(divisor) < TINY: divisor = fsign(divisor)*TINY
    xvector[nbm1] = yvector[nbm1] / divisor
    nbm2 = nbm1 - 1
    for i in range(nbm2, -1, -1):
        summ = 0.0
        ip1  = i + 1
        for j in range(ip1, nb):  summ += upper[i][j]*xvector[j]
        divisor = float(upper[i][i])
        if abs(divisor) < TINY: divisor = fsign(divisor)*TINY
        xvector[i] = (yvector[i]-summ) / divisor


    return xvector

# end of lusubs

# ------------------------------------------------------------------------------

def lusubs_imp(matrix, lower, upper, bvector, permlist, xvector, \
                                     tolf=SQRTMACHEPS, nitermax=4):
    """
    May be used to polish the result from lusubs (some of the necessary 
    checks are made in lusubs). 
    
    tolf is the maximum fractional difference between two consecutive sums
    of absolute values of the output vector, and nitermax is the maximum 
    number of improvements carried out regardless of whether the tolerance 
    is met or not.
    """

    assert tolf >= 0.0, \
            "max fractional tolerance must not be negative in lusubs_imp!"
    
    assert is_posinteger(nitermax), \
            "max number of iterations must be a positive number in lusubs_imp!"

    ndim = len(bvector)

    sumx      = fsum(abs(x) for x in xvector)
    converged = False
    for n in range(0, nitermax):
        resid  = array('d', [])   # will get len = ndim
        for k in range(0, ndim):
            sdp = -bvector[k]
            for j in range(0, ndim):
                sdp += matrix[k][j]*xvector[j]
            resid.append(sdp)
        resid = lusubs(lower, upper, resid, permlist)
        for k in range(0, ndim): xvector[k] = xvector[k] - resid[k]
        sumn = fsum(abs(x) for x in xvector)
        if abs(sumn-sumx) < sumn*tolf:
            converged = True
            break
        sumx = sumn

    wtext = "lusubs_imp did not converge. Try changing tolerance or nitermax"
    if not converged: warn(wtext)

    return xvector

# end of lusubs_imp

# ------------------------------------------------------------------------------

def squaredim(matrix, caller='caller'):
    """
    Test for squareness. 
    """

    nrows, ncols = sized(matrix, 'squaredim')
    if ncols != nrows:
        errortext = "Unsquare matrix in " + caller + "!"
        raise Error(errortext)
    else:
        ndim = ncols

    return ndim

# end of squaredim

# ------------------------------------------------------------------------------

def is_posdefinite(matrix):
    """
    The test for positive definiteness using the determinants of the nested 
    principal minor matrices is taken from Varian; "Microeconomic Analysis". 
    Returns True if input matrix is positive definite, False otherwise. 
    """

    flag = True

    ndim = squaredim(matrix, 'is_posdefinite')

    for k in range(0, ndim):
        '''# Test No. 1 - Necessary condition for positive SEMI-definiteness:
        if matrix[k][k] <= 0.0:
            flag = False
            break'''
        # (Test No. 2 -) Sufficient condition for positive definiteness:
        minor = Matrix()
        kp1 = k + 1
        minor.zero(kp1, kp1)
        for j in range(0, kp1):
            for i in range(0, kp1): minor[j][i] = matrix[j][i]
        x = determinant(minor)
        del minor
        if x <= 0.0:
            flag = False
            break

    return flag

# end of is_posdefinite

# ------------------------------------------------------------------------------

def is_symmetrical(matrix, caller='caller'):
    """
    Test for symmetry. Returns True if matrix is symmetrical, False otherwise. 
    """

    flag = True

    ndim = squaredim(matrix, 'symmetry')

    for k in range(0, ndim):
        kp1 = k + 1
        for j in range(0, kp1):
            if matrix[k][j] != matrix[j][k]:
                flag = False
                break

    return flag

# end of is_symmetrical

# ------------------------------------------------------------------------------
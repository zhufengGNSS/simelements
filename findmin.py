# findmin.py
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
Module containing a function for finding the minimum of an objective function. 
Lists, tuples and 'd' arrays can be entered as inputs ('d' arrays are used 
internally).
"""
# ------------------------------------------------------------------------------

from copy  import deepcopy
from array import array

from misclib.matrix  import Matrix
from misclib.numbers import is_nonneginteger
from machdep.machnum import SQRTTINY, SQRTMACHEPS, MACHEPS
from misclib.errwarn import Error, warn

# ------------------------------------------------------------------------------

def nelder_mead(objfunc, point0, spans, \
                trace=False, tolf=SQRTMACHEPS, tola=SQRTTINY, maxniter=256, \
                rho=1.0, xsi=2.0, gamma=0.5, sigma=0.5):
    """
    The Nelder & Mead downhill simplex method is designed to find the minimum 
    of an objective function that has a multi-dimensional input, (see for 
    instance Lagarias et al. (1998), "Convergence Properties of the Nelder-Mead 
    Simplex in Low Dimensions", SIAM J. Optim., Society for Industrial and 
    Applied Mathematics Vol. 9, No. 1, pp. 112-147 for details). The algorithm 
    is said to first have been presented by Nelder and Mead in Computer Journal,
    Vol. 7, pp. 308-313 (1965).

    The initial simplex must be entered by entering an initial point (an 
    array of coordinates), plus an array of spans for the corresponding 
    point coordinates.

    For trace=True a trace is printed to stdout consisting of the present 
    number of iterations, the present low value of the objective function, 
    the present value of the absolute value of difference between the high and
    the low value of the objective function, and the present list of vertices 
    of the low value of the objective function = the present "best" point.
    
    tolf is the fractional tolerance and tola is the absolute tolerance of 
    the absolute value of difference between the high and the low value of 
    the objective function.

    maxniter is the maximum allowed number of iterations.

    rho, xsi, gamma and sigma are the parameters for reflection, expansion,
    contraction and shrinkage, respectively (cf. the references above).
    """

    # Check the input parameters
    assert is_nonneginteger(maxniter), \
       "max number of iterations must be a non-negative integer in nelder_mead!"
    if tolf < MACHEPS:
        tolf = MACHEPS
        wtext  = "fractional tolerance smaller than machine epsilon is not "
        wtext += "recommended in nelder_mead. Machine epsilon is used instead"
        warn(wtext)
    assert rho > 0.0, "rho must be positive in nelder_mead!"
    assert xsi > 1.0, "xsi must be > 1.0 in nelder_mead!"
    assert xsi > rho, "xsi must be > rho in nelder_mead!"
    assert 0.0 < gamma < 1.0, "gamma must be in (0.0, 1.0) in nelder_mead!"
    assert 0.0 < sigma < 1.0, "sigma be in (0.0, 1.0) in nelder_mead!"
    assert tola >= 0.0, "absolute tolerance must be positive in nelder_mead!"

    # Prepare matrix of vertices
    ndim     = len(point0)
    assert len(spans) == ndim
    vertices = Matrix()
    vertices.append(array('d', list(point0)))
    ndimp1   = ndim + 1
    fndim    = float(ndim)
    for j in range(0, ndim): vertices.append(array('d', list(point0)))
    for j in range(0, ndim): vertices[j+1][j] += spans[j]

    # Prepare a few variants of parameters
    oneprho = 1.0 + rho

    # LOOP!!!!!!!!
    niter = 0
    while True:
        niter += 1
        if niter > maxniter:
            txt1 = "nelder_mead did not converge. Absolute error = "
            txt2 = str(abs(high-low)) + " for " + str(niter-1)
            txt3 = " iterations. Consider new tols or maxniter!"
            raise Error(txt1+txt2+txt3)
        # Compute the objective function values for the vertices
        flist = array('d', [])
        for k in range(0, ndimp1):
            fk = objfunc(vertices[k])
            flist.append(fk)

        # Establish the highest point, the next highest point and the lowest
        low   = flist[0]
        high  = nxhi = low
        ilow  = 0
        ihigh = 0
        for k in range(1, ndimp1):
            fk = flist[k]
            if fk > high:
                nxhi   = high
                high   = fk
                ihigh  = k
            elif fk < low:
                low  = fk
                ilow = k

        if trace: print(niter, low, abs(high-low), list(vertices[ilow]))
        if low < tola: tol = tola
        else:          tol = abs(low)*tolf
        if abs(high-low) < tol: return low, list(vertices[ilow])

        # Reflect the high point
        # First find a new vertix = the centroid of the non-max vertices
        cntr  = array('d', ndim*[float('nan')])
        newr  = array('d', ndim*[float('nan')])
        for j in range(0, ndim):
            xsum = 0.0
            for k in range(0, ndimp1):
                if k != ihigh:
                    xsum += vertices[k][j]
            cntr[j] = xsum/fndim
        # Then move from the centroid in an away-from-max direction
        for j in range(0, ndim):
            newr[j] = oneprho*cntr[j] - rho*vertices[ihigh][j]

        # Check the new vertix
        accepted = False
        phir = objfunc(newr)
        if low <= phir < nxhi:
            # Everything is OK!
            if trace: print("Reflection sufficient")
            vertices[ihigh] = newr
            phi             = phir
            accepted        = True
        elif phir < low:
            # Expand:
            if trace: print("Expansion")
            newe = array('d', ndim*[float('nan')])
            for j in range(0, ndim):
                newe[j] = cntr[j] + xsi*(newr[j]-cntr[j])
            phie = objfunc(newe)
            if phie < phir:
                vertices[ihigh] = newe
                phi             = phie
            else:
                vertices[ihigh] = newr
                phi             = phir
            accepted = True
        elif phir >= nxhi:
            # Contract
            if phir < high:
                # -outside:
                if trace: print("Outside contraction")
                newo = array('d', ndim*[float('nan')])
                for j in range(0, ndim):
                    newo[j] = cntr[j] + gamma*(newr[j]-cntr[j])
                phio = objfunc(newo)
                if phio <= phir:
                    vertices[ihigh] = newo
                    phi             = phio
                    accepted        = True
            else:
                # -inside:
                if trace: print("Inside contraction")
                newi = array('d', ndim*[float('nan')])
                for j in range(0, ndim):
                    newi[j] = cntr[j] - gamma*(cntr[j]-vertices[ihigh][j])
                phii = objfunc(newi)
                if phii <= high:
                    vertices[ihigh] = newi
                    phi             = phii
                    accepted        = True
        if not accepted:
            # Shrink:
            if trace: print("Shrinkage")
            for k in range(0, ndimp1):
                for j in range(j, ndim):
                    vertices[k][j] = vertices[ilow][j] + sigma*(vertices[k][j] -
                                                             vertices[ilow][j])

# end of nelder_mead

# ------------------------------------------------------------------------------
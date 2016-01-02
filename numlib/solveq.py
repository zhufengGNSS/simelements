# numlib/solveq.py
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
Module contains a number of functions for solving equations of one variable. 
"""
# ------------------------------------------------------------------------------

import math, cmath

from numlib.miscnum    import fsign
from misclib.numbers   import is_posinteger, is_nonneginteger
from machdep.machnum   import MACHEPS, FOURMACHEPS, SQRTTINY
from misclib.mathconst import GOLDPHI1   # Approx. 1.6
from misclib.errwarn   import Error, warn

# ------------------------------------------------------------------------------

def z2nddeg_real(a, b, c):
    """
    Finds and returns the real roots of the quadratic equation 
    a*x^2 + b*x + c = 0, are there any.

    Seems to handle the problems in Forsythe-Malcolm-Moler 
    in a  l e s s  reliable manner than does z2nddeg_complex!

    Returns:
    ----------
    x1, x2    The two roots of the second degree equation
              (returns None, None if roots are complex)       
    """

    assert a != 0.0, "x^2 coefficient must not be zero in z2nddeg_real!"

    undersquareroot  =  b**2 - 4.0*a*c

    if undersquareroot >= 0.0:
        x1  =  -b + fsign(b)*math.sqrt(undersquareroot)
        x1  =  0.5 * x1 / a
        x2  =  c / (a*x1)
        return x1, x2
    else:
        return None, None

# end of z2nddeg_real

# ------------------------------------------------------------------------------

def z2nddeg_complex(a, b, c):
    """
    Finds and returns the complex roots of the quadratic equation 
    a*z^2 + b*z + c = 0

    Seems to handle the problems in Forsythe-Malcolm-Moler 
    in a  m o r e  reliable manner than does z2nddeg_real! 

    Returns:
    ----------
    z1, z2    The two roots of the second degree equation
    """

    assert a != 0.0, "x^2 coefficient must not be zero in z2nddeg_complex!"

    # az^2 + bz + c = 0
    # By substituting z = y-t and t = a/2, the equation reduces to 
    #    y^2 + (b-t^2) = 0 
    # which has easy solution
    #    y = +/- sqrt(t^2-b)

    a, b = b / float(a), c / float(a)
    t = 0.5 * a 
    r = t*t - b

    #if r >= 0:
        #y1 = math.sqrt(r)    # real roots
    #else:
        #y1 = cmath.sqrt(r)   # complex roots
    y1 = cmath.sqrt(r)

    y2 = -y1
    z1 = y1 - t
    z2 = y2 - t

    return z1, z2

# end of z2nddeg_complex

# ------------------------------------------------------------------------------

def zbisect(func, x1, x2, caller='caller', tolf=FOURMACHEPS, \
                          tola=SQRTTINY, maxniter=256, bracket=False):
    """
    Solves the equation func(x) = 0 on [x1, x2] using a bisection algorithm. 
    zbisect converges slower than zbrent in most cases, but it might be faster 
    in some cases!
    
    NB. The function always returns a value but a warning is printed to stdout 
    if the iteration procedure has not converged! Cf. comment below regarding 
    convergence!

    Arguments:
    ----------
    func      Function having the proposed root as its argument
    
    x1        Lower search limit (root must be known to be >= x1 unless 
              prior bracketing is used)
    
    x2        Upper search limit (root must be known to be <= x2 unless 
              prior bracketing is used)
    
    tolf      Desired fractional accuracy of root (a combination of fractional 
              and absolute will actually be used: tolf*abs(root) + tola)

    tola      Desired absolute accuracy of root (a combination of fractional 
              and absolute will actually be used: tolf*abs(root) + tola)
              AND
              desired max absolute difference of func(root) from zero

    maxniter  Maximum number of iterations

    bracket   If True, x1 and x2 are used in an initial bracketing before 
              solving 

    Returns:
    ---------
    Final value of root
    
    This algorithm needs on the average log2((b-a)/tol) function evaluations to 
    reach convergence. For instance: b-a = 1.0 and tol = 1.8e-12 will on the 
    average provide convergence in about 40 iterations. Bisection is "dead 
    certain" and will always converge if there is a root. It is likely to pass 
    the tolerances with no extra margin. If there is no root, it will converge 
    to a singularity if there is one...
    """

    if tolf < MACHEPS:
        tolf  = MACHEPS
        wtxt1 = "Fractional tolerance less than machine epsilon is not a "
        wtxt2 = "good idea in zbisect. Machine epsilon will be used instead!"
        warn(wtxt1+wtxt2)

    if tola < 0.0:
        tola  = 0.0
        wtxt1 = "Negative absolute tolerance is not a good idea "
        wtxt2 = "in zbisect. 0.0 (zero) will be used instead!"
        warn(wtxt1+wtxt2)

    assert is_posinteger(maxniter), \
          "Maximum number of iterations must be a positive integer in zbisect!"

    if bracket: x1, x2 = bracketzero(func, x1, x2, caller, maxniter)

    assert x2 > x1, \
                  "Bounds must be given with the lower bound first in zbisect!"


    fmid = func(x2)
    if fmid == 0.0: return x2
    f    = func(x1)
    if f   ==  0.0: return x1
    if fsign(fmid) == fsign(f):
        x1, x2 = bracketzero(func, x1, x2, caller, maxniter)
        wtxt1 = "Starting points must be on opposite sides of the root in "
        wtxt2 = "zbisect. Bracketing will be used to find an appropriate span!"
        warn(wtxt1+wtxt2)

    if f < 0.0:
        root = x1
        h    = x2 - x1
    else:
        root = x2
        h    = x1 - x2
    
    niter = 0
    while niter <= maxniter:
        niter += 1
        h      = 0.5 * h
        xmid   = root + h
        fmid   = func(xmid)
        if abs(fmid) < tola: return xmid
        if fmid <= 0.0: root = xmid
        absh = abs(h)
        if absh < tolf*abs(root) + tola: return root
        
    else:
        wtxt1 = str(maxniter) + " it'ns not sufficient in zbisect called by "
        wtxt2 = caller + ".\nfunc(x) = " + str(fmid) + " for x = " + str(root)
        warn(wtxt1+wtxt2)
        return root

# end of zbisect

# ------------------------------------------------------------------------------

def zbrent(func, x1, x2, caller='caller', tolf=FOURMACHEPS, \
                         tola=SQRTTINY, maxniter=128, bracket=False):
    """
    Solves the equation func(x) = 0 on [x1, x2] using a variant of Richard 
    Brent's algorithm (more like the "ZEROIN" of Forsythe-Malcolm-Moler). 
    
    NB. The function always returns a value but a warning is printed to stdout 
    if the iteration procedure has not converged! Cf. comment below regarding 
    convergence! 
    
    Arguments:
    ----------
    func      Function having the proposed root as its argument

    x1        Lower search limit (root must be known to be >= x1 unless 
              prior bracketing is used)
    
    x2        Upper search limit (root must be known to be <= x2 unless 
              prior bracketing is used)

    tolf      Desired fractional accuracy of root (a combination of fractional 
              and absolute will actually be used: tolf*abs(root) + tola). tolf 
              should not be < 4.0*machine epsilon since this may inhibit 
              convergence!

    tola      Desired absolute accuracy of root (a combination of fractional 
              and absolute will actually be used: tolf*abs(root) + tola)

    maxniter  Maximum number of iterations

    bracket   If True, x1 and x2 are used in an initial bracketing before 
              solving

    Returns:
    ---------
    Final value of root

    This algorithm is claimed to guarantee convergence within about 
    (log2((b-a)/tol))**2 function evaluations, which is more demanding 
    than bisection. For instance: b-a = 1.0 and tol = 1.8e-12 is guaranteed 
    to converge with about 1,500 evaluations. It normally converges with 
    fewer ITERATIONS, however, and for reasonably "smooth and well-behaved" 
    functions it will be on the average more efficient and accurate than 
    bisection. For details on the algorithm see Forsythe-Malcolm-Moler, 
    as well as Brent, R.P.; "An algorithm with guaranteed convergence 
    for finding a zero of a function", The Computer Journal 14(4), 
    pp. 422-425, 1971.
    """

    if tolf < FOURMACHEPS:
        tolf  = FOURMACHEPS
        wtxt1 = "Fractional tol. less than 4.0*machine epsilon may prevent "
        wtxt2 = "convergence in zbrent. 4.0*macheps will be used instead!"
        warn(wtxt1+wtxt2)

    if tola < 0.0:
        tola  = 0.0
        wtxt1 = "Negative absolute tolerance is not a good idea "
        wtxt2 = "in zbrent. 0.0 (zero) will be used instead!"
        warn(wtxt1+wtxt2)

    assert is_posinteger(maxniter), \
            "Maximum number of iterations must be a positive integer in zbrent!"

    if bracket: x1, x2 = bracketzero(func, x1, x2, caller, maxniter)

    assert x2 > x1, "Bounds must be given with the lower bound first in zbrent!"


    a       = x1
    b       = x2
    c       = x2 ###############################  NOT IN REFERENCES !!!!!
    fa      = func(x1)
    if fa == 0.0: return x1
    fb      = func(x2)
    if fb == 0.0: return x2
    if fsign(fa) == fsign(fb):
        x1, x2 = bracketzero(func, x1, x2, caller, maxniter)
        wtxt1 = "Starting points must be on opposite sides of the root in "
        wtxt2 = "zbrent. Bracketing will be used to find an appropriate span!"
        warn(wtxt1+wtxt2)
    fc      = fb

    niter = 0
    while niter <= maxniter:
        niter += 1

        if fsign(fb) == fsign(fc):
            c  = a
            fc = fa
            d  = b - a
            e  = d

        if abs(fc) < abs(fb):
            a  = b
            b  = c
            c  = a
            fa = fb
            fb = fc
            fc = fa

        tol  = tolf*abs(b) + tola
        tol1 = 0.5 * tol
        xm   = 0.5 * (c-b)

        if abs(xm) <= tol1 or fb == 0.0: return b

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0*xm*q*(q-r)-(b-a)*(r-1.0))
                q = (q-1.0) * (r-1.0) * (s-1.0)

            if p > 0.0: q = -q
            p = abs(p)
            if 2.0*p < min(3.0*xm*q-abs(tol1*q), abs(e*q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d

        else:
            d = xm
            e = d

        a  = b
        fa = fb

        if abs(d) > tol1:
            b = b + d
        else:
            #b = b + sign(tol1, xm)
            if   xm < 0.0: b = b - tol1
            elif xm > 0.0: b = b + tol1
            else:          b = b

        fb = func(b)

    else:
        numb  = int(math.log((x2-x1)/tol, 2)**2 + 0.5)
        wtxt1 = str(maxniter) + " iterations not sufficient in zbrent called by"
        wtxt2 = " " + caller + ". func(x) = " + str(fb) + " for x = " + str(b)
        warn(wtxt1+wtxt2)
        return b

# end of zbrent

# ------------------------------------------------------------------------------

def znewton(fifi2fid, x0, caller='caller', tolf=FOURMACHEPS, \
                            tola=SQRTTINY, maxniter=64):
    """
    Solves the equation fi(x) = 0 using the Newton-Raphson algorithm. 
    
    NB. The function always returns a value but a warning is printed 
    to stdout if the iteration procedure has not converged!
    
    Convergence is fast for the Newton algorithm - at the price of having to 
    compute the derivative of the function. Convergence cannot be guaranteed 
    for all functions and/or initial guesses, either...

    Arguments:
    ----------
    fifi2fid    Function having the desired root as its argument and 
                1: the value of fi, AND
                2: the value of the ratio of its value to the 
                value of its derivative given that root as its 
                outputs, in that order

    x0          Initial guess as to the value of the root
   
    tolf        Desired fractional accuracy of root (a combination of absolute 
                and fractional will actually be used: tolf*abs(root) + tola)

    tola        Desired max absolute difference of fi(root) from zero 
                AND
                desired absolute accuracy of root (a combination of absolute 
                and fractional will actually be used: tolf*abs(root) + tola)
                
    maxniter    Maximum number of iterations

    Additional feature:
    -------------------
    For maxniter == 0 the solution will be taken beyond tolf and tola (if 
    possible) to the limit where either abs(fi(x)) or abs(fi(x)/fid(x)) has 
    stopped shrinking. If convergence is not reached after 2048 iterations, 
    the procedure is halted and the present estimate is returned anyway (a 
    minimum of 8 iterations will be carried out anyhow).

    Returns:
    ---------
    Final value of root
    """

    if tolf < MACHEPS:
        tolf  = MACHEPS
        wtxt1 = "Fractional tolerance less than machine epsilon is not a "
        wtxt2 = "good idea in znewton. Machine epsilon will be used instead!"
        warn(wtxt1+wtxt2)

    if tola < 0.0:
        tola  = 0.0
        wtxt1 = "Negative absolute tolerance is not a good idea "
        wtxt2 = "in znewton. 0.0 (zero) will be used instead!"
        warn(wtxt1+wtxt2)

    assert is_nonneginteger(maxniter), \
       "Maximum number of iterations must be a non-negative integer in znewton!"

    MAXMAX   = 2048
    MINNITER =    8

    if maxniter == 0:
        limit  = True
        maxnit = MAXMAX
    else:
        limit  = False
        maxnit = maxniter

    x  = x0
    fi, fi2fid  = fifi2fid(x)
    if fi == 0.0: return x
    af = abs(fi)
    if not limit: 
        if af < tola: return x
    x -= fi2fid
    ah = abs(fi2fid)
    if not limit:
        if ah < tolf*abs(x) + tola: return x

    niter = 0
    while True:
        niter +=  1
        afprev = af
        ahprev = ah
        fi, fi2fid  = fifi2fid(x)
        if fi == 0.0: return x
        af = abs(fi)
        if limit:
            if af < tola and niter >= MINNITER and af >= afprev: return x
        else:
            if af < tola: return x
        x -= fi2fid
        ah = abs(fi2fid)
        if limit:
            if ah < tolf*abs(x) + tola and niter >= MINNITER and ah >= ahprev: \
                                        return x
        else:
            if ah < tolf*abs(x) + tola: return x
        if niter >= maxnit:
            break

    wtxt1 = str(maxnit) + " iterations not sufficient in znewton called by "
    wtxt2 = caller + ". fi(x) = " + str(fi) + " for x = " + str(x)
    warn(wtxt1+wtxt2)
    return x

# end of znewton

# ------------------------------------------------------------------------------

def zsteffen(func, x0, caller='caller', tolf=FOURMACHEPS, \
                         tola=SQRTTINY, maxniter=512):
    """
    Solves the equation func(x) = 0 using Steffensen's method. Steffensen's 
    method is a method with second order convergence (like Newton's method) 
    at the price of two function evaluations per iteration. It does NOT 
    require that the derivative be computed, as opposed to Newton's method.
        In practice zsteffen seems to require a rather clever initial guess 
    as to the root and/or a lot of iterations since it starts slowly when the 
    initial guess is too far away from the actual root, but it might anyway 
    converge in the end. It is clearly inferior to znewton and should be 
    avoided unless the derivative offers problems with numerics or speed. 
        If computation of the derivative is slow, a good idea is to use 
    znewton for a very small number of iterations to produce an initial guess 
    that can be used as an input to zsteffen (may at times be very efficient).
        And/or: Try the additional feature of taking the solution to the 
    practical limit (cf. below) rather than trying to guess the required 
    number of iterations! 

    For the theory behind Steffensen's method, consult Dahlquist-Bjorck-
    Anderson. 
    
    NB. The function always returns a value but a warning is printed 
    to stdout if the iteration procedure has not converged!

    Arguments:
    ----------
    func        Function having the proposed root as its argument

    x0          Initial guess as to the value of the root
   
    tolf        Desired fractional accuracy of root (a combination of absolute 
                and fractional will actually be used: tolf*abs(root) + tola)

    tola        Desired max absolute difference of func(root) from zero 
                AND
                desired absolute accuracy of root (a combination of absolute 
                and fractional will actually be used: tolf*abs(root) + tola)
                
    maxniter    Maximum number of iterations

    Additional feature: 
    -------------------
    For maxniter == 0 the solution will be taken beyond tolf and tola (if 
    possible) to the limit where either abs(func(x)) or abs(h) - where h is 
    the increment of the root estimate - has stopped shrinking. If convergence 
    is not reached after 2**16 (= 65,536) iterations, the procedure is halted 
    and the present estimate is returned anyway (a minimum of 16 iterations 
    will be carried out anyhow).
    
    Returns:
    ---------
    Final value of root
    """

    if tolf < MACHEPS:
        tolf  = MACHEPS
        wtxt1 = "Fractional tolerance less than machine epsilon is not a "
        wtxt2 = "good idea in zsteffen. Machine epsilon will be used instead!"
        warn(wtxt1+wtxt2)

    if tola < 0.0:
        tola  = 0.0
        wtxt1 = "Negative absolute tolerance is not a good idea "
        wtxt2 = "in zsteffen. 0.0 (zero) will be used instead!"
        warn(wtxt1+wtxt2)

    assert is_nonneginteger(maxniter), \
      "Maximum number of iterations must be a non-negative integer in zsteffen!"

    MAXMAX   = 2**16
    MINNITER = 16

    if maxniter == 0:
        limit  = True
        maxnit = MAXMAX
    else:
        limit  = False
        maxnit = maxniter

    x  = x0
    f  = func(x)
    if f == 0.0: return x
    af = abs(f)
    if not limit:
        if af < tola: return x
    g  = (func(x+f) - f) / f
    h  = f/g
    x -= h
    ah = abs(h)
    if not limit:
        if ah < tolf*abs(x) + tola: return x

    niter = 0
    while True:
        niter +=  1
        afprev = af
        ahprev = ah
        f  = func(x)
        if f == 0.0: return x
        af = abs(f)
        if limit:
            if af < tola and niter >= MINNITER and af >= afprev: return x
        else:
            if af < tola: return x
        g  = (func(x+f) - f) / f
        h  = f/g
        x -= h
        ah = abs(h)
        if limit:
            if ah < tolf*abs(x) + tola and niter >= MINNITER and ah >= ahprev: \
                                        return x
        else:
            if ah < tolf*abs(x) + tola: return x
        if niter >= maxnit:
            break
            

    wtxt1 = str(maxnit) + " iterations not sufficient in zsteffen called by "
    wtxt2 = caller + ". func(x) = " + str(f) + " for x = " + str(x)
    warn(wtxt1+wtxt2)
    return x

# end of zsteffen

# ------------------------------------------------------------------------------

def bracketzero(func, x1, x2, caller='caller', 
                factor=GOLDPHI1, maxniter=32):  # GOLDPHI1 is approx. 1.6
    """
    Bracket a root by expanding from the input "guesses" x1, x2. 
    NB. It is not required that x2 > x1.
    Designed for use prior to any of the one-variable equation solvers. 
    
    The function carries out a maximum of 'maxniter' iterations, 
    each one expanding the original span by a factor of 'factor', 
    until a span is reached in which there is a zero crossing.
    """

    assert factor > 1.0, "Expansion factor must be > 1.0 in bracketzero!"
    assert is_posinteger(maxniter), \
       "Maximum number of iterations must be a positive integer in bracketzero!"

    lo = min(x1, x2)
    up = max(x1, x2)

    flo = func(lo)
    fup = func(up)

    for k in range(0, maxniter):

        if fsign(flo) != fsign(fup): return lo, up

        if abs(flo) < abs(fup):
            lo += factor*(lo-up)
            flo = func(lo)
        else:
            up += factor*(up-lo)
            fup = func(up)

    errtxt1 = "Root bracketing failed after " + str(maxniter)
    errtxt2 = " iterations in bracketzero " + "(called from " + caller + ")"
    raise Error(errtxt1 + errtxt2)

# end of bracketzero

# ------------------------------------------------------------------------------

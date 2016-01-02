# misclib/mathconst.py
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
File contains a number of often used math constants, most of which were taken 
from Abramowitz & Stegun and from L. Rade & B. Westergren, "Beta - Mathematics 
Handbook", 2nd Ed., Chartwell-Bratt, 1990. 
"""
# ------------------------------------------------------------------------------

SQRT05    = 0.7071067811865475244008444  # sqrt(0.5)
SQRT2     = 1.4142135623730950488016887  # sqrt(2.0)
PI        = 3.1415926535897932384626434  # There is also a built-in 'pi'
TWOPI     = 6.2831853071795864769252868  # 2.0*PI
PIHALF    = 1.5707963267948966192313217  # 0.5*PI
PISQRD    = 9.8696044010893586188344910  # PI**2
PIINV     = 0.3183098861837906715377675  # 1.0/PI
SQRTPI    = 1.7724538509055160272981675  # sqrt(PI)
SQRTTWOPI = 2.506628274631000502415765   # sqrt(2.0*PI)
SQRTPIINV = 0.5641895835477562869480795  # sqrt(1.0/PI)
E         = 2.7182818284590452353602875  # e (base of natural logarithms)
LN10      = 2.3025850929940456840179915  # natural logarithm of 10.0 = 1.0/LOGE
LOGE      = 0.4342944819032518276511289  # log10(e), log(e) base 10 = 1.0/LN10
LN2       = 0.6931471805599453094172321  # natural logarithm of 2.0
LOGEB2    = 1.4426950408889634073599247  # log2(e); log(e) base 2 (apocryphic)
GOLDPHI1  = 1.6180339887498948482045868  #"Golden section"= 0.5*(1.0+sqrt(5.0)))
GOLDPHI0  = 0.6180339887498948482045868  # The inverse of "the golden section"

# ------------------------------------------------------------------------------
# machdep/machnum.py
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
File contains a number of machine-specific constants (mainly numerically 
related). The numbers are those of Nils Kjellbert's MacBook (IEEE 754?).

The basic parameters on YOUR computer may be computed using machineparams.py,
with the exception of the smallest positive float with full precision, which 
may be computed using Berthold Hollman's Python version of the well-known 
'machar' function. machar.py can at the time of writing (June 2010) be 
downloaded from the web page http://python.net/crew/bhoel/bhoelHelper-0.1/.
In order for machar.py to run under Python 3.x you will have to replace 
'print machar()' on line 284 by 'print(machar())'. You may also have to 
remove the non-ASCII characters on lines 3, 4 and 26 in machar.py in order 
for it to run.
"""
# ------------------------------------------------------------------------------

from math import sqrt

# ------------------------------------------------------------------------------
_X          = 52           # Machine epsilon exponent - multiple of 4 preferred
MACHEPS     = 0.5**_X      # 2.220446049250313080847263336181640625e-016 exactly
TWOMACHEPS  = 2.0*MACHEPS  # 4.44089209850062616169452667236328125e-016 exactly
FOURMACHEPS = 4.0*MACHEPS  # 8.8817841970012523233890533447265625e-016  exactly
SQRTMACHEPS = 0.5**(_X/2)  # = sqrt(MACHEPS) = 1.490116119384765625e-008 exactly
SQRTSQRTME  = 0.5**(_X/4)  # = sqrt(SQRTMACHEPS) = 1.220703125e-004 exactly
HALFWAYEPS  = SQRTMACHEPS*SQRTSQRTME # 1.818989403545856475830078125e-12 exactly
ONEPMACHEPS = 1.0 + MACHEPS  # Smallest positive float > 1.0
ONEMMACHEPS = 1.0 - MACHEPS  # Largest positive float < 1.0
MINFLOAT    = 0.5**1074      # 4.9406564584124654e-324 = absolute minimum pos.
MAXFLOAT    = 1.797693134862315708e+308  # = 2.0**1023 + 2.0*(1023-1) + ... +
                                         #     + 2.0**(1023-52)
MAXEPSFLOAT = MAXFLOAT    # = largest pos. float with full machine precision
MINEPSFLOAT = 0.5**1022   # = 2.2250738585072013830902327e-308 = 
                          #    = smallest pos. float with full precision
TINY        = 0.5**511   # = sqrt(MINEPSFLOAT), 1.4916681462400413486581931e-154
SQRTTINY    = sqrt(TINY)  # 1.2213386697554620e-077
HUGE        = 2.0**511    # 1.0/TINY, 6.7039039649712985497870125e+153
SQRTHUGE    = sqrt(HUGE)  # 1.0/SQRTTINY, 8.1877371507464133e+076

# ------------------------------------------------------------------------------
# __init__.py
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
# -----------------------------------------------------------------------------
"""
The SimElements package contains classes and modules for general simulation 
problems which may be formulated as dynamic/continuous or discrete-event. 
Combined simulation can also be accomplished using this package. There are 
provisions for uncertainty analysis and optimization as well. 
"""
# -----------------------------------------------------------------------------

__version__ = "1.0"

__author__  = "Nils A. Kjellbert"

__all__     = [ 'abcline',  'abcrand',     'crossing',      'cumrandstrm',  \
                'delays',   'dynamics',    'eventschedule', 'eventschtack', \
                'findmin',  'genrandstrm', 'invrandstrm',   'line',         \
                'linestack', 'randstruct', 'stiffdyn',      'stochdyn'      ]

# -----------------------------------------------------------------------------
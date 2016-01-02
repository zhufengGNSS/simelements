# misclib/errwarn.py
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

def warn(string):     # Does not belong to the Error class!
    """
    Prints out a warning to stdout consisting of the user-provided input
    string preceded by the text "UserWarning: " and closed with a "!".
    DOES NOT FORMALLY BELONG TO THE Error CLASS! 
    
    'warn' makes it possible to issue warnings with a single string output, 
    replacing the overly verbose function from the 'warnings' module. 
    """

    warning  =  "\nUserWarning: " + string + "!"
    print(warning)

# end of warn

# ------------------------------------------------------------------------------

class Error(Exception):
    """
    The class inherits from the built-in Exception class and makes it possible
    to raise an Error without alluding to a built-in exception type by: 
    raise Error(string)
    """
# ------------------------------------------------------------------------------

    def __init__(self, string):
        """
        'string' is some user-provided description of the possible error. 
        """

        self.string = string

    # end of __init__

# ------------------------------------------------------------------------------

    def __str__(self):
        """
        Makes sure that the user-provided input text string is sent to 
        stderr preceded by the text "Error: " 
        """

        string = self.string
        return repr(string)

    # end of __str__

# ------------------------------------------------------------------------------

# end of Error

# ------------------------------------------------------------------------------
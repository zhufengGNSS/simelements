# misclib/deque.py
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

from collections import deque

# ------------------------------------------------------------------------------

class Deque(deque):
    """
    Creates an object from a Python deque to make possible the easy stack
    handling of the kind that is provided by Perl. The class adds a number of 
    methods to those that are available for deques already - methods that 
    operate on objects of this new Deque class as well due to the inheritance.

    The methods of this class are perfectly analogous to the corresponding 
    methods of the Stack class!

    NBNB The class inherits from the built-in 'deque' class!
    NBNBNB THE METHODS ARE NOT FOOL PROOF - THEY COULD CRASH IF EXPOSED 
    TO FOUL PLAY! 
    """
# ------------------------------------------------------------------------------

    def shift(self):
        """
        Analogous to Perl's 'shift': fetch the element at the BOTTOM of the 
        deque, remove it from the deque and let the rest fall down one notch. 
        If the deque is empty 'None' is returned (the equivalent of Perl's 
        'undef').
        
        NB the bottom of the deque corresponds to the index = 0 element.
        
        Outputs:
        --------
        The bottom element of the deque 
        """

        if self:
            bottom = self[0]
            del self[0]
            return bottom

    # end of shift

# ------------------------------------------------------------------------------

    def pop(self):
        """
        Analogous to Perl's 'pop': fetch the element at the TOP of the deque 
        and remove it from the deque. If the deque is empty 'None' is returned 
        (the equivalent of Perl's 'undef').
        NB the top of the deque corresponds to the last element.

        NB Python's has its own 'pop' but it does not return 'None' when the 
        deque is empty - it raises an error....

        Outputs:
        --------
        The top element of the deque 
        """

        if self:
            top = self[-1]
            del self[-1] 
            return top

    # end of pop

# ------------------------------------------------------------------------------

    def unshift(self, x):
        """
        Analogous to Perl's 'unshift': insert one or several elements at the 
        BOTTOM of the deque and move the rest up one notch. The number of 
        elements of the expanded deque is returned.
        
        NBNB WILL PROBABLY NOT WORK FOR LISTS OF STRINGS OR NESTED LISTS!!!! 

        Arguments:
        ----------
        x    single item or list of items

        Outputs:
        ----------
        The number of elements in the unshifted deque 
        """

        try: # First try the hunch that the argument is a sequence of numbers...
            abs(x[0])
            y = reversed(x)   # Don't screw up the original argument sequence
            for b in y: self.extendleft(b)    # Insert elements one by one

        except TypeError:        # ...otherwise the argument is 
            self.appendleft(x)    # a single number or a string...

        return len(self)


    # end of unshift

# ------------------------------------------------------------------------------

    def push(self, x):
        """
        Analogous to Perl's 'push': add one or several elements at the TOP of 
        the deque. The number of elements in the expanded deque is returned. 
        Python's own built-in 'extend' and 'append' have similar properties 
        but do not return the number of elements of the expanded deque.
        
        NBNB WILL PROBABLY NOT WORK FOR LISTS OF STRINGS OR NESTED LISTS!!!! 

        Arguments:
        ----------
        x    single item or list of items

        Outputs:
        ----------
        The number of elements in the pushed deque 
        """

        try:  # First try the hunch that the argument is a sequence of numbers..
            abs(x[0])
            self.extend(x)

        except TypeError:   # ...otherwise the argument is 
            self.append(x)  # a single number or a string...

        return len(self)


    # end of push

# ------------------------------------------------------------------------------

    def next(self):
        """
        Returns the index 0 item but leaves the deque intact. 
        """

        return self[0]

    # end of next

# ------------------------------------------------------------------------------

    def zap(self):
        """
        Empties the deque and returns the prior-to-zapping length 
        (zapping removes all the elements from the deque). 
        """

        length = len(self)

        for k in range(0, length): del self[0]

        return length

    # end of zap

# ------------------------------------------------------------------------------

# end of Deque

# ------------------------------------------------------------------------------
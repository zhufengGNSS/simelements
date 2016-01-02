# misclib/stack.py
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

from misclib.numbers import is_integer, is_nonneginteger

# ------------------------------------------------------------------------------

class Stack(list):
    """
    Creates an object - a stack - from a list to make possible the easy 
    handling of the kind that is provided by Perl. The class adds a number 
    of methods to those that are available for lists already - methods that 
    operate on objects of the Stack class as well due to the inheritance.

    NBNB The class inherits from the built-in 'list' class!
    NBNBNB THE METHODS ARE NOT FOOL PROOF - THEY COULD CRASH IF EXPOSED 
    TO FOUL PLAY! 
    """
# ------------------------------------------------------------------------------

    def shift(self):
        """
        Analogous to Perl's 'shift': fetch the element at the BOTTOM of the 
        stack, remove it from the stack and let the rest fall down one notch. 
        If the stack is empty 'None' is returned (the equivalent of Perl's 
        'undef').
        
        NB the bottom of the stack corresponds to the index = 0 element.
        
        Outputs:
        --------
        The bottom element of the stack 
        """

        if self:
            bottom = self[0]
            del self[0]
            return bottom

    # end of shift

# ------------------------------------------------------------------------------

    def pop(self):
        """
        Analogous to Perl's 'pop': fetch the element at the TOP of the stack 
        and remove it from the stack. If the stack is empty 'None' is returned 
        (the equivalent of Perl's 'undef').
        NB the top of the stack corresponds to the last element.

        NB Python's has its own 'pop' but it does not return 'None' when the 
        stack is empty - it raises an error....

        Outputs:
        --------
        The top element of the stack 
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
        BOTTOM of the stack and move the rest up one notch. The number of 
        elements of the expanded stack is returned.
        
        NBNB WILL PROBABLY NOT WORK FOR LISTS OF STRINGS OR NESTED LISTS!!!! 

        Arguments:
        ----------
        x    single item or list of items

        Outputs:
        ----------
        The number of elements in the unshifted stack 
        """

        try: # First try the hunch that the argument is a sequence of numbers...
            abs(x[0])
            y = reversed(x)   # Don't screw up the original argument sequence
            for b in y: self.insert(0, b)    # Insert elements one by one

        except TypeError:        # ...otherwise the argument is 
            self.insert(0, x)    # a single number or a string...

        return len(self)


    # end of unshift

# ------------------------------------------------------------------------------

    def push(self, x):
        """
        Analogous to Perl's 'push': add one or several elements at the TOP of 
        the stack. The number of elements in the expanded stack is returned. 
        Python's own built-in 'extend' and 'append' have similar properties 
        but do not return the number of elements of the expanded stack.
        
        NBNB WILL PROBABLY NOT WORK FOR LISTS OF STRINGS OR NESTED LISTS!!!! 

        Arguments:
        ----------
        x    single item or list of items

        Outputs:
        ----------
        The number of elements in the pushed stack 
        """

        try:  # First try the hunch that the argument is a sequence of numbers..
            abs(x[0])
            self.extend(x)

        except TypeError:   # ...otherwise the argument is 
            self.append(x)  # a single number or a string...

        return len(self)


    # end of push

# ------------------------------------------------------------------------------

    def splice(self, offset, length, x):
        """
        Analogous to Perl's splice (but all arguments must be present and x 
        must be a single number/string or a list/stack)
        
        This is how it works: remove the slice self(offset:offset+length) 
        and replace it with the contents of x.
        
        If offset < 0 then the offset is from the bottom of the present 
        object and upward - offset will then be len(self) - abs(offset).
        
        If length < 0, then length = len(self) - 1 will be used.
        
        offset > len(self) or offset+length > len(self) is OK.
        
        Returns the slice removed as a stack (or None if nothing was removed).
        """

        assert is_integer(offset), "offset must be an integer in splice!"
        assert is_integer(length), "length must be an integer in splice!"
        if offset < 0: index1 = len(self) + offset
        else:          index1 = offset
        errtxt  = "Negative offset must not try to reach below the "
        errtxt += "bottom element of the object stack in splice!"
        assert index1 >= 0, errtxt
        if length < 0: index2 = len(self) - 1
        else:          index2 = index1 + length

        try:
            removed = Stack(self[index1:index2])
            del self[index1:index2]
        except IndexError:
            removed = None
        if len(removed) == 0: removed = None

        if not isinstance(x, list): x = [x]
        for element in x:
            self.insert(index1, element)
            index1 += 1

        return removed   # A stack (or None)

    # end of splice

# ------------------------------------------------------------------------------

    def next(self):
        """
        Returns the index 0 item but leaves the stack intact. 
        """

        return self[0]

    # end of next

# ------------------------------------------------------------------------------

    def zap(self):
        """
        Empties the stack and returns the prior-to-zapping length 
        (zapping removes all the elements from the stack). 
        """

        length = len(self)

        for k in range(0, length): del self[0]

        return length

    # end of zap

# ------------------------------------------------------------------------------

# end of Stack

# ------------------------------------------------------------------------------
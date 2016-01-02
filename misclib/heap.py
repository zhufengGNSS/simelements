# misclib/heap.py
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

from heapq import heapify, heappush, heappop

from misclib.stack   import Stack
from misclib.errwarn import Error

# ------------------------------------------------------------------------------

class Heap(Stack):
    """
    A class defining a heap (aka a heap queue) - a binary tree structure 
    maintained in a stack and kept sorted so that the smallest item (the 
    top item) is always available as the item having index = 0. 
    NB. The top of a heap is not the same as the top of a stack: the top 
    of the heap is inst_obj[0] while the top of the stack is inst_obj[-1]!

    Heap inherits from the Stack class - a practical matter since the heap is 
    a kind of stack and since some of the Stack methods have their equivalent 
    for heaps. The methods associated with Python's built-in list class are 
    inherited via Stack, but list sort is overridden here. unshift, pop, next 
    and zap are inherited from Stack without overriding, while push, shift 
    and slip are overridden by their corresponding heap methods defined in 
    this class.
    
    WARNING: Using other list-related methods on a Heap object than the ones 
    defined in this class or the ones inherited first-hand from Stack may ruin 
    the heap structure!!!!!!
    """
# ------------------------------------------------------------------------------

    def sort(self):
        """
        Sorts the stack half-way to form a bona fide heap using 
        Python's built-in heapify function. 
        
        NB. This is NOT the ordinary list sort. List sort (from the 
        list class via the Stack class) is overridden. 
        """

        heapify(self)    # Heapifies the input list/stack in place

    # end of sort

# ------------------------------------------------------------------------------

    def push(self, x):
        """
        Adds a single item or a sequence to the heap and places it/them in 
        its/their right place in the binary tree. Returns the input argument, 
        which has turned out to be a practical thing...

        Arguments:
        ----------
        x     the single item or sequence to place in the heap
        
        Outputs:
        ----------
        the argument 
        """

        try:
            for item in x: heappush(self, item)    # If it is a list/stack

        except TypeError:
            heappush(self, x)                # If it is a single item

        return x   # NB VERY IMPORTANT !!!!!!!!!!

    # end of push

# ------------------------------------------------------------------------------

    def shift(self):
        """
        Returns the heap's top item, removes it from the heap and lets items 
        below percolate up through the binary tree to maintain the heap 
        structure. Based on Python's built-in heappop but returns None when 
        the heap is empty (practical).

        Outputs:
        --------
        the top element of the heap 
        """

        try:               return heappop(self)
        except IndexError: return None

    # end of shift

# ------------------------------------------------------------------------------

    def splice(self, *arg):
        """
        Same as the push method but None is returned and iflag is not used 
        (the method is motivated by the fact that there is a corresponding 
        Stack method).
        """

        raise Error("splice is not implemented in the Heap class!")

    # end of splice

# ------------------------------------------------------------------------------

# end of Heap

# ------------------------------------------------------------------------------
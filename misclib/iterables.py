# misclib/iterables.py
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
A number of functions for manipulating iterables, lists and dicts. Some of 
them are alternative versions of built-in list object class methods known 
to change the instance in place. The alternatives operate on a copy and 
returns the desired, modified version of the input without screwing up the 
original. 
"""
# ------------------------------------------------------------------------------

from array  import array
from bisect import bisect

from misclib.errwarn import Error

# ------------------------------------------------------------------------------

def is_darray(vector):
    """
    Used to find out if a vector (list/tuple) is a 'd' array. Returns True 
    if input is a 'd' array, returns False if it is not. 
    """

    return isinstance(vector, array) and isinstance(vector[0], float)

# end of is_darray

# ------------------------------------------------------------------------------

def reorder(vector, indices, straighten=False):
    """
    Returns a reordered version of an input list/tuple/array given an index 
    list/tuple containing the desired new order/permutation expressed for 
    instance as (2, 4, 3, 0...). The input vector is reordered so that its 
    2:nd element is now placed first, its 4:th element is placed second etc. 
    in the example above. The list/tuple/array to be reordered is first 
    sorted if so requested.
    --------------------------------
    NB. The input list/tuple/array is NOT affected - all operations are 
    made on a copy!

    Arguments:
    ----------
    vector      a list/tuple/array of elements to be reordered
    indices     a list/tuple containing the desired order (zero-based)
        (they must have the same length)
    straighten  indicator for prior sorting of input vector (True or False)
    
    Outputs:
    ----------
    reordered copy of the input list/tuple/array 
    """

    assert len(vector) == len(indices), \
                         "Inputs are of unequal length in reorder!"

    auxvect = list(vector)
    if straighten: auxvect.sort()  # First sort vector to be 
    outvect = []                   # reordered if so requested
    if is_darray(vector): outvect = array('d', outvect)
    for index in indices: outvect.append(auxvect[index])

    return outvect

# end of reorder

# ------------------------------------------------------------------------------

def binsearch(vector, x, sortcheck=False):
    """
    Binary search to find the place of x in an ordered list.
    Returns the index of the place x would occupy in the sequence
    and the last index of the input list.
    The input vector is checked with respect to order if so requested
    NB the function might not work for all kinds of sequences !!!!!!!!!!!!!!

    Python's own 'bisect' module might do the same trick but presupposes that 
    the list to be inspected is sorted already. It replaces the portion of the
    function that follows after the sortcheck only.

    Arguments:
    ----------
    vector       hopefully ordered list
    x            item the place of which is requested
    sortcheck    logical - answer to the question whether the function check
                     the sorting or not (if found not sorted on check an error
                     is raised)
    
    Outputs:
    --------
    index        the place in the list the new item would occupy
    nindex       the number of the last element in the original/input list 
    """

    nindex = 0

    try:
        nindex = len(vector) - 1

        if sortcheck:
            errortext = "Input vector in binSearch is not sorted!"
            for k in range (1, nindex):
                assert vector[k] >= vector[k-1], errortext
                pass
        index = bisect(vector, x) # replaces the hashed-in portion below

    except TypeError:
        index = 0


    return index, nindex   # NB nindex is the last index of original/input list

# end of binsearch

# ------------------------------------------------------------------------------

def iters2dict(iterk, iterv):
    """
    Assembles two associated iterables (may be lists, arrays or any iterable 
    data types) into a dict. The first iterable is assumed to contain the keys. 
    """

    assert len(iterk) == len(iterv), \
                        "inputs are of unequal length in iters2dict!"
        
    return dict(zip(iterk, iterv))

# end of iters2dict

# ------------------------------------------------------------------------------

def dict2iters(dikt):
    """
    Breaks down a dict into two iterables. The result is returned with the 
    iterable containing the keys first. The original association may be lost!
    """

    return dikt.keys(), dikt.values()

# end of dict2iters

# ------------------------------------------------------------------------------

def dict2iternlist(dikt):
    """
    Breaks down a dict into an iterable and a list that are linked according 
    to the original key-value association. The result is returned with the 
    iterable containing the keys first.
    """

    keys   = dikt.keys()
    values = []
    for a in keys: values.append(dikt[a])

    return keys, values

# end of dict2iternlist

# ------------------------------------------------------------------------------

def dict2iterndarray(dikt):
    """
    Breaks down a dict into an iterable and a 'd' array that are linked 
    according to the original key-value association. The result is returned 
    with the iterable containing the keys first.
    """

    keys   = dikt.keys()
    values = array('d', [])
    for a in keys: values.append(dikt[a])

    return keys, values

# end of dict2iterndarray

# ------------------------------------------------------------------------------

def dict2lists(dikt, sort=False):
    """
    Breaks down a dict into two lists that are linked according to the original 
    key-value association. The lists may be sorted based on the original keys, 
    if so desired.
    """

    values = []

    if sort:
        keys = sorted(dikt)
        for key in keys:
            values.append(dikt[key])
    else:
        keys = list(dikt.keys())
        for key in keys:
            values.append(dikt[key])

    return keys, values

# end of dict2lists

# ------------------------------------------------------------------------------

def dict2listndarray(dikt, sort=False):
    """
    Breaks down a dict into a list and a 'd' array that are linked according 
    to the original key-value association. The lists may be sorted based on 
    the original keys, if so desired.
    """

    values = array('d', [])

    if sort:
        keys = sorted(dikt)
        for key in keys:
            values.append(dikt[key])
    else:
        keys = list(dikt.keys())
        for key in keys:
            values.append(dikt[key])
        
    return keys, values

# end of dict2listndarray

# ------------------------------------------------------------------------------
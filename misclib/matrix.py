# misclib/matrix.py
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

from array import array
from copy  import deepcopy

from misclib.errwarn import Error

# ------------------------------------------------------------------------------

class Matrix(list):
    """
    A class for creating and manipulating matrices, based on the built-in array 
    and list classes. Most methods CHANGE the instance object IN PLACE. Some of 
    the methods have a corresponding function in the numlib.matrixops module 
    that does NOT screw up the object matrix.
    ------------
    The class can be used to build matrices row-by-row with constructs such as
        from array import array
        ...
        arr1   = array('d', [x0, x1, x2...])
        arr2   = array('d', [xi, xj, xk...])
        matrix = Matrix()    # An empty matrix
        matrix.append(arr1)
        matrix.append(arr2)
        ...
    or to fill a matrix directly with the construct
        matrix = Matrix([arr1, arr2])

    Blank, zero and unity matrices can also be used to predefine a matrix 
    so as to make possible the exact initiation of individual elements by 
    addressing matrix[row_number][column_number]

    ----------
    Vector objects in the matrix class can be built using
        vector = Matrix([array('d', [7.0, 8.0, 9.0])])
    or
        arr    = array('d', [x0, x1, x2...])
        vector = Matrix([arr])
    yielding a row vector that is a single-row matrix. A column vector can
    be created by first creating a row vector and then transposing it.
    NB Do not try using 'append' to create column vectors from single numbers 
    directly - it does not work!!

    ----------
    The analogue to copylist = list(existing_list) to create a new list object 
    that is a copy of an existing list is - here:
        from copy import deepcopy
        copymatrix = deepcopy(existing_matrix) 
    to create a copy of an existing matrix; or a list structure that has the 
    structure of a matrix:
        copymatrix = Matrix(deepcopy(existing_list_with_matrix_structure))
    and not screw up the original by subsequent operations on the copy.

    ----------
    NB1 The square/box parentheses must be used in the manner desribed above!!!
    ----------
    NB2 The class and its methods will generally assume that floating-point 
    numbers appear as elements in matrices and vectors.
    ----------
    NB3 The class methods will work even if ordinary lists are used instead of 
    arrays when bulding a matrix. The internals will still use arrays whenever 
    possible.
    """
# ------------------------------------------------------------------------------

    def blank(self, nrows, ncols):
        """
        Create a matrix with dimension nrows*ncols containing only NaN 
        elements from scratch. 
        """

        if self:
            raise Error("Instance matrix exists already in Matrix.blank!")

        for k in range(0, nrows):
            row = array('d', ncols*[float('nan')])
            self.append(row)

        self.__nrows = nrows
        self.__ncols = ncols

    # end of blank

# ------------------------------------------------------------------------------

    def zero(self, nrows, ncols):
        """
        Create a zero matrix with dimension nrows*ncols from scratch. 
        """

        if self:
            raise Error("Instance matrix exists already in Matrix.zero!")

        for k in range(0, nrows):
            row = array('d', ncols*[0.0])
            self.append(row)

        self.__nrows = nrows
        self.__ncols = ncols

    # end of zero

# ------------------------------------------------------------------------------

    def unity(self, ndim):
        """
        Create a unity matrix from scratch. 
        """

        if self:
            raise Error("Instance matrix exists already in Matrix.unity!")

        for k in range(0, ndim):
            row    = array('d', ndim*[0.0])
            row[k] = 1.0
            self.append(row)

        self.__nrows = ndim
        self.__ncols = ndim

    # end of unity

# ------------------------------------------------------------------------------

    def transpose(self):
        """
        Transpose matrix - NB the method changes the object matrix in place -
        the original is destroyed!!!! 
        """

        try:
            nrows = self.__nrows
            ncols = self.__ncols
        except AttributeError:
            nrows, ncols = self.size('Matrix.transpose')

        copymatrix   = deepcopy(self)

        for k in range(nrows): del self[0]    # Delete one row at a time

        for k in range(0, ncols): # List compreh. is used for the innermost loop
            self.append(array('d', [row[k] for row in copymatrix]))  
        del copymatrix

        self.__nrows = ncols
        self.__ncols = nrows

    # end of transpose

# ------------------------------------------------------------------------------

    def flatten(self, column=False):
        """
        Flattens the instance object matrix into a row vector matrix, or a  
        column vector matrix if so requested. NB the method changes the matrix 
        in place - the original is destroyed!!!! 
        """

        try:
            nrows = self.__nrows
            ncols = self.__ncols
        except AttributeError:
            nrows, ncols = self.size('Matrix.flatten')

        flist  = []
        for k in range(0, nrows):
            flist.extend(self[k])
        fmatrix = Matrix([flist])

        for k in range(0, nrows): del self[0]    # Delete one row at a time

        self.append(array('d', fmatrix[0]))  # Add the one row from fmatrix
        self.__nrows = 1
        self.__ncols = nrows*ncols
        del fmatrix

        if column: self.transpose()

    # end of flatten

# ------------------------------------------------------------------------------

    def scale(self, scalar):
        """
        Multiplies the instance object matrix by a scalar constant. 
        NB the method changes the matrix in place - the original is 
        destroyed!!!! 
        """

        try:
            nrows = self.__nrows
            ncols = self.__ncols
        except AttributeError:
            nrows, ncols = self.size('Matrix.scale')

        newmatrix = Matrix()
        newmatrix.blank(nrows, ncols)
        for k in range(0, nrows):
            for j in range(0, ncols):
                newmatrix[k][j] = scalar*self[k][j]

        for k in range(0, nrows):
            del self[0]              # Delete one row at a time

        for k in range(0, nrows):    # Add one row at a time of tmatrix
            self.append(array('d', newmatrix[k]))

        del newmatrix

        self.__nrows = nrows
        self.__ncols = ncols

    # end of scale

# ------------------------------------------------------------------------------

    def __add__(self, other):
        """
        For adding two matrices using "m1 + m2". 
        """

        try:
            nrowss = self.__nrows
            ncolss = self.__ncols
        except AttributeError:
            nrowss, ncolss = self.size('Matrix.__add__')
        try:
            nrowso = other._nrows
            ncolso = other._ncols
        except AttributeError:
            nrowso, ncolso = other.size('Matrix.__add__')

        assert nrowss == nrowso and ncolss == ncolso, \
                     "Matrices are of unequal dimension in Matrix.__add__!"


        smatrix = Matrix()
        for k in range(0, nrowss):
            row = []
            for j in range(0, ncolss):
                x = self[k][j] + other[k][j]
                row.append(x)
            smatrix.append(array('d', row))

        smatrix._nrows = len(smatrix)
        smatrix._ncols = len(smatrix[0])

        return smatrix

    # end of __add__

# ------------------------------------------------------------------------------

    def __sub__(self, other):
        """
        For subtracting two matrices using "m1 - m2".
        """

        try:
            nrowss = self.__nrows
            ncolss = self.__ncols
        except AttributeError:
            nrowss, ncolss = self.size('Matrix.__sub__')
        try:
            nrowso = other._nrows
            ncolso = other._ncols
        except AttributeError:
            nrowso, ncolso = other.size('Matrix.__sub__')

        assert nrowss == nrowso and ncolss == ncolso, \
                      "Matrices are of unequal dimension in Matrix.__sub__!"


        dmatrix = Matrix()
        for k in range(0, nrowss):
            row = []
            for j in range(0, ncolss):
                x = self[k][j] - other[k][j]
                row.append(x)
            dmatrix.append(array('d', row))

        dmatrix._nrows = len(dmatrix)
        dmatrix._ncols = len(dmatrix[0])

        return dmatrix

    # end of __sub__

# ------------------------------------------------------------------------------

    def __mul__(other, self):
        """
        For multiplying two matrices using "m1 * m2".
        NB this is the INNER (dot) product!!!
        "self" is multiplied by "other" from the left. 
        """

        try:
            nrowss = self.__nrows
            ncolss = self.__ncols
        except AttributeError:
            nrowss, ncolss = self.size('Matrix.__mul__')
        try:
            nrowso = other._nrows
            ncolso = other._ncols
        except AttributeError:
            nrowso, ncolso = other.size('Matrix.__mul__')

        assert ncolso == nrowss, \
                  "Matrices have uncompatible dimensions in Matrix.__mul__!"


        # Create product matrix
        pmatrix = Matrix()

        # Turn product matrix into a zero nrowso by ncolss matrix
        pmatrix.zero(nrowso, ncolss)

        # Replace elements in the product matrix
        for k in range(0, nrowso):
            for j in range(0, ncolss):
                element = 0.0
                for m in range(0, ncolso):
                    element += other[k][m]*self[m][j]
                pmatrix[k][j] = element

        pmatrix._nrows = len(pmatrix)
        pmatrix._ncols = len(pmatrix[0])
        return pmatrix

    # end of __mul__

# ------------------------------------------------------------------------------

    def size(self, caller='caller'):
        """
        Checks the object matrix with respect to size. Returns the number of 
        rows and the number of columns. Raises an error if structure of matrix 
        is flawed. 'caller' may be used to indicate from which method/function 
        'size' was called. 
        """

        try:
            nrows = len(self)
            ncols = len(self[0])
            return nrows, ncols

        except IndexError:
            errortext = "Structure of matrix is flawed in " + caller + "!"
            raise IndexError(errortext)

    # end of size

# ------------------------------------------------------------------------------

# end of Matrix

# ------------------------------------------------------------------------------
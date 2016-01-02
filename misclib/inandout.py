# misclib/inandout.py
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
Functions for reading from input files and and writing to output files. 
"""
# ------------------------------------------------------------------------------

def readtab(filename):
    """
    This functions reads an extracts the data from a multiple-line,
    tab-delimited input file where the first string of each line is
    treated as an identifier and the rest as integers, floats or strings.
    The input is returned in a dict having the string before the first
    tab of each line as keys and the rest of the line as a list (or a 
    single constant if appropriate). Lines having the hash character ('#')
    as its leading character in the first position will be disregarded.
    This feature may be used to place comments in the input file for 
    more clarity.
    
    For instance: assume that we need to extract the data from a file
    called 'spam.in' with the following contents (the delimiting tabs 
    are not visible below but are present...):

    # This is a spam example
    Ham	4.5
    14	2.0e-3	3	eggs
    
    where 'Ham' and '14' are used for identification. Then
    
    inputs = readtab('spam.in')
    
    in a program gives the dict 'inputs' the following content:
    
    {'Ham': 4.5, '14': [0.002, 3, 'eggs']}
    """

    # ---------------------

    # Read the input file and place the lines in list 'lines'
    input = open(filename, 'r')
    lines = input.readlines()
    input.close()

    # Remove the line feeds and carriage returns and split each line at the 
    # tabs, place each line as a separate list containing a list containing 
    # the separate elements: i.e. a nested list, here named 'first' (lines 
    # having the hash character '#' as its leading character are omitted):
    first  = []
    for line in lines:
        if line[0] != '#':
            newline = line.replace('\n', '')     # Removes the linefeeds
            newline = newline.replace('\r', '')  # Removes the carriage returns
            first.append(newline.rsplit('\t'))   # Splits the string

    # Go through all the lists in 'first' and turn the first element into 
    # a key having the rest of the elements in a list as an item placed in 
    # a dict called 'second':
    second = {}
    for line in first:
        second[line[0]] = line[1:]

    # Convert string elements to integers and floats as appropriate:
    for key in second:
        k = 0
        for elem in second[key]:
            if elem[0].isdigit():
                if elem.find('.') == -1:  # then it's an integer:
                    second[key][k] = int(elem)
                else:                     # then it's a float:
                    second[key][k] = float(elem)
            k += 1

    # Single constants should not be placed in lists:
    for key in second:
        if len(second[key]) == 1:
            second[key] = second[key][0]

    # Return the final 'second':
    return second

# end of readtab

# ------------------------------------------------------------------------------

def writetab(fileattrdp, fileattrdc, inputline, \
                                          newline=None, frmt=0, screen=False):
    """
    Function for writing ONE LINE of numerical output to the end of a file 
    using a tab-delimited format. In fact it writes to TWO files: one using 
    decimal points and the other using decimal commas.

    NB:  The two files have first to be opened for writing and then finally 
    closed using standard Python procedure. If you open the files using the 
    'newline' specifier set to the appropriate escape character(s) for your 
    operating system - either '\n', '\r' or '\r\n' - then you don't have to 
    provide a specific input for the 'newline' argument to this function. 
    Otherwise...(cf. below)...
    
    fileattrdp and fileattrdc are the names of the two output file OBJECTS 
    that have to be created using the 'open' function before writing,
    
    inputline is the next line to be written = a tuple of floats,

    newline is used to specify which newline format should be used when 
    your output files are opened NOT using the 'newline' specifier in the 
    built-in Python 'open' function; it must be either '\n', '\r' or 
    '\r\n', 

    frmt is an integer such that frmt < 0 provides %e type output with 
    -frmt as the number of decimals and frmt = 0 gives %e type format with 
    Python's default number of decimals (6?); otherwise frmt is interpreted 
    as the desired number of decimals in a %f format,
    
    screen=True sends the output to stdout as well, using decimal point 
    representation.
    """

    # ---------------

    assert newline==None or newline=='\n' or newline=='\r' or newline=='\r\n', \
                     "newline must be '\n', '\r' or '\r\n' in writetab!"
    assert isinstance(frmt, int), "frmt must be an integer in writetab!"

    if    frmt <  0: pre = '{0:.' + str(-frmt) + 'e}'
    elif  frmt == 0: pre = '{0:e}'
    else:            pre = '{0:.' + str( frmt) + 'f}'

    lenm1  = len(inputline) - 1
    putout = ''
    for k in range(0, lenm1):
        putout += pre.format(inputline[k]) + '\t'
    putout += pre.format(inputline[lenm1])

    if screen: print(putout)

    if newline: putout += newline

    fileattrdp.write(putout)
    putout = putout.replace(".", ",")
    fileattrdc.write(putout)

# end of writetab

# ------------------------------------------------------------------------------

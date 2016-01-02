# abcline.py
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

from abc    import ABCMeta, abstractmethod
from array  import array

from misclib.deque    import Deque
from misclib.stack    import Stack
from misclib.errwarn  import Error

# ------------------------------------------------------------------------------

class ABCLine(metaclass=ABCMeta):
    """
    This class contains everything that is common to the Line and LineStack 
    classes. Since this is also an abstract base class, it cannot be used in a 
    standalone fashion. Its methods and attributes can only be reached through 
    its subclasses Line and LineStack, which inherit from this class.
    """
# ------------------------------------------------------------------------------

    @abstractmethod
    def __init__(self, line, nserv):
        """
        line  is 'Line' or 'LineStack'
        nserv is the initial number of servers
        """

        # Attributes made available from the outside (not assignable, though):
        # --------------------------------------------------------------------
        
        self.__narriv    = 0     # Accumulated number of arrivers until present
        
        self.__ninline   = 0     # Present number waiting in line

        self.__nfreeserv = nserv # Present number of free servers

        self.__nbalked   = 0     # Accumulated number of balkers

        self.__nreneged  = 0     # Accumulated number of renegers  

        self.__nescaped  = 0     # = self.__nbalked + self.__nreneged


        # Attributes not available from the outside:
        # --------------------------------------------------------------------
        
        self.__nserv    = nserv # Initial number of free servers

        if   line == 'Line':       self.__line = Deque()
        elif line == 'LineStack':  self.__line = Stack()
        else: raise Error("'line' must be 'Line' or 'LineStack' in ABCLine!")

                                # dict containing the "reneging times" (time  
        self.__renegers = {}    # points of the future reneging events) with 
                                # the corresponding arrival times as keys

        self.__wtimes   = array('d', [])   
                          # 'd' array of waiting times for those not escaped

        self.__length   = {}    # dict containing the line length history with 
                                # the corresponding time spans as keys

        self.__prevctl  = 0.0   # The previous clock time when the line length 
                                # was last changed

                                # dict containing the history of the number of 
        self.__systemh  = {}    # customers in the system with the corresponding
                                # time spans as keys

        self.__prevcts  = 0.0   # The previous clock time when the number of 
                                # customers in system was last changed

        self.__length   = {}    # dict containing the line length history with 
                                # the corresponding time spans as keys

        self.__prevctl  = 0.0   # The previous clock time when the line length 
                                # was last changed

                                # dict containing the history of the number of 
        self.__idleh    = {}    # of free/idle servers in the system with the 
                                # corresponding time spans as keys

        self.__prevcti  = 0.0   # The previous clock time when the number of 
                                # free/idle servers was last changed

    # end of __init__

# ------------------------------------------------------------------------------

    def __getattr__(self, attr_name):
        """
        This method overrides the built-in __getattr__ and makes the values
        of the internal attributes externally available.
        """

        if   attr_name == "ninline":    return self.__ninline
        elif attr_name == "nfreeserv":  return self.__nfreeserv
        elif attr_name == "narriv":     return self.__narriv
        elif attr_name == "nbalked":    return self.__nbalked
        elif attr_name == "renegers":   return self.__renegers
        elif attr_name == "nreneged":   return self.__nreneged
        elif attr_name == "nescaped":   return self.__nescaped

    # end of __getattr__

# ------------------------------------------------------------------------------

    def __setattr__(self, attr_name, value):
        """
        This method overrides the built-in __setattr__ and makes the values of 
        the internal attributes externally available more difficult to screw 
        up from the outside.
        """

        if attr_name == "ninline"   \
        or attr_name == "nfreeserv" \
        or attr_name == "narriv"    \
        or attr_name == "nbalked"   \
        or attr_name == "renegers"  \
        or attr_name == "nreneged"  \
        or attr_name == "nescaped"  :
            errtxt1 = ": Can't change value/length of attribute"
            errtxt2 = " from the outside!"
            raise Error(attr_name + errtxt1 + errtxt2)

        else:
            self.__dict__[attr_name] = value

    # end of __setattr__

# ------------------------------------------------------------------------------

    def place_last_in_line(self, times):
        """
        Add one or several arrival times at the back end of the line. 
        The length of the expanded line is returned. 

        NB The elements in an iterable will be placed so that the last 
        will be the last in the line etc.

        Arguments:
        ----------
        times     single time or tuple/list of times

        Outputs:
        ----------
        The number presently in the expanded line
        """

        try:       # First try the hunch that the argument is a sequence:
            self.__narriv += len(times)
            for tim in times:
                self.__length[tim-self.__prevctl]  = len(self.__line)
                self.__systemh[tim-self.__prevcts] = len(self.__line) + \
                                                 self.__nserv - self.__nfreeserv
                self.__prevctl = tim
                self.__prevcts = tim
            
        except TypeError:   # ..otherwise the arg is a single number
            self.__narriv += 1
            self.__length[times-self.__prevctl]  = len(self.__line)
            self.__systemh[times-self.__prevcts] = len(self.__line) + \
                                               self.__nserv - self.__nfreeserv
            self.__prevctl = times
            self.__prevcts = times

        self.__line.push(times)
            
        self.__ninline = len(self.__line)

        return self.__ninline

    # end of place_last_in_line

# ------------------------------------------------------------------------------

    def place_first_in_line(self, times):
        """
        Add one or several arrival times at the front end of the line. 
        The length of the expanded line is returned. 

        NB The elements in an iterable will be placed so that the first 
        will be the first in the line etc.

        Arguments:
        ----------
        times     single time or tuple/list of times

        Outputs:
        ----------
        The length of the expanded line 
        """

        try:          # First try the hunch that the argument is a sequence:
            self.__narriv += len(times)
            for tim in times:
                self.__length[tim-self.__prevctl]  = len(self.__line)
                self.__systemh[tim-self.__prevcts] = len(self.__line) + \
                                                 self.__nserv - self.__nfreeserv
                self.__prevctl = tim
                self.__prevcts = tim

        except TypeError:   # ..otherwise the arg is a single number
            self.__narriv += 1
            self.__length[times-self.__prevctl]  = len(self.__line)
            self.__systemh[times-self.__prevctl] = len(self.__line) + \
                                               self.__nserv - self.__nfreeserv
            self.__prevctl = times
            self.__prevcts = times

        self.__line.unshift(times)
            
        self.__ninline = len(self.__line)

        return self.__ninline

    # end of place_first_in_line

# ------------------------------------------------------------------------------

    def call_next_in_line(self, calltime):
        """
        Fetch the first arrival time at the front end of the line,
        remove it from the line, and make one server busy.
        
        Outputs:
        --------
        The arrival time at the front end of the line 
        """

        self.__length[calltime-self.__prevctl]  = len(self.__line)
        self.__idleh[calltime-self.__prevcti]   = self.nfreeserv
        self.__systemh[calltime-self.__prevcts] = len(self.__line) + \
                                              self.__nserv - self.__nfreeserv
        self.__prevctl    = calltime
        self.__prevcti    = calltime
        self.__prevcts    = calltime

        arrivaltime       = self.__line.shift()

        self.__nfreeserv -= 1
        if self.__nfreeserv < 0:
            raise Error("Number of servers are negative in call_next_in_line!")

        self.__wtimes.append(calltime-arrivaltime)

        self.__ninline    = len(self.__line)

        return arrivaltime

    # end of call_next_in_line

# ------------------------------------------------------------------------------

    def remove_last_in_line(self, tim):
        """
        Fetch the last arrival time at the back end of the line and remove 
        it from the line. To be used - for instance - when a customer that 
        already has been placed in line balks (even balkers must first be 
        placed in line - before balking!).

        Outputs:
        --------
        The arrival time at the back end of the line 
        """

        self.__length[tim-self.__prevctl]  = len(self.__line)
        self.__systemh[tim-self.__prevcts] = len(self.__line) + \
                                         self.__nserv - self.__nfreeserv
        self.__prevctl = tim
        self.__prevcts = tim

        lasttime       = self.__line.pop()

        self.__ninline = len(self.__line)

        return lasttime

    # end of remove_last_in_line

# ------------------------------------------------------------------------------

    def server_freed_up(self, tim):
        """
        Adds '1' to the present number of free servers - 
        to be used when a customer has been served.
        May also be used to change the total number of servers 
        during the course of a simulation.
        """

        self.__idleh[tim-self.__prevcti]   = self.nfreeserv
        self.__systemh[tim-self.__prevcts] = len(self.__line) + \
                                         self.__nserv - self.__nfreeserv
        self.__nfreeserv += 1
        self.__prevcti = tim
        self.__prevcts = tim

    # end of server_freed_up

# ------------------------------------------------------------------------------

    def balker(self, tim):
        """
        Removes the arrival just placed last in line. Returns the arrival 
        time of the balker.
        
        NB. Balkers must first be placed in line - before balking!
        Outputs:
        --------
        The arrival time of the balker 
        """

        self.__nbalked  += 1
        self.__nescaped += 1
        
        return self.remove_last_in_line(tim)

    # end of balker

# ------------------------------------------------------------------------------

    def reneger(self, arentime):
        """
        To be used when a "renege" type event is picked up by get_next_event 
        and removes the corresponding arrival time (arentime) in the line of 
        arrival times GIVEN that it has not been removed already from calling  
        call_next_in_line (the existence of the corresponding arrival time in 
        the line is checked first). 
        """

        arrivaltime = self.__renegers[arentime]
        del self.__renegers[arentime]

        if arrivaltime in self.__line:
            self.__length[arentime-self.__prevctl]  = len(self.__line)
            self.__systemh[arentime-self.__prevcts] = len(self.__line) + \
                                                  self.__nserv - self.__nfreeserv
            self.__prevctl   = arentime
            self.__prevcts   = arentime
            self.__line.remove(arrivaltime)
            self.__ninline   = len(self.__line)
            self.__nreneged += 1
            self.__nescaped += 1
            return True 
        else:
            return False

    # end of reneger

# ------------------------------------------------------------------------------

    def waiting_times_all(self):
        """
        Returns an unsorted 'd' array containing the waiting times 
        for all served. 
        """

        return self.__wtimes

    # end of waiting_times_all

# ------------------------------------------------------------------------------

    def waiting_times_linedup(self):
        """
        Returns an unsorted 'd' array containing the waiting times 
        only for those who had to wait in line before being served. 
        """

        wtimesshort = array('d', [])
        for wt in self.__wtimes:
            if wt != 0.0: wtimesshort.append(wt)

        return wtimesshort

    # end of waiting_times_linedup

# ------------------------------------------------------------------------------

    def line_stats(self):
        """
        Returns a dict containing line length statistics with the line lengths 
        as keys and the corresponding times as values.
        """

        # This method turns the self.__length dict around: the statistics are 
        # collected with the line length as keys and the corresponding times 
        # as values (there will be fewer elements in the returned dict than in 
        # self.__length, of course...)

        statdict = {}
        for keytime in self.__length:
            try:
                statdict[self.__length[keytime]] += keytime
            except KeyError:
                statdict[self.__length[keytime]]  = keytime

        return statdict

    # end of line_stats

# ------------------------------------------------------------------------------

    def idle_stats(self):
        """
        Returns a dict containing server statistics with the number of idle 
        servers as keys and the corresponding times as values.
        """

        # This method turns the self.__idleh dict around: the statistics are 
        # collected with the number of free/idle servers in the system as keys 
        # and the corresponding times as values (there will be fewer elements 
        # in the returned dict than in self.__idleh, of course...)

        statdict = {}
        for keytime in self.__idleh:
            try:
                statdict[self.__idleh[keytime]] += keytime
            except KeyError:
                statdict[self.__idleh[keytime]]  = keytime

        return statdict

    # end of idle_stats

# ------------------------------------------------------------------------------

    def system_stats(self):
        """
        Returns a dict containing statistics for the total number of customers 
        in the system as keys and the corresponding times as values.
        """

        # This method turns the self.__systemh dict around: the statstics are 
        # collected with the number of customers in the system as keys and the 
        # corresponding times as values (there will be fewer elements in the 
        # returned dict than in self.__systemh, of course...)

        statdict = {}
        for keytime in self.__systemh:
            try:
                statdict[self.__systemh[keytime]] += keytime
            except KeyError:
                statdict[self.__systemh[keytime]]  = keytime

        return statdict

    # end of system_stats

# ------------------------------------------------------------------------------

# end of ABCLine

# ------------------------------------------------------------------------------
# eventschtack.py
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

from bisect import bisect

from misclib.stack import Stack

# ------------------------------------------------------------------------------

class EventScheduleStack:
    """
    The class defines a schedule of events in a general discrete-event 
    simulation. It relies on the misclib.Stack class (but it does not 
    inherit from the Stack class). 

    EventScheduleStack is less efficient (=slower) than the heap-based 
    EventSchedule class but the two classes are otherwise equivalent in 
    principle. Since it uses the Stack class also for the event types,
    and since all the methods of the list class are available via the 
    Stack class, EventScheduleStack may be used for creating subclasses 
    to this class which can handle more complex types of schedules than 
    those that can be handled by the dict-based EventSchedule.

    An additional feature of EventScheduleStack is that it can handle TIES 
    perfectly (the dict-based EventSchedule only handles ties approximately).
    """
# ------------------------------------------------------------------------------

    def __init__(self, eventlist=[], timelist=[], sort=False):
        """
        Creates two new stacks: one for the events and one for the corresponding 
        time points. The events could for instance be described by strings. The 
        times are (of course) floating-point numbers. The two stacks can be 
        filled here but they have to be synchronized and in temporal order. 
        """

        assert len(timelist) == len(eventlist), \
                "input lists are of unequal length in EventScheduleStack!"

        # If sort:    # to be added later

        self.__eventstack = Stack(eventlist)
        self.__timestack  = Stack(timelist)

    # end of __init__

# ------------------------------------------------------------------------------

    def put_event(self, eventtype, eventtime):
        """
        Method used to place an event in the event schedule. The event is placed 
        in temporal order in the synchronized stacks eventstack and timestack. 
        """

        # Place in correct time order (what push and unshift returns - the 
        # number of  elements in the Stack After the "putting" - is not needed 
        # for anything and it is OK to do as below)

        if not self.__timestack or eventtime >= self.__timestack[-1]: # Put last
            self.__timestack.push(eventtime)
            self.__eventstack.push(eventtype)
            #self.__timestack  = [eventtime] # Does not work - turns the 
            #self.__eventstack = [eventtype] # Stack into a list again
    
        elif eventtime < self.__timestack[0]:                # Put first
            self.__timestack.unshift(eventtime)
            self.__eventstack.unshift(eventtype)

        else:
            index = bisect(self.__timestack, eventtime)
            self.__timestack.splice(index, 0, eventtime)
            self.__eventstack.splice(index, 0, eventtype)

        return eventtime  # For symmetry with put_event of EventSchedule

    # end of put_event

# ------------------------------------------------------------------------------

    def show_next_event(self):
        """
        Just look at the next event without touching the stack. 
        """

        #if not equal_length(self.__eventstack, self.__timestack):
        # is not needed due to put_event

        try:
            nextevent = self.__eventstack[0]
        except IndexError:
            nextevent = None

        try:
            nexttime  = self.__timestack[0]
        except IndexError:
            nexttime  =  None

        return nextevent, nexttime

    # end of show_next_event

# ------------------------------------------------------------------------------

    def get_next_event(self):
        """
        Method used to get the next stored event (the first in time) from the 
        event schedule. The synchronized stacks eventstack and timestack are 
        shifted in Perl fashion - i. e. the element that is returned is also 
        removed from the stack. Returns None if the stacks are empty. 
        """

        #if not equal_length(self.__eventstack, self.__timestack):
        # is not needed due to put_event

        nextevent = self.__eventstack.shift()   # None if empty
        nexttime  = self.__timestack.shift()    # None if empty

        return nextevent, nexttime

    # end of get_next_event

# ------------------------------------------------------------------------------

    def zap_events(self):
        """
        Empty the schedule to allow for a restart. Return the length of 
        the stack as it was before zapping.
        """

        #if not equal_length(self.__eventstack, self.__timestack):
        # is not needed due to put_event

        length = self.__eventstack.zap()
        length = self.__timestack.zap()

        return length

    # end of zap_events

# ------------------------------------------------------------------------------

# end of EventScheduleStack

# ------------------------------------------------------------------------------
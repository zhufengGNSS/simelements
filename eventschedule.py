# eventschedule.py
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

from misclib.heap     import Heap
from machdep.machnum  import TWOMACHEPS, MINFLOAT

# ------------------------------------------------------------------------------

class EventSchedule:
    """
    The class defines a schedule of events in a general discrete-event 
    simulation. The event times are kept in a heap; the event types are 
    kept in a corresponding dictionary. Safest is, of course, to assure 
    there are no ties among the event times.

    EventSchedule is notably more efficient (=faster) than EventScheduleStack 
    despite the crudeness of its implementation (might be better having an 
    event type list shadowing the event time heap in a specific implementation 
    of the Heap class that does not use the built-in heapq library, but this 
    has not been tried). But the two classes are otherwise equivalent in 
    principle. All the methods of the Stack class are available for handling 
    the event types via EventScheduleStack, a feature which may be used for 
    creating subclasses to EventScheduleStack which can handle more complex 
    types of schedules than the ones handled by the dict-based EventSchedule. 
    But EventSchedule should normally be preferred!.
    
    NB. TIES are only handled approximately in EventSchedule. The put_event 
    method looks for ties in the eventtimes/eventtypes dictionary. If a tie 
    is found, 2*(machine epsilon)*abs(eventtime) + minimum float > 0 will be 
    added to eventtime before placing it in the dict. If perfect handling of 
    ties is required, then EventScheduleStack must be used!
    """
# ------------------------------------------------------------------------------

    def __init__(self, eventlist=[], timelist=[], sort=False):
        """
        Creates a heap for the event times and a dictionary to keep track of 
        of the corresponding events. The events could for instance be described 
        by strings. The event times are (of course) floating-point numbers. 
        The heap and the dict can be filled here but the input lists must be 
        synchronized but not necessarily created in time order - sort=True will 
        turn the input time list into a bona fide heap and must be used if the 
        inputs are not sorted beforehand.
        """

        n = len(eventlist)
        assert len(timelist) == n, \
                     "input lists are of unequal length in EventSchedule!"

        # First create an eventtype dictionary with the eventtimes as keys:
        self.__eventsdict = {}
        for k in range(0, n): self.__eventsdict[timelist[k]] = eventlist[k]

        # Then create a heap from the time list (sorted if so requested):
        self.__timeheap = Heap(timelist)
        if sort: self.__timeheap.sort()  # sort into a bona fide heap

    # end of __init__

# ------------------------------------------------------------------------------

    def put_event(self, eventtype, eventtime):
        """
        Add an event to the schedule: place it in the eventtype/eventtime 
        dictionary and heap. 
        """

        eventtim  = eventtime
        delta     = TWOMACHEPS*abs(eventtim) + MINFLOAT
        
        while True:   # Take care of ties!
            if (eventtim in self.__eventsdict):
                eventtim += delta
            else:
                break
            
        self.__eventsdict[self.__timeheap.push(eventtim)] = eventtype
        # Does it all in this nice construct due to the way the push 
        # method works (it returns eventtim)

        # CF above
        #self.__eventsdict[eventtim] = eventtype
        #self.__timeheap.push(eventtim)

        return eventtim

    # end of put_event

# ------------------------------------------------------------------------------

    def get_next_event(self):
        """
        Get the next event (event type, event time) and remove it from 
        the schedule. 
        """

        try:
            nexttime = self.__timeheap.shift()
        except IndexError:
            nexttime = None

        try:
            nextevent = self.__eventsdict[nexttime]
            del self.__eventsdict[nexttime]
        except KeyError:
            nextevent = None

        return nextevent, nexttime

    # end of get_next_event

# ------------------------------------------------------------------------------

    def show_next_event(self):
        """
        Just look at the next event in the schedule (event type, event time) 
        without touching! 
        """

        try:
            nexttime = self.__timeheap[0]
        except IndexError:
            nexttime = None

        try:
            nextevent = self.__eventsdict[nexttime]
        except KeyError:
            nextevent = None

        return nextevent, nexttime

    # end of show_next_event

# ------------------------------------------------------------------------------

    def zap_events(self):
        """
        Empty the schedule to allow for a restart. Return the length of 
        the heap as it was before zapping.
        """

        length = self.__timeheap.zap()
        self.__eventsdict = {}

        return length

    # end of zap_events

# ------------------------------------------------------------------------------

# end of EventSchedule

# ------------------------------------------------------------------------------
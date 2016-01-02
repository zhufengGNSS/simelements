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

from eventschedule    import EventSchedule
from abcline          import ABCLine
from machdep.machnum  import TWOMACHEPS, MINFLOAT

# ------------------------------------------------------------------------------

class Line(EventSchedule, ABCLine):
    """
    Class used for handling physical queues in discrete-event simulation 
    (it might have been called "Queue" but Queue is a class in Python's 
    library...). Line inherits from the EventSchedule class but adds the 
    organization of queues/lines including balking, reneging and the 
    registration of waiting and service times using deque objects from 
    the misclib.Deque class together with dicts and 'd' arrays. 
    
    Besides the output from the methods of the class (including those of 
    EventSchedule), the following attributes are available externally from 
    each instance (but not directly assignable) at all times:
      instance.narriv      # Accumulated number of arrivers until present time
      instance.ninline     # Present number waiting in line
      instance.nfreeserv   # Present number of free servers
      instance.nbalked     # Accumulated number of balkers
      instance.nreneged    # Accumulated number of renegers  
      instance.nescaped    # = instance.nbalked + instance.nreneged
    
    Line is notably more efficient (=faster) than LineHeap and LineStack, 
    but the classes are otherwise equivalent in principle. But just like 
    EventScheduleHeap and EventScheduleStack are more general than 
    EventSchedule, LineHeap and LineStack may be used when there are arrival 
    time ties or when more complex queueing situations must be handled.
    
    Multiple queues in parallel may be handled using multiple line objects 
    and may be handled by using separate event schedules or one single event 
    schedule, depending on what seems best in the situation at hand. Jockeying 
    between queues/lines may be handled by letting customers renege from one 
    queue/line and subsequently arrive at another. Special, separate care must 
    be taken to record the  t o t a l  waiting time for jockeys.

    NB  An excellent feature of Python allows you to add new attributes to 
    an object dynamically, so you are free to add your own data structures 
    to a Line object to suit your needs in a given situation!

    This class only adds the stuff that is specific to the Line class as 
    compared to the LineHeap and LineStack classes. Everything else is 
    inherited from the ABCLine abstract base class. Always consult the 
    docstring documentation of ABCLine before using this class!!
    """
# ------------------------------------------------------------------------------

    def __init__(self, nserv, eventlist=[], timelist=[], sort=False):
        """
        Creates a heap for the event times and a dictionary to keep track
        of the corresponding events. nserv is the initial number of servers. 
        The events could for instance be desribed by strings. The times are 
        (of course) floating-point numbers. The two input lists (if there are 
        any) must be synchronized but not necessarily input in time order 
        (will be sorted if sort=True). 

        Creates deques, dicts and lists for keeping track of the attributes 
        associated with the line/queue object.
        """

        # First take care of what is inherited:
        EventSchedule.__init__(self, eventlist, timelist, sort)
        ABCLine.__init__(self, nserv)

# ------------------------------------------------------------------------------

    def prepare_to_renege(self, arrivaltime, renevent, drentime):
        """
        Used for all non-balkers when all servers are busy - if reneging is 
        treated at all. The input 'drentime' is the time endured waiting in 
        line before reneging and should be drawn from the appropriate
        probability distribution. 
        
        THE EVENT IS PLACED IN THE EVENT SCHEDULE WITH renevent AS THE EVENT 
        TYPE AND drentime+arrivaltime AS THE EVENT (CLOCK) TIME! 
        
        Ties are handled in a way so that the time placed in the event schedule 
        is incremented by TWOMACHEPS*(drentime+arrivaltime) + MINFLOAT for each 
        consecutive tie.
        """

        arentime = drentime + arrivaltime # Conversion to clock time (abs time)
        delta    = TWOMACHEPS*abs(arentime) + MINFLOAT
        
        while True:   # Take care of ties!
            if (arentime in self.renegers):   # A tie!!!
                arentime += delta
            else:
                break
        
        arentime = self.put_event(renevent, arentime) 
        self.renegers[arentime]  = arrivaltime  # Reneged at 'arentime' has 
                                                # arrived at 'arrivaltime'

    # end of prepare_to_renege

# ------------------------------------------------------------------------------

# end of Line

# ------------------------------------------------------------------------------
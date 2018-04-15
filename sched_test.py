import sched
import time
from datetime import datetime, timedelta

scheduler   = sched.scheduler(timefunc=time.time)

### functions ###

def saytime():
    print(time.ctime())
    scheduler.enter(10, priority=0, action=saytime)

### script ###

saytime()

try:
    scheduler.run(blocking=True)
except KeyboardInterrupt:
    print('Stopped.')
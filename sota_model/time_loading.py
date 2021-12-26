"""
Util file for timer methods in an encapsulated way.
"""

import time

def start_timer():
    """
    Starts the timer. 
    Returns the starting time.
    """
    return time.time()

def end_timer():
    """
    Ends the time.
    Returns the ending time.
    """
    return time.time()

def time_for_report(start, end, raw=False):
    """
    Computes the time for the report.
    Returns a string of the time to be stored in the report.
    
    Parameters:
        start: float
            Starting time.
        end: float
            Ending time.
        raw: bool (default=False)
            Indicates wether to return the raw duration as second output.
    """
    duration = end-start
    minutes = int(duration // 60)
    hours = int(minutes // 60)
    days = int(hours // 24)
    seconds = duration % 60
    
    time_str = f"{days}days"+'-'+f"{hours:02}"+':'+f"{minutes:02}"+':'+f"{seconds:2.4}"
    if raw :
        return time_str, duration
    return time_str
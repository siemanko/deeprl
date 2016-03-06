"""Simulator

Every simulator that the algorithm can use is required to implement the following API:

class Simulator(...):
    def __init__(self, settings, record=False) -> None
        initialize using simulator settings
        record=True means that entire execution should be recorded
    def get_state(self) -> STATE
        return current simulator state
        when the execution finished it should return None
    def take_action(self, a) -> REWARD
        take action a, also update state to next state and return reward
    def evaluation_metrics(self):
        returns a dictionary of pairs metric_name:metric_value
        where metric_name is a string and metric_value is a float
    def save_recording(self,directory):
        it should assert that record was passed as true in __init__
        save_recording of current execution in a directory named dir
"""

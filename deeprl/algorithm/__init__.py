"""Model

Every model that the algorithm can use is required to implement the following API:

class Model(...):
    def __init__(self, settings) -> None
        initialize using model settings
        session is a tensorflow session to use
    def iteration(make_simulator)
    def action(self, STATE) -> ACTION
    def save(self, directory)
    def load(self, directory))
    def iteration_metrics() -> {str key: float value}
        returns a dictionary of key metrics describing the model's learning behavior so far.
"""

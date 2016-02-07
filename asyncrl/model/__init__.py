"""Model

Every model that the algorithm can use is required to implement the following API:

class Model(...):
    def __init__(self, settings) -> None
        initialize using model settings

    def action(self, STATE) -> ACTION
    def value(self, STATE) -> NUMBER

    def update_grads(self, R, s, a) -> None
    def get_grads(self) -> GRADS
    def reset_grads(self)

    def apply_grads(self, GRADS)
    def get_params(self) -> PARAMS
    def set_params(self, PARAMS)

    def save(self, directory)
    def load(self, directory)
"""

from .enc_dec import EncDec

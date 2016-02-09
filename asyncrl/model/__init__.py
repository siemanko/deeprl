"""Model

Every model that the algorithm can use is required to implement the following API:

class Model(...):
    def __init__(self, settings, session) -> None
        initialize using model settings
        session is a tensorflow session to use
    def action(self, STATE) -> ACTION
    def value(self, STATE) -> NUMBER

    def update_gradients(self, R, s, a) -> None
    def eval_gradients(self) -> GRADS
    def reset_gradients(self)

    def apply_gradients(self, GRADS)
    def eval_parameters(self) -> PARAMS
    def set_parameters(self, PARAMS)

    def save(self, directory)
    def load(self, directory)
"""

from .enc_dec import EncDec

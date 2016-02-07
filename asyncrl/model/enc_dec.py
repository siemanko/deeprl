import tensorflow as tf


from .blocks import parse_block

class EncDec(object):
    def __init__(self, settings):
        self.create_variables(settings)

    def create_variables(self, settings):
        self.state_encoder  = parse_block(settings['state_encoder'])
        self.action_decoder = parse_block(settings['action_decoder'])
        self.value_decoder  = parse_block(settings['value_decoder'])

    def action(self, STATE):
        return [0,]

    def update_grads(self, R, s, a):
        pass

    def apply_grads(self, GRADS):
        pass

    def get_params(self):
        pass

    def set_params(self, PARAMS):
        pass

    def save(self, directory):
        pass

    def load(self, directory):
        pass

import tensorflow as tf
import random

from .blocks import parse_block, SequenceWrapper

class EncDec(object):
    def __init__(self, settings, session):
        self.s = session

        self.action_type = settings["action"]["type"]
        if self.action_type == "discrete":
            self.num_actions = settings["action"]["num_actions"]
        else:
            assert False, "Unknown action type:" % (self.action_type,)

        self.create_variables(settings)
        self.s.run(tf.initialize_variables(self.variables()))

    def create_variables(self, settings):
        self.state_encoder  = parse_block(settings['state_encoder'])
        self.action_decoder = parse_block(settings['action_decoder'])
        self.value_decoder  = parse_block(settings['value_decoder'])

        self.action_network = SequenceWrapper([self.state_encoder, self.action_decoder],
                                              scope="action_network")
        self.value_network  = SequenceWrapper([self.state_encoder, self.value_decoder],
                                              scope="value_network")

        self.state        = self.state_encoder.input_placeholder()
        self.action_probs = self.action_network(self.state)
        self.action_id    = tf.argmax(self.action_probs, dimension=1)

        self.value        =  self.value_network(self.state)

    def action(self, state, exploration=0.0):
        if random.random() < exploration:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.s.run(self.action_id, {
                self.state: state,
            })

    def value(self, state):
        return self.s.run(self.value, {
            self.state: state,
        })

    def update_grads(self, R, s, a):
        pass

    def apply_grads(self, GRADS):
        pass

    def variables(self):
        return (
            self.state_encoder.variables() +
            self.action_decoder.variables() +
            self.value_decoder.variables()
        )

    def get_params(self):
        pass

    def set_params(self, PARAMS):
        pass

    def save(self, directory):
        pass

    def load(self, directory):
        pass

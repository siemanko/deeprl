import numpy as np
import random
import tensorflow as tf

from .blocks import (
    parse_block,
    parse_optimizer,
    SequenceWrapper,
)
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
        self.s.run(tf.initialize_variables(self.gradients()))

    def create_variables(self, settings):
        self.network_names = [
            'state_encoder',
            'action_decoder',
            'value_decoder',
        ]

        #### CREATE ALL THE NETWORKS
        self.networks = {
            name:parse_block(settings['networks'][name])
            for name in self.network_names
        }

        #### CREATE ALL THE OPTIMIZERS
        self.optimizers   = {
            name:parse_optimizer(settings['optimizers'][name])
            for name in self.network_names
        }

        #### CREATE VARIABLES TO STORE GRADIENTS
        self.net_grads = {}
        for n in self.network_names:
            self.net_grads[n] = [
                tf.Variable(tf.zeros_like(v), name=v.name.split(':')[0]+"_grad")
                for v in self.networks[n].variables()
            ]

        #### CREATE COMBINED NETWORK: state -> action
        self.action_network = SequenceWrapper(
            [self.networks["state_encoder"], self.networks["action_decoder"]],
            scope="action_network")

        #### CREATE COMBINED NETWORK: state -> state_value
        self.value_network = SequenceWrapper(
            [self.networks["state_encoder"], self.networks["value_decoder"]],
            scope="value_network")

        #### COMPUTE STATE VALUE AND ACTION
        self.state        = self.networks["state_encoder"].input_placeholder()
        self.action_probs = self.action_network(self.state)
        self.action_id    = tf.argmax(self.action_probs, dimension=1)

        self.value        =  self.value_network(self.state)


        #### COMPUTE ACTOR UPDATE
        self.reward             = tf.placeholder(tf.float32, (None,))
        self.chosen_action_id   = tf.placeholder(tf.int64, (None,))

        self.advantage          = self.reward - tf.stop_gradient(self.value)
        self.onehot             = tf.constant(np.diag(
                np.ones((self.num_actions,), dtype=np.float32)))
        self.chosen_action_mask = tf.nn.embedding_lookup(self.onehot, self.chosen_action_id)
        self.chosen_action_prob = tf.reduce_sum(self.action_probs * self.chosen_action_mask, 1)
        self.actor_loss         = - tf.log(self.chosen_action_prob) * self.advantage
        self.update_actor_grads = tf.group(*[
            self.update_network_grads('state_encoder', self.actor_loss),
            self.update_network_grads('action_decoder', self.actor_loss),
        ])

        #### COMPUTE VALUE NETWORK UPDATE
        self.value_loss         = tf.square(self.reward - self.value)
        self.update_value_grads = tf.group(*[
            self.update_network_grads('state_encoder', self.value_loss),
            self.update_network_grads('value_decoder', self.value_loss),
        ])

        #### OP TO RESET GRADIENTS
        self.reset_gradients_op = tf.group(*[tf.assign(g, g.initialized_value())
                                             for g in self.gradients() ])

        #### SET GRADIENTS
        self.gradient_placeholders = [tf.placeholder(tf.float32)
                                      for _ in self.gradients()]
        self.set_gradients_op = tf.group(*[
            tf.assign(g, gp)
            for g, gp in zip(self.gradients(), self.gradient_placeholders)
        ])

        ### SET PARAMETERS
        self.variable_placeholders = [tf.placeholder(tf.float32)
                                      for _ in self.variables()]
        self.set_variables_op = tf.group(*[
            tf.assign(g, gp)
            for g, gp in zip(self.variables(), self.variable_placeholders)
        ])

        ### APPLY GRADIENTS USING OPTIMIZERS
        self.apply_gradients_op = tf.group(*[
            self.apply_network_gradients(n)
            for n in self.network_names
        ])

    def variables(self):
        result = []
        for n in self.network_names:
            result.extend(self.networks[n].variables())
        return result

    def gradients(self):
        result = []
        for n in self.network_names:
            result.extend(self.net_grads[n])
        return result

    def apply_network_gradients(self, network):
        grad_values = [g.value() for g in self.net_grads[network]]
        grad_var = list(zip(grad_values, self.networks[network].variables()))
        return self.optimizers[network].apply_gradients(grad_var)

    def update_network_grads(self, network, loss):
        partial_grads = tf.gradients(loss, self.networks[network].variables())
        grads = self.net_grads[network]
        return tf.group(*[tf.assign_add(g, pg) for g, pg in zip(grads, partial_grads)])


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

    def update_gradients(self, R, s, a):
        self.s.run(self.update_actor_grads, {
            self.state: state,
            self.chosen_action_id: a,
            self.reward: R
        })
        self.s.run(self.update_value_grads, {
            self.state: state,
            self.reward: R
        })

    def eval_gradients(self):
        result = []
        for g in self.gradients():
            result.append(self.s.run(g))
        return result

    def reset_gradients(self):
        self.s.run(self.reset_gradients_op)

    def apply_gradients(self, gradient_values):
        self.s.run(self.set_gradients_op, {
            gp: gv
            for gp,gv in zip(self.gradient_placeholders, gradient_values)
        })

        self.s.run(self.apply_gradients_op)

    def eval_parameters(self):
        result = []
        for g in self.variables():
            result.append(self.s.run(g))
        return result

    def set_parameters(self, parameter_values):
        self.s.run(self.set_variables_op, {
            gp: gv
            for gp,gv in zip(self.variable_placeholders, parameter_values)
        })

    def save(self, directory):
        raise NotImplemented()

    def load(self, directory):
        raise NotImplemented()

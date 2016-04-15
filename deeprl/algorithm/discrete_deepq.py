import numpy as np
import random
import tensorflow as tf

from collections import deque, namedtuple

from .utils import (
    linear_annealing,
    onehot_encode,
    none_mask
)
from deeprl.model.blocks import parse_block, parse_optimizer
from deeprl.settings import update_settings

# TODO(szymon):
# 1. observation -> state
# 2. add summary writer

DEFAULT_SETTINGS = {
    'exploration_period'         : 10000,
    'random_action_probability'  : 0.05,
    'discount_rate'              : 0.95,
    'store_every_nth'            : 1,
    'train_every_nth'            : 1,
    'minibatch_size'             : 30,
    'replay_buffer_size'         : 10000,
    'target_network_update_rate' : 10000,
}

Memory = namedtuple("Memory", ("state", "action", "reward", "new_state"))

class DiscreteDeepQ(object):
    def __init__(self, settings):
        self.settings       = update_settings(DEFAULT_SETTINGS, settings)

        # network and training
        self.q_network = parse_block(settings["model"])
        self.optimizer = parse_optimizer(settings["optimizer"])

        out_sh = self.q_network.output_shape()
        assert len(out_sh) == 2 and out_sh[0] is None, \
                "Output of the Discrete DeepQ must be (None, num_actions), where None corresponds to batch_size"
        self.num_actions      = out_sh[1]
        self.minipatch_size   = self.settings["minibatch_size"]

        self.train_every_nth              = self.settings['train_every_nth']
        self.discount_rate    = self.settings["discount_rate"]

        self.transitions_so_far        = 0
        self.exploration_period        = self.settings['exploration_period']
        self.random_action_probability = self.settings['random_action_probability']

        self.replay_buffer                = deque()
        self.store_every_nth              = self.settings['store_every_nth']
        self.replay_buffer_size           = self.settings['replay_buffer_size']

        self.target_network_update_rate   = self.settings['target_network_update_rate']

        self.summary_writer = None

        self.s = tf.Session()

        self.create_variables()
        self.s.run(tf.initialize_variables(
                self.q_network.variables() + self.target_q_network.variables()))


    def action(self, observation, exploration=False):
        """Given observation returns the action that should be chosen using
        DeepQ learning strategy. Does not backprop."""
        assert observation.shape[0] == 1, \
                "Action is performed based on single observation."

        if exploration and random.random() < self.exploration_probability():
            return [random.randint(0, self.num_actions - 1)
                    for _ in range(observation.shape[0])]
        else:
            return self.s.run(self.predicted_actions, {self.observation: observation})

    def iteration(self, make_simulator):
        simulator = make_simulator(record=False)

        state = simulator.observe()
        while not simulator.is_terminal():
            action = self.action(state, exploration=True)
            reward = simulator.act(action)
            new_state = simulator.observe()

            self.transitions_so_far += 1
            if self.transitions_so_far % self.store_every_nth == 0:
                self.store(state, action, reward, new_state)
            if self.transitions_so_far % self.train_every_nth == 1:
                self.training_step()

            state = new_state

    def save_model(self, directory):
        # make sure to save transitions_so_far to preserve trianing
        # also --continue
        raise NotImplemented()

    def load_model(self, directory):
        raise NotImplemented()

    def exploration_probability(self):
        """Probability that the next action should be selected at random"""
        return linear_annealing(self.transitions_so_far,
                                self.exploration_period,
                                1.0,
                                self.random_action_probability)

    def create_variables(self):
        self.target_q_network    = self.q_network.copy(scope="target_network")

        # FOR REGULAR ACTION SCORE COMPUTATION
        with tf.name_scope("taking_action"):
            self.observation        = self.q_network.input_placeholder("observation")
            self.action_scores      = tf.identity(self.q_network(self.observation), name="action_scores")
            tf.histogram_summary("action_scores", self.action_scores)
            self.predicted_actions  = tf.argmax(self.action_scores, dimension=1, name="predicted_actions")

        with tf.name_scope("estimating_future_rewards"):
            # FOR PREDICTING TARGET FUTURE REWARDS
            self.next_observation          = self.q_network.input_placeholder("next_observation")
            self.next_observation_mask     = tf.placeholder(tf.float32,
                                                            (None,),
                                                            name="next_observation_mask")
            self.next_action_scores        = self.target_q_network(self.next_observation)

            tf.histogram_summary("target_action_scores", self.next_action_scores)
            self.rewards                   = tf.placeholder(tf.float32, (None,), name="rewards")
            target_values                  = \
                    tf.reduce_max(self.next_action_scores, reduction_indices=[1,]) * self.next_observation_mask
            self.future_rewards            = self.rewards + self.discount_rate * target_values

        with tf.name_scope("q_value_precition"):
            # FOR PREDICTION ERROR
            self.action_mask                = tf.placeholder(tf.float32,
                                                              self.q_network.output_shape(),
                                                              name="action_mask")
            self.masked_action_scores       = tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1,])
            temp_diff                       = self.masked_action_scores - self.future_rewards
            self.prediction_error           = tf.reduce_mean(tf.square(temp_diff))
            gradients                       = self.optimizer.compute_gradients(
                                                    self.prediction_error,
                                                    var_list=self.q_network.variables())
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 5), var)
            # Add histograms for gradients.
            for grad, var in gradients:
                tf.histogram_summary(var.name, var)
                if grad is not None:
                    tf.histogram_summary(var.name + '/gradients', grad)
            self.train_op                   = self.optimizer.apply_gradients(gradients)

        # UPDATE TARGET NETWORK
        with tf.name_scope("target_network_update"):
            self.target_network_update = []
            for v_source, v_target in zip(self.q_network.variables(), self.target_q_network.variables()):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.target_network_update_rate * (v_target - v_source))
                self.target_network_update.append(update_op)
            self.target_network_update = tf.group(*self.target_network_update)

        # summaries
        tf.scalar_summary("prediction_error", self.prediction_error)

        self.summarize = tf.merge_all_summaries()
        self.no_op1    = tf.no_op()




    def store(self, observation, action, reward, newobservation):
        """Store experience, where starting with observation and
        execution action, we arrived at the newobservation and got thetarget_network_update
        reward reward
        If newstate is None, the state/action pair is assumed to be terminal
        """
        self.replay_buffer.append(Memory(observation, action, reward, newobservation))
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.popleft()


    def training_step(self):
        """Pick a self.minibatch_size exeperiences from reply buffer
        and backpropage the value function.
        """
        if len(self.replay_buffer) <  self.minibatch_size:
            return

        # sample experience.
        samples   = random.sample(range(len(self.replay_buffer)), self.minibatch_size)
        samples   = [self.replay_buffer[i] for i in samples]

        # batch states
        states     = self.q_network.batch_inputs([s.state for s in samples])
        new_states = self.q_network.batch_inputs([s.new_state for s in samples])
        action_mask    = onehot_encode([s.action for s in samples], self.num_actions)
        newstates_mask = none_mask([s.new_state for s in samples])
        rewards        = np.array([s.new_state for s in samples], dtype=np.float32)

        calculate_summaries = self.transitions_so_far % 100 == 0 and \
                self.summary_writer is not None

        cost, _, summary_str = self.s.run([
            self.prediction_error,
            self.train_op,
            self.summarize if calculate_summaries else self.no_op1,
        ], {
            self.observation:            states,
            self.next_observation:       newstates,
            self.next_observation_mask:  newstates_mask,
            self.action_mask:            action_mask,
            self.rewards:                rewards,
        })

        self.s.run(self.target_network_update)

        if calculate_summaries:
            self.summary_writer.add_summary(summary_str, self.iteration)

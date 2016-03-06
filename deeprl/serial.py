import tensorflow as tf
import numpy as np

from itertools import count

from deeprl.model.blocks import parse_optimizer
from .utils import init_experiment, make_session

def make_apply_gradients_fun(settings, model):
    #### CREATE ALL THE OPTIMIZERS
    optimizers   = {
        name:parse_optimizer(settings['model']['settings']['optimizers'][name])
        for name in settings['model']['settings']['optimizers']
    }

    update_ops = []
    for var, grad in zip(model.variables(), model.gradients()):
        var_optimizer = None
        for optimizer_name, optimizer in optimizers.items():
            if var.name.startswith(optimizer_name):
                var_optimizer = optimizer
                break
        if optimizer is None:
            raise Exception("Could not match optimizer for variable %s" % (var.name))

        update_op = optimizer.apply_gradients([(grad.value(), var)])
        update_ops.append(update_op)
    combined_update_op = tf.group(*update_ops)

    return lambda: model.get_session().run(combined_update_op)


def serial_mode(settings):
    session = make_session() # parallel session
    model, make_simulator = init_experiment(settings, session)

    steps_before_update = settings["training"]['steps_before_update']
    gamma               = settings["training"]['gamma']

    apply_gradients = make_apply_gradients_fun(settings, model)


    while True:
        simulator = make_simulator()
        state = simulator.get_state()
        while state is not None:
            transitions = []
            for _ in range(steps_before_update):
                action = model.action(state)
                reward = simulator.take_action(action)
                transitions.append((state, action, reward))

                state = simulator.get_state()
                if state is None: break
            value = model.value(state) if state is not None else np.array([0.0,])
            for s, a, r in reversed(transitions):
                value = r + gamma * value
                model.update_gradients(s, a, value)
            apply_gradients()

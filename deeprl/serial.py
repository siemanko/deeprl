import numpy as np
import os
import tensorflow as tf
import time

from collections import defaultdict
from itertools import count

from deeprl.model.blocks import parse_optimizer
from .utils import init_experiment, make_session, ensure_directory
from .record import capture_metrics

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

    save_dir                 = settings['__runtime__']['savedir']
    ensure_directory(save_dir)
    eval_dir                 = os.path.join(save_dir, "evaluation")
    ensure_directory(eval_dir)
    stats_file               = os.path.join(eval_dir, "stats.tsv")
    steps_before_update      = settings["training"]['steps_before_update']
    gamma                    = settings["training"]['gamma']
    time_between_evaluations = settings["evaluation"]["time_between_evaluations"]
    runs_per_evaluation      = settings["evaluation"]["runs_per_evaluation"]

    apply_gradients = make_apply_gradients_fun(settings, model)

    last_metrics_capture = 0

    while True:
        simulator = make_simulator()
        state = simulator.observe()
        while state is not None:
            transitions = []
            for _ in range(steps_before_update):
                action = model.action(state)
                reward = simulator.act(action)
                transitions.append((state, action, reward))

                state = simulator.observe()
                if simulator.is_terminal():
                    break
            value = model.value(state) if state is not None else np.array([0.0,])
            for s, a, r in reversed(transitions):
                value = r + gamma * value
                model.update_gradients(s, a, value)
            apply_gradients()

            if last_metrics_capture + time_between_evaluations < time.time():
                print("Evaluation...")
                combined_metrics = defaultdict(lambda: [])
                for _ in range(runs_per_evaluation):
                    metrics = capture_metrics(model, make_simulator)
                    print (metrics)
                    for key, value in metrics.items():
                        combined_metrics[key].append(value)
                with open(stats_file, "a+") as f:
                    ts = time.time()
                    for metric, values in combined_metrics.items():
                        f.write("%f %s %f\n" % (ts, metric, np.mean(values)))
                last_metrics_capture = time.time()

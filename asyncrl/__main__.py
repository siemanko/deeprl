import argparse
import json
import os
import sys
import tempfile
import tensorflow as tf

from .utils import import_class

def init_experiment(settings, session, record=False):
    simulator_class = import_class(settings['simulator']['class'])
    simulator       = simulator_class(settings['simulator']['settings'], record)

    model_class = import_class(settings['model']['class'])
    model       = model_class(settings['model']['settings'], session)

    return model, simulator

def make_session(max_cpu_cores=None):
    """Makes a multi-core session.
    If max_cpu_cores is None, it adopts the number of cores
    automatically
    """
    config = tf.ConfigProto()

    if max_cpu_cores is not None:
        configdevice_count.update({'CPU': max_cpu_cores})

    return tf.Session(config=config)

def create_recording(model, simulator, dir_name):
    reward = 0.0
    while reward is not None:
        state  = simulator.get_state()
        action = model.action(state)
        reward = simulator.take_action(action)

    simulator.save_recording(dir_name)


def add_boolean_flag(parser, name, default):
    cmd_style = name.replace('_', '-')
    parser.add_argument('--%s' % (cmd_style,), dest=name, action='store_true')
    parser.add_argument('--no-%s' % (cmd_style,), dest=name, action='store_false')
    parser.set_defaults(**{name:default})

def parse_args():
    parser = argparse.ArgumentParser(description='Asynchronous Deep Reinforcement Learning training procedure.')

    parser.add_argument('--experiment', '-e', type=str, required=True, help="Location of json file which specifies the experiment")
    parser.add_argument('--mode',       '-m', type=str, default="train", choices=["train", "record"], help="What should we do today? Train? Record execution trace?")

    return parser.parse_args()

def main(args):
    with open(args.experiment) as f:
        settings = json.load(f)

    if args.mode == 'record':
        dir_name = tempfile.mkdtemp(prefix="recording_", dir=os.getcwd())
        print ("Recording will be saved at:\n%s\n" % (dir_name,), flush=True)

        session = make_session() # parallel session
        model, simulator = init_experiment(settings, session, record=True)
        create_recording(model, simulator, dir_name)
    elif args.mode == 'train':
        print("Training")
    else:
        assert False



if __name__ == '__main__':
    args = parse_args()
    main(args)

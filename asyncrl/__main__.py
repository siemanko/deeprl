import argparse
import json
import os
import sys
import tempfile

def import_class(path):
    path_split = path.split('.')
    module_name, class_name = '.'.join(path_split[:-1]), path_split[-1]
    module = __import__(module_name, fromlist=(class_name,))
    return getattr(module, class_name)

def init_experiment(experiment, record=False):
    simulator_class = import_class(experiment['simulator']['class'])
    simulator       = simulator_class(experiment['simulator']['settings'], record)

    model_class = import_class(experiment['model']['class'])
    model       = model_class(experiment['model']['settings'])

    return model, simulator

def create_recording(experiment, dir_name):
    model, simulator = init_experiment(experiment, record=True)

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
        experiment = json.load(f)

    if args.mode == 'record':
        dir_name = tempfile.mkdtemp(prefix="recording_", dir=os.getcwd())
        print ("Recording will be saved at:\n%s\n" % (dir_name,), flush=True)
        create_recording(experiment, dir_name)
    elif args.mode == 'train':
        print("Training")
    else:
        assert False



if __name__ == '__main__':
    args = parse_args()
    main(args)

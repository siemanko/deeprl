import argparse
import json
import os
import sys
import tempfile
import time

from .parallel import parallel_mode
from .record import record_mode
from .serial import serial_mode
from .settings import DEFAULT_SETTINGS, update_settings



def add_boolean_flag(parser, name, default):
    cmd_style = name.replace('_', '-')
    parser.add_argument('--%s' % (cmd_style,), dest=name, action='store_true')
    parser.add_argument('--no-%s' % (cmd_style,), dest=name, action='store_false')
    parser.set_defaults(**{name:default})

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning.')

    parser.add_argument('--experiment', '-e', type=str, required=True,
            help="Location of json file which specifies the experiment")
    parser.add_argument('--mode',       '-m', type=str, default="train",
            choices=["parallel", "serial", "distributed", "record"],
            help="What should we do today? Train? Record execution trace?")
    parser.add_argument('--savedir',    '-s', type=str, default=None,
            help="Where to store the data related to the experiment. If the same folder is used multiple times the experiment will be resumed.")

    return parser.parse_args()



def parse_savedir(args, settings):
    savedir = args.savedir
    if savedir is None:
        savedir = tempfile.mkdtemp(prefix="saved_", dir=os.getcwd())
    print ("Results will be saved at %s." % (savedir))
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    settings['__runtime__']['savedir'] = savedir

def main(args):
    settings = DEFAULT_SETTINGS
    with open(args.experiment) as f:
        settings = update_settings(settings, json.load(f))
    settings['__runtime__'] = {}
    parse_savedir(args, settings)

    if args.mode == 'record':
        record_mode(settings)
    elif args.mode == 'parallel':
        parallel_mode(settings)
    elif args.mode == 'serial':
        serial_mode(settings)
    else:
        assert False



if __name__ == '__main__':
    args = parse_args()
    main(args)

import argparse

from .run import run

def add_boolean_flag(parser, name, default):
    cmd_style = name.replace('_', '-')
    parser.add_argument('--%s' % (cmd_style,), dest=name, action='store_true')
    parser.add_argument('--no-%s' % (cmd_style,), dest=name, action='store_false')
    parser.set_defaults(**{name:default})

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning.')

    parser.add_argument('--spec', '-s', type=str, required=True,
            help="Location of json file which specifies the experiment")
    parser.add_argument('--mode',       '-m', type=str, default="train",
            choices=["train", "record"],
            help="What should we do today? Train? Record execution trace?")
    parser.add_argument('--directory',    '-d', type=str, default=None,
            help="Where to store the data related to the experiment. If the same folder is used multiple times the experiment will be resumed.")
    add_boolean_flag(parser, 'quiet', False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args.spec, args.mode, args.directory, args.quiet)

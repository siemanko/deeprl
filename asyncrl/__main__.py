import argparse
import sys
import json



def add_boolean_flag(parser, name,):
    cmd_style = name.replace('_', '-')
    parser.add_argument('--%s' % (cmd_style,), dest=name, action='store_true')
    parser.add_argument('--no-%s' % (cmd_style,), dest=name, action='store_false')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Asynchronous Deep Reinforcement Learning training procedure.')


    parser.add_argument('--experiment', '-e', type=str, help="Location of json file which specifies the experiment")

    parser.set_defaults(use_top_right_corner=True,
                        use_rectangle_in_the_middle=True,
                        print_progress=False)

    return parser.parse_args()

def main(args):
    with open(args.experiment) as f:
        experiment = json.load(f)
    print(experiment["message"])


if __name__ == '__main__':
    args = parse_args()
    main(args)

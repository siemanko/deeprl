import json
import os

from .utils import import_class

def load_algorithm(directory):
    spec_file = os.path.join(directory, 'spec.json')
    state_dir = os.path.join(directory, 'state')
    make_alg, make_sim = load_spec_from_file(spec_file)

    a = make_alg()
    a.load(state_dir)
    return a

def load_spec_from_file(json_file):
    if isinstance(json_file, str):
        with open(json_file, "rt") as f:
            return load_spec_from_json(json.load(f))
    elif isinstance(json_file, file):
        return load_spec_from_json(json.load(json_file))
    else:
        assert False

def load_spec_from_json(json_dict):
    def make_algorithm():
        algorithm_class = import_class(json_dict['algorithm']['class'])
        algorithm       = algorithm_class(json_dict['algorithm']['settings'])
        return algorithm

    def make_simulator(record=False):
        simulator_class = import_class(json_dict['simulator']['class'])
        simulator       = simulator_class(json_dict['simulator']['settings'], record)
        return simulator

    return make_algorithm, make_simulator

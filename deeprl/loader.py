import json
import os

from .utils import import_class, ensure_json

def load_algorithm(directory):
    """Loads saved algorithm from a directory"""
    spec_file = os.path.join(directory, 'spec.json')
    state_dir = os.path.join(directory, 'state')
    make_alg, make_sim = load_spec(spec_file)

    a = make_alg()
    a.load(state_dir)
    return a


def load_spec(spec):
    """Parses a spec which must be either of file, file_name or a json dict."""
    spec = ensure_json(spec)

    def make_algorithm():
        algorithm_class = import_class(spec['algorithm']['class'])
        algorithm       = algorithm_class(spec['algorithm']['settings'])
        return algorithm

    def make_simulator(record=False):
        simulator_class = import_class(spec['simulator']['class'])
        simulator       = simulator_class(spec['simulator']['settings'], record)
        return simulator

    return make_algorithm, make_simulator

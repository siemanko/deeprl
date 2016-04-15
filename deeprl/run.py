import json
import numpy as np
import os
import sys
import tempfile
import time

from collections import defaultdict
from itertools   import count

from .loader   import load_spec
from .utils    import ensure_directory, ensure_json
from .settings import update_settings

def parse_savedir(args, settings):
    savedir = args.savedir
    if savedir is None:
        savedir = tempfile.mkdtemp(prefix="saved_", dir=os.getcwd())
    print ("Results will be saved at %s." % (savedir))
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    settings['__runtime__']['savedir'] = savedir


DEFAULT_SPEC = {
    "system": {
         "time_between_evaluations" : 120,
         "runs_per_evaluation"      : 1,
         "time_between_model_saves" : 120,
         "keep_old_model_saves"     : False,
    }
}

class ExperimentRunner(object):
    def __init__(self, spec, directory, quiet):
        self.spec      = update_settings(DEFAULT_SPEC, ensure_json(spec))
        self.directory = directory
        self.quiet     = quiet

        if self.directory is None:
            self.directory = tempfile.mkdtemp(prefix="saved_", dir=os.getcwd())

        ensure_directory(self.directory)
        self.recording_directory  = os.path.join(self.directory, 'recordings')
        ensure_directory(self.recording_directory)
        self.evaluation_directory = os.path.join(self.directory, 'evaluation')
        ensure_directory(self.evaluation_directory)
        self.evaluation_tsv       = os.path.join(self.evaluation_directory, 'metrics.tsv')


        self.log("Results will be saved at %s." % (self.directory))

        self.time_between_evaluations = self.spec["system"]["time_between_evaluations"]
        self.runs_per_evaluation      = self.spec["system"]["runs_per_evaluation"]
        self.time_between_model_saves = self.spec["system"]["time_between_model_saves"]
        self.keep_old_model_saves     = self.spec["system"]["keep_old_model_saves"]



        make_alg, make_sim = load_spec(self.spec)

        self.alg = make_alg()
        # TODO(szymon): load if possible

        self.make_simulator = make_sim

    def save_alg(self):
        # remember to add spec
        raise NotImplemented()

    def log(self, msg):
        if not self.quiet:
            print(msg)

    def record(self):
        timestamp = time.strftime("%Y%m%d_%H:%M:%S")
        recording_dir = os.path.join(self.recording_directory, timestamp)
        ensure_directory(recording_dir)

        simulator = self.make_simulator(record=True)
        state  = simulator.observe()
        while not simulator.is_terminal():
            action = self.alg.action(state)
            _ = simulator.act(action)
            state = simulator.observe()

        simulator.execution_recording(recording_dir)

        latest_recording_dir = os.path.join(self.recording_directory, 'latest')

        if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            if os.path.exists(latest_recording_dir):
                os.remove(latest_recording_dir)
            os.symlink(timestamp, latest_recording_dir, target_is_directory=True)

    def evaluate(self):
        aggregated_metrics = defaultdict(lambda: [])
        for eval_run in range(self.runs_per_evaluation):
            self.log("Evaluation run %d" % (eval_run,))
            simulator = self.make_simulator(record=False)
            state  = simulator.observe()
            while not simulator.is_terminal():
                action = self.alg.action(state)
                _ = simulator.act(action)
                state = simulator.observe()
            metrics = simulator.execution_metrics()
            for key, value in metrics.items():
                aggregated_metrics[key].append(np.mean(value))

        for key, value in aggregated_metrics.items():
            aggregated_metrics[key] = np.mean(value)
        self.append_metrics(metrics)

    def append_metrics(self, metrics):
        # TODO(szymon): append to tsv
        for key, value in metrics.items():
            print("%s: %f" % (key, value))


    def train(self):
        last_evaluation_time = time.time()
        for iteration in count():
            self.log("Iteration %d" % (iteration,))
            self.alg.iteration(self.make_simulator)
            if time.time() - last_evaluation_time > self.time_between_evaluations:
                self.evaluate()
                last_evaluation_time = time.time()

    def run(self, mode):
        if mode == 'record':
            self.record()
        elif mode == 'train':
            self.train()
        else:
            assert False

def run(spec, mode, directory=None, quiet=False):
    runner = ExperimentRunner(spec,directory,quiet)
    runner.run(mode)

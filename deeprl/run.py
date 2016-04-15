import os
import sys
import tempfile
import time

from .loader import load_spec_from_file
from .utils import ensure_directory

def parse_savedir(args, settings):
    savedir = args.savedir
    if savedir is None:
        savedir = tempfile.mkdtemp(prefix="saved_", dir=os.getcwd())
    print ("Results will be saved at %s." % (savedir))
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    settings['__runtime__']['savedir'] = savedir

class ExperimentRunner(object):
    def __init__(self, spec_file, directory, quiet):
        self.spec_file = spec_file
        self.directory = directory
        self.quiet     = quiet

        if self.directory is None:
            self.directory = tempfile.mkdtemp(prefix="saved_", dir=os.getcwd())

        ensure_directory(self.directory)
        self.recording_directory = os.path.join(self.directory, 'recordings')
        ensure_directory(self.recording_directory)

        self.log("Results will be saved at %s." % (self.directory))

        make_alg, make_sim = load_spec_from_file(self.spec_file)

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

    def train(self):
        while True:
            self.log("Iteration")
            self.alg.iteration(self.make_simulator)

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

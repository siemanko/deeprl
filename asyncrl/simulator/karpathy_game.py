import os

class KarpathyGame(object):
    def __init__(self, settings, record=False):
        pass

    def get_state(self):
        pass

    def take_action(self, a):
        pass

    def evaluation_metrics(self):
        pass

    def save_recording(self, directory):
        with open(os.path.join(directory, "lol.txt"), "wt") as f:
            f.write("Guten tag!\n")

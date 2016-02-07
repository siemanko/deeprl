import os

from euclid import Vector2

from .colliding_marbles import HeroSimulator

class KarpathyGame(object):
    def __init__(self, settings, record=False):
        self.record = record
        self.fps                    = settings["fps"]
        self.frames_between_actions = settings["frames_between_actions"]
        self.max_frames             = settings["max_frames"]

        self.sim = HeroSimulator(settings)
        self.num_frames = 1

        self.actions = [Vector2(*a) for a in settings["action_acc"]]

        if self.record:
            self.recording = [self.sim.to_svg()]

    def get_state(self):
        return self.sim.observe()

    def take_action(self, a):
        self.sim.hero.speed += self.actions[a[0]]

        for _ in range(self.frames_between_actions):
            self.sim.step(1.0 / self.fps)
            self.num_frames += 1
            if self.record:
                self.recording.append(self.sim.to_svg())

        reward = 0.0

        if self.num_frames >= self.max_frames:
            return None
        else:
            return reward

    def evaluation_metrics(self):
        pass

    def save_recording(self, directory):
        assert self.record, "To create a recording pass record=True to the constructor."

        for i, frame in enumerate(self.recording):
            file_name = 'frame_%09d.svg' % (i,)

            with open(os.path.join(directory, file_name), "wt") as f:
                frame.write_svg(f)

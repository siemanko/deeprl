import os
import subprocess

from euclid import Vector2
from progress.bar import Bar

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

        frame_files = []
        bar = Bar('Saving frames', max=len(self.recording))
        for i, frame in enumerate(self.recording):
            bar.next()
            file_path_svg = os.path.join(directory, 'frame_%09d.svg' % (i,))
            file_path_png = os.path.join(directory, 'frame_%09d.png' % (i,))

            with open(file_path_svg, "wt") as f:
                frame.write_svg(f)

            subprocess.check_output(["rsvg-convert", file_path_svg, "-o", file_path_png,
                                     "-b", "white"])
            frame_files.extend([file_path_svg, file_path_png])
        bar.finish()
        print("Merging frames...", flush=True)
        subprocess.check_output(["ffmpeg", "-r", str(self.fps), "-pattern_type", "glob","-i",
                                 '*.png', "-c:v", "libx264", "video.mp4"],
                                 stderr=subprocess.DEVNULL, cwd=directory)
        for file_path in frame_files:
            os.unlink(file_path)

import os
import random
import subprocess

from euclid import Vector2, Point2
from progress.bar import Bar

from .colliding_marbles import HeroSimulator, GameObject

class KarpathyGame(object):
    def __init__(self, settings, record=False):
        self.record = record
        self.fps                    = settings["fps"]
        self.frames_between_actions = settings["frames_between_actions"]
        self.max_frames             = settings["max_frames"]
        self.obj_radius             = settings["obj_radius"]

        self.sim = HeroSimulator(settings)
        self.num_frames = 1

        self.actions = [Vector2(*a) for a in settings["action_acc"]]

        for obj_type, num_items in settings["num_objects"].items():
            for _ in range(num_items):
                self.spawn_object(obj_type)
        self.rewards = settings["object_reward"]

        if self.record:
            self.recording = [self.sim.to_svg()]

        self.sim.collision_observer = lambda x,y: self.handle_collision(x,y)

        self.partial_reward = 0.0
        self.metrics = {
            'score': 0.
        }

    def handle_collision(self, x, y):
        if y is self.sim.hero:
            x, y = y, x

        if x is self.sim.hero:
            assert y is not self.sim.hero
            self.partial_reward += self.rewards[y.obj_type]
            self.sim.remove(y)
            self.spawn_object(y.obj_type)

            return False

        return True

    def spawn_object(self, obj_type):
        speed = Vector2(random.gauss(0., 0.2), random.gauss(0., 0.2))
        self.sim.add(GameObject(Point2(0.,0.), speed,
                 obj_type,
                 radius=self.obj_radius),
                 ensure_noncolliding=True,
                 randomize_position=True)

    def get_state(self):
        if self.num_frames >= self.max_frames:
            return None
        else:
            return self.sim.observe()

    def take_action(self, a):
        self.sim.hero.speed += self.actions[a[0]]

        for _ in range(self.frames_between_actions):
            self.sim.step(1.0 / self.fps)
            self.num_frames += 1
            if self.record and self.num_frames <= self.max_frames:
                self.recording.append(self.sim.to_svg())

        reward = self.partial_reward
        self.metrics['score'] += reward
        self.partial_reward = 0.

        return reward

    def evaluation_metrics(self):
        return self.metrics

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

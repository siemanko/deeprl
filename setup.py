#!/usr/bin/env python
import os

from distutils.core import setup

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

print(ROOT_DIR)

def dicover_packages():
    for path, d,f in os.walk(os.path.join(ROOT_DIR, 'deeprl')):
        if '__init__.py' in f:
            yield os.path.relpath(path, ROOT_DIR).replace(os.path.sep, '.')

print(list(dicover_packages()))

setup(name='deeprl',
      version='0.0.1',
      description='Deep Reinforcement Learning',
      author='Szymon Sidor',
      author_email='szymon.sidor@gmail.com',
      url='http://www.deeprl.net',
      packages=list(dicover_packages()),
)

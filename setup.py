#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='prosody-wavelet-toolkit',
      version='0.1a0',
      description='Prosody wavelet analysis toolkit',
      author='Antti Suni',
      author_email='antti.suni@helsinki.fi',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=find_packages(),
      install_requires=[
          "pycwt", "matplotlib", "numpy", "scipy",
          "soundfile", "tgt", "pyreaper",  "wavio",
          "pyqt5",
      ],
     )

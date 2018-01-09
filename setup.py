#!/usr/bin/env python

from setuptools import setup, find_packages

REQUIREMENTS = [
    # Math
    "pycwt", "matplotlib", "numpy", "scipy",

    # Audio/speech
    "soundfile", "tgt", "pyreaper",  "wavio",

    # Rendering
    "pyqt5", "Sphinx"
]

setup(name='wavelet-prosody-toolkit',
      version='0.1a0',
      description='Prosody wavelet analysis toolkit',
      author='Antti Suni',
      author_email='antti.suni@helsinki.fi',
      packages=find_packages(),
      install_requires=REQUIREMENTS,
      entry_points={
        'console_scripts': [
            'prosody_labeller = wavelet_prosody_toolkit.prosody_labeller:main',
        ],
        'gui_scripts': [
            'wavelet_gui = wavelet_prosody_toolkit.wavelet_gui:main'
        ]
      }
)

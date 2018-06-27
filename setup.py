#!/usr/bin/env python

from setuptools import setup, find_packages

# For documentation => sphinx
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

REQUIREMENTS = [
    # Math
    "pycwt", "matplotlib", "numpy", "scipy",

    # Audio/speech
    "soundfile", "tgt", "pyreaper",  "wavio",

    # Rendering
    "pyqt5", "Sphinx"
]

name='wavelet-prosody-toolkit'
version='0.1a0'
release='0.1'
description='Prosody wavelet analysis toolkit'
author='Antti Suni'

setup(
    name=name,
    author=author,
    version=release,
    cmdclass=cmdclass,
    # these are optional and override conf.py settings
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release)}},
    author_email='antti.suni@helsinki.fi',
    packages=find_packages(),
    package_data={'': ['configs/default.yaml']},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    entry_points={
        'console_scripts': [
            'prosody_labeller = wavelet_prosody_toolkit.prosody_labeller:main',
            'cwt_analysis_synthesis = wavelet_prosody_toolkit.cwt_analysis_synthesis:main',
        ],
        'gui_scripts': [
            'wavelet_gui = wavelet_prosody_toolkit.wavelet_gui:main'
        ]
    }
)

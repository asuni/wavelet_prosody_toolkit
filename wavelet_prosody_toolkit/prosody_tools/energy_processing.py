#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>
    - Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION
    Module which provides the energy routines to be able to apply a wavelet analysis

LICENSE
    See https://github.com/asuni/wavelet_prosody_toolkit/blob/master/LICENSE.txt
"""

import numpy as np
from . import smooth_and_interp, misc


# Logging
import logging
logger = logging.getLogger(__name__)



def extract_energy(waveform, fs=16000, min_freq=200, max_freq=3000, method='rms', target_rate=200):
    #python 2, 3 compatibility hack
    try:
        basestring
    except NameError:
        basestring = str
    # accept both wav-files and waveform arrays
    if isinstance(waveform, basestring):

        (fs, waveform) = misc.read_wav(waveform)

    import scipy.signal
    from . import filter

    lp_waveform =  filter.butter_bandpass_filter(waveform, min_freq, max_freq, fs, order=5)

    # verify that filtering works
    #lp_waveform = waveform
    #scipy.io.wavfile.write("/tmp/tmp.wav", fs, lp_waveform.astype(np.int16))

    # hilbert is sometimes prohibitively slow, should pad to next power of two
    if method == 'hilbert':
        energy=abs(scipy.signal.hilbert(lp_waveform))

    elif method == "true_envelope":
        # window should be about one pitch period, ~ 5 ms
        win = 0.005 *fs
        energy = smooth_and_interp.peak_smooth(abs(lp_waveform), 200,win)

    elif method == "rms":
        energy=np.sqrt(lp_waveform**2)
    logger.debug("fs = %d, target_rate = %d, fs/target_rate = %f" % (fs, target_rate, fs/target_rate))
    energy = misc.resample(energy, fs, target_rate)
    #energy = scipy.signal.resample_poly(energy, 1., int(round(fs/target_rate)))
    logger.debug("len(energy) = %d, len(energy)/target_rate = %f" % (len(energy), len(energy)/target_rate))
    return energy


def process(energy, voicing=[]):
    energy = smooth_and_interp.peak_smooth(energy, 100, 5, voicing=voicing)
    return energy

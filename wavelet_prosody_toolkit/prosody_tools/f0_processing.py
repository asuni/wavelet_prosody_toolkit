#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>
    - Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION
    Module which provides the F0 routines to be able to apply a wavelet analysis

LICENSE
    See https://github.com/asuni/wavelet_prosody_toolkit/blob/master/LICENSE.txt
"""


# Global/system packages
import sys
import os

# Math/signal processing
import numpy as np
from scipy.io import wavfile
import pylab

# Local packages
from . import smooth_and_interp
from . import pitch_tracker

# Logging
import logging
logger = logging.getLogger(__name__)

# Pyreaper
try:
    import pyreaper
    USE_REAPER = True
    logger.info("Pyreaper is available")
except ImportError:
    USE_REAPER = False
    logger.debug("Pyreaper is not available so falling back into the default pitch tracker")


###############################################################################


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _cut_boundary_vals(params, num_vals):
    cutted = np.array(params)
    for i in range(num_vals, len(params)-num_vals):
        if (params[i] <= 0) and (params[i+1] > 0):
            for j in range(i, i+num_vals):
                cutted[j] = 0.0

        if (params[i] > 0) and (params[i+1] <= 0):
            for j in range(i-num_vals, i+1):
                cutted[j] = 0.0

    return cutted


def _remove_outliers(lf0, trace=False):

    if np.nanmean(lf0[lf0 > 0]) > 10:
        raise("logF0 expected")

    fixed = np.array(lf0)

    # remove f0 values from voicing boundaries, if they make a large difference for
    # interpolation
    boundary_cut = smooth_and_interp.interpolate_zeros(_cut_boundary_vals(fixed, 3), 'linear')
    interp = smooth_and_interp.interpolate_zeros(fixed, 'linear')
    fixed[abs(interp-boundary_cut) > 0.1] = 0
    interp = smooth_and_interp.interpolate_zeros(fixed, 'linear')

    # iterative outlier removal
    # 1. compare current contour estimate to a smoothed contour and remove deviates larger than threshold
    # 2. smooth current estimate with shorter window, thighten threshold
    # 3. goto 1.

    # In practice, first handles large scale octave jump type errors,
    # finally small scale 'errors' like consonant perturbation effects and
    # other irregularities in voicing boundaries
    #
    # if this appears to remove too many correct values, increase thresholds
    num_iter = 30
    max_win_len = 100
    min_win_len = 10  # 20
    max_threshold = 3.  # threshold with broad window

    min_threshold = 0.5  # threshold with shorted window

    if trace:
        pylab.rcParams['figure.figsize'] = 20, 5
        pylab.figure()
        pylab.title("outlier removal")

    _std = np.std(interp)
    # do not tie fixing to liveliness of the original
    _std = 0.3

    win_len = np.exp(np.linspace(np.log(max_win_len), np.log(min_win_len),
                                 num_iter+1))
    outlier_threshold = np.linspace(_std*max_threshold, _std*min_threshold,
                                    num_iter+1)
    for i in range(0, num_iter):
        smooth_contour = smooth_and_interp.smooth(interp, win_len[i])
        low_limit = smooth_contour - outlier_threshold[i]
        hi_limit = smooth_contour + outlier_threshold[i]*1.5  # bit more careful upwards, not to cut emphases

        # # octave jump down fix, more harm than good?
        # fixed[interp<smooth_contour-0.45]=interp[interp<smooth_contour-0.45]+0.5
        # fixed[interp>smooth_contour+0.45]=interp[interp>smooth_contour+0.45]-0.5
        fixed[interp > hi_limit] = 0
        fixed[interp < low_limit] = 0

        if trace:
            pylab.clf()
            pylab.title("outlier removal %d" % i)
            # pylab.ylim(3.5,7)
            pylab.plot((low_limit), 'black', linestyle='--')
            pylab.plot((hi_limit), 'black', linestyle='--')
            pylab.plot((smooth_contour), 'black', linestyle='--')
            pylab.plot((interp), linewidth=3)
            pylab.plot(lf0)
            pylab.show()

        interp = smooth_and_interp.interpolate_zeros(fixed, 'linear')

    # if trace:
    #     raw_input("press any key to continue")

    return fixed


def _interpolate(f0, method="true_envelope"):

    if method == "linear":
        return smooth_and_interp.interpolate_zeros(f0, 'linear')
    elif method == "pchip":
        return smooth_and_interp.interpolate_zeros(f0, 'pchip')

    elif method == 'true_envelope':
        interp = smooth_and_interp.interpolate_zeros(f0)

        _std = np.std(interp)
        _min = np.min(interp)
        low_limit = smooth_and_interp.smooth(interp, 200)-1.5*_std
        low_limit[low_limit < _min] = _min
        hi_limit = smooth_and_interp.smooth(interp, 100)+2.0*_std
        voicing = np.array(f0)
        constrained = np.array(f0)
        constrained = np.maximum(f0, low_limit)
        constrained = np.minimum(constrained, hi_limit)

        interp = smooth_and_interp.peak_smooth(constrained, 100, 20,
                                               voicing=voicing)
        # smooth voiced parts a bit too
        interp = smooth_and_interp.peak_smooth(interp, 3, 2)  # ,voicing=raw)
        return interp
    else:
        raise("no such interpolation method: %s", method)


def extract_f0(waveform, fs=16000, f0_min=30, f0_max=550, harmonics=10., voicing=50., configuration="pitch_tracker"):
    """Extract F0 from a waveform

    """
    # first determine f0 without limits, then use mean and std of the first estimate
    # to limit search range.
    if (f0_min == 0) or (f0_max == 0):
        if USE_REAPER and (configuration == "REAPER"):
            _, _, _, f0, _ = pyreaper.reaper(waveform, fs, f0_min, f0_max)
        else:
            (f0, _) = pitch_tracker.inst_freq_pitch(waveform, fs, f0_min, f0_max, harmonics, voicing, False, 200)

        mean_f0 = np.mean(f0[f0 > 0])
        std_f0 = np.std(f0[f0 > 0])
        f0_min = max((mean_f0 - 3*std_f0, 40.0))
        f0_max = mean_f0 + 6*std_f0

        logger.debug("f0_min = %f, f0_max = %f" % (f0_min, f0_max))

    if USE_REAPER and (configuration == "REAPER"):
        _, _, _, f0, _ = pyreaper.reaper(waveform, fs, f0_min, f0_max)
    else:
        (f0, _) = pitch_tracker.inst_freq_pitch(waveform, fs, f0_min, f0_max, harmonics, voicing, False, 200)

    return f0


def process(f0, fix_outliers=True, interpolate=True, do_trace=False):

    lf0 = np.array(f0)
    log_scaled = True
    if np.mean(f0[f0 > 0]) > 20:
        log_scaled = False
        lf0[f0 > 0] = np.log(f0[f0 > 0])
        lf0[f0 <= 0] = 0

    if fix_outliers:
        lf0 = _remove_outliers(lf0, trace=do_trace)
    if interpolate:
        lf0 = _interpolate(lf0, 'true_envelope')
    if not log_scaled:
        return np.exp(lf0)
    else:
        return lf0


# this is temporary: assumes 5ms frame shift,
# assumes format to be either one f0 value per line
# or praat matrix format

def read_f0(filename):
    import os.path
    for ext in [".f0", ".F0"]:
        f0_f = os.path.splitext(filename)[0]+ext

        if os.path.exists(f0_f):
            logger.info("reading F0 file", f0_f)
            try:
                # one f0 value per line
                return np.loadtxt(f0_f)
            except:
                # praat matrix
                try:
                    return np.loadtxt(f0_f, skiprows=4)
                except:
                    logger.error("unknown format for F0 value in file \"%s\"" % filename)

    return None

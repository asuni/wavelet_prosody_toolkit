#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>
    - Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION
    Module which interpolation routines

LICENSE
    See https://github.com/asuni/wavelet_prosody_toolkit/blob/master/LICENSE.txt
"""

# Global/system packages
import sys

# Math/signal processing
import numpy as np
from scipy.io import wavfile
from scipy.signal import decimate
from scipy import interpolate
import pylab

# Logging
import logging
logger = logging.getLogger(__name__)


def remove_bias(params, win_len=300):
    return params-smooth(params, win_len)


# copied from https://stackoverflow.com/questions/23024950/interp-function-in-python-like-matlab/40346185#40346185
def interpolate_by_factor(vector, factor):
    """
    Interpolate, i.e. upsample, a given 1D vector by a specific interpolation factor.
    :param vector: 1D data vector
    :param factor: factor for interpolation (must be integer)
    :return: interpolated 1D vector by a given factor
    """
    # print(vector, factor)

    x = np.arange(np.size(vector))
    y = vector
    f = interpolate.interp1d(x, y)

    x_extended_by_factor = np.linspace(x[0], x[-1],
                                       int(round(np.size(x) * factor)))
    y_interpolated = np.zeros(np.size(x_extended_by_factor))

    i = 0
    for x in x_extended_by_factor:
        y_interpolated[i] = f(x)
        i += 1

    return y_interpolated


def interpolate_zeros(params, method='pchip', min_val=0):
    """
    Interpolate 0 values
    :param params: 1D data vector
    :param method:
    :param factor: factor for interpolation (must be integer)
    :return: interpolated 1D vector by a given factor
    """

    voiced = np.array(params, float)
    for i in range(0, len(voiced)):
        if voiced[i] == min_val:
            voiced[i] = np.nan

    # last_voiced = len(params) - np.nanargmax(params[::-1] > 0)

    if np.isnan(voiced[-1]):
        voiced[-1] = np.nanmin(voiced)
    if np.isnan(voiced[0]):
        voiced[0] = np.nanmean(voiced)

    not_nan = np.logical_not(np.isnan(voiced))

    indices = np.arange(len(voiced))
    if method == 'spline':
        interp = interpolate.UnivariateSpline(indices[not_nan],
                                              voiced[not_nan],
                                              k=2, s=0)
        # return voiced parts intact
        smoothed = interp(indices)
        for i in range(0, len(smoothed)):
            if not np.isnan(voiced[i]):
                smoothed[i] = params[i]

        return smoothed

    elif method == 'pchip':
        interp = interpolate.pchip(indices[not_nan], voiced[not_nan])
    else:
        interp = interpolate.interp1d(indices[not_nan], voiced[not_nan],
                                      method)
    return interp(indices)


def smooth(params, win, type="HAMMING"):

    """
    gaussian type smoothing, convolution with hamming window
    """
    win = int(win+0.5)
    if win >= len(params)-1:
        win = len(params)-1

    if win % 2 == 0:
        win += 1

    s = np.r_[params[win-1:0:-1], params, params[-1:-win:-1]]

    if type == "HAMMING":
        w = np.hamming(win)
        # third = int(win/3)
        # w[:third] = 0
    else:
        w = np.ones(win)

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[int(win/2):-int(win/2)]


def peak_smooth(params, max_iter, win,
                min_win=2, voicing=[], TRACE=False):
    """
    Iterative smoothing while preserving peaks, 'true envelope' -style

    """

    smoothed = np.array(params)
    win_reduce = np.exp(np.linspace(np.log(win), np.log(min_win), max_iter))
    # std = np.std(params)
    if TRACE:
        pylab.ion()
        pylab.plot(params, 'black')

    for i in range(0, max_iter):

        smoothed = np.maximum(params, smoothed)
        # if TRACE:
        #     if (i > 0) and (i % 2 == 0):
        #         pass
        #         pylab.plot(smoothed, 'gray', linewidth=1)
        #         raw_input()

        if len(voicing) > 0:
            smoothed = smooth(smoothed, int(win+0.5))
            smoothed[voicing > 0] = params[voicing > 0]
        else:
            smoothed = smooth(smoothed, int(win+0.5), type='rectangle')

        win = win_reduce[i]

    if TRACE:
        pylab.plot(smoothed, 'red', linewidth=2)
        pylab.show()
    return smoothed

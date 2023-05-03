#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>
    - Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION
    Module which provides a set of helper routines (wav, sginal, scales)

LICENSE
    See https://github.com/asuni/wavelet_prosody_toolkit/blob/master/LICENSE.txt
"""

import os
from scipy.signal import resample_poly
import fractions
import soundfile
import numpy as np
from pylab import ginput

# Logging
import logging
logger = logging.getLogger(__name__)


def read_wav(filename):
    """Read wave file using soundfile.read

    Parameters
    ----------
    filename: string
        Name of the file.

    Returns
    -------
    samplerate: int
        The audio signal sample rate.

    data: 1D arraylike
        The audio samples of the first channel with memory layout as C-order
    """
    # various packages tried.. difficulties with channels, 24bit files, various dtypes
    # pysoundfile appears to mostly work

    data, samplerate = soundfile.read(filename, dtype='int16', always_2d=True)

    return (samplerate, data[:, 0].copy(order='C'))

    """Alternative solutions:
    # import wavio
    # wav = wavio.read(filename)
    # print wav.data.shape
    # pylab.plot(wav.data[:,0])
    # return (wav.rate, wav.data[:, 0])


    import scipy.io.wavfile
    try:
        return scipy.io.wavfile.read(filename)
    except Exception as e:

        print e
    """


def write_wav(filename, data, sr, format="WAV"):
    """Write audio file using soundfile

    Parameters
    ----------
    filename: string
        The name of the wave file.
    data: 1D arraylike
        The audio samples.
    sr: int
        The sample rate.
    format: string
        The output audio format (Default value is WAV for wav file).

    """

    soundfile.write(filename, data, sr, format=format)


def resample(waveform, s_sr, t_sr):
    """resampling for waveforms, should work also with when source and
    target rate ratio is fractional

    Parameters
    ----------
    waveform: np.array
       speech waveform, mono
    s_sr: float
       original sample rate
    t_sr: float
       target sample rate

    returns: resampled waveform as np.array
    """
    ratio = fractions.Fraction(int(t_sr), int(s_sr))
    return resample_poly(waveform.astype(float), ratio.numerator, ratio.denominator)


def play(utt):
    wavfile = utt + ".wav"
    wavfile = wavfile.replace(" ", "\ ")
    st = 0.2
    end = 1

    while (st > 0.01):
        try:
            pts = ginput(1)
            st = pts[0][0] / 200.0
            end = 1.0
        except:
            continue
        os.system("play %s trim 0:0:%f 0:0:%f " % (wavfile, st, end))


def match_length(sig_list):
    """Reduce length of all signals to a the minimum one.

    Parameters
    ----------
    sig_list: list
        List of signals which are 1D array of samples.

    """
    length = min(map(len, sig_list))

    for i in range(0, len(sig_list)):
        sig_list[i] = sig_list[i][:int(length)]

    return sig_list


def get_peaks(params, threshold=-10):
    """Find the peaks based on the given prosodic parameters.

    Parameters
    ----------
    params: ?
        Prosodic parameters
    threshold: int
        description

    Returns
    -------
    peaks: arraylike
        array of peak values and peak indices
    """
    # zc = np.where(np.diff(np.sign(np.diff(params))))[0]  # FIXME SLM: not used
    indices = (np.diff(np.sign(np.diff(params))) < 0).nonzero()[0] + 1

    peaks = params[indices]
    return np.array([peaks[peaks > threshold], indices[peaks > threshold]])


def calc_prominence(params, labels, func=np.max, use_peaks=True, rate=200):
    """Compute prominences

    Parameters
    ----------
    params: type
        description
    labels: type
        description
    func: function handle
    use_peaks: boolean
        Use peaks (True) or not (False) to determine the prominence
    rate: int
        The rate (default=200 (Hz) for 5ms)

    """
    labelled = []
    # norm = params.astype(float)  # FIXME SLM: not used
    for (start, end, segment, word) in labels:
        if use_peaks:
            peaks = []
            (peaks, indices) = get_peaks(params[start*rate-1:end*rate], 0.0)

            if len(peaks) > 0:
                labelled.append(np.max(peaks))
            else:
                labelled.append(0.0)
        else:
            # labelled.append([word, func(params[start-10:end])])
            labelled.append(func(params[start*rate:end*rate]))

    return labelled


def get_best_scale(wavelet_matrix, num_units):
    """Find the scale whose number of peaks is closest to the number of units

    Parameters
    ----------
    wavelet_matrix: arraylike
        The wavelet matrix data.
    num_units: int
        The target number of units

    Returns
    -------
    int
        the index of the best scale
    """
    best_i = 0
    best = 999
    for i in range(0, wavelet_matrix.shape[0]):
        num_peaks = len(get_peaks(wavelet_matrix[i])[0])

        dist = abs(num_peaks - num_units)
        if dist < best:
            best = dist
            best_i = i

    return best_i


def get_best_scale2(scales, labels):
    """Find the scale whose width is the closes to the average unit length represented in the labels

    Parameters
    ----------
    scales: 1D arraylike
        The scale indices
    labels: list of tuple (float, float, string)
        List of labels which are lists of 3 elements [start, end, description]


    Returns
    -------
    int
        the index of the best scale

    """
    mean_length = 0
    for l in labels:
        mean_length += (l[1] - l[0])

    mean_length /= len(labels)
    dist = scales - mean_length

    return np.argmin(np.abs(dist))


def normalize_minmax(params, epsilon=0.1):
    """Normalize parameters into a 0,1 scale

    Parameters
    ----------
    params: arraylike
        The parameters to normalize.
    epsilon: float
        The epsilon to deal with numerical stability

    Returns
    ------
    arraylike
        the normalized parameters

    """
    return (params-min(params)+epsilon)/(max(params)-min(params))


def normalize_std(params, std=0):
    """Normalize parameters using a z-score paradigm

    Parameters
    ----------
    params: arraylike
        The parameters to normalize.
    std: float
        A given standard deviation. If 0, the standard deviation is computed on the params. (Default: 0)


    Returns
    ------
    arraylike
        the normalized parameters
    """
    if std == 0:
        std = np.nanstd(params)

    # empty array or all zeros
    # if std==0:
    if std < 0.00001:  # np.isclose([std,0]):
        return np.zeros(len(params))

    mean = np.nanmean(params)

    return (params - mean) / float(std)

# -*- coding: utf-8 -*-
"""Butter filter utilities

This module contains butter filter help functions copied from http://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
"""

from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Generate the butter bandpass filter

    For more details see scipy.signal.butter documentation

    Parameters
    ----------
    lowcut: int
        The low cut value
    highcut: type
        description
    fs: int
        Signal sample rate
    order: int
        Order of the butter fiter

    Returns
    -------
    Tuple
    	Numerator (`b`) and denominator (`a`) polynomials of the IIR filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    if highcut >=nyq*0.95:
        highcut = nyq*0.95
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Filter signal data using a butter filter type

    For more details see scipy.signal.butter and scipy.signal.lfilter documentation

    Parameters
    ----------
    data: array_like
        An N-dimensional input array.
    lowcut: int
        The lowcut filtering value.
    highcut: type
        The highcut filtering value.
    fs: int
        The signal sample rate.
    order: int
        The order of the butter filter.

    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)

    return y

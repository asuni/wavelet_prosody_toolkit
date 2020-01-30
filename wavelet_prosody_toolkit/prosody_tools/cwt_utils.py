#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>
    - Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION
    Module which provides continuous wavelet transform (cwt) analysis/synthesis routines

LICENSE
    See https://github.com/asuni/wavelet_prosody_toolkit/blob/master/LICENSE.txt
"""

from numpy import array,concatenate, sqrt, pad, mean, std, real, nan, zeros, nanmean, nanstd, pi, around, log2

import pycwt as cwt

###########################################################################################
# Private routines
###########################################################################################
def _unpad(matrix, num):
    """Private function to unpad axis 1 of a matrix

    Parameters
    ----------
    matrix: ndarray
        a NDarray
    num: int
       the unpadding size

    Returns
    -------
    ndarray
    	the unpadded matrix
    """
    unpadded = matrix[:,num:len(matrix[0])-num]
    return unpadded


def _padded_cwt(params, dt, dj, s0, J, mother, padding_len):
    """Private function to compute a wavelet transform on padded data

    Parameters
    ----------
    params: arraylike
        The prosodic parameters.
    dt: ?
        ?
    dj: ?
        ?
    s0: ?
        ?
    J: ?
        ?
    mother: ?
        The mother wavelet.
    padding_len: int
        The padding length

    Returns
    -------
    wavelet_matrix: ndarray
    	The wavelet data resulting from the analysis
    scales: arraylike
    	The scale indices corresponding to the wavelet data
    freqs: ?
    	?
    coi: array
    	The cone of influence values
    fft: ?
    	?
    fftfreqs: ?
    	?
    """
    #padded = concatenate([params,params,params])
    padded = pad(params, padding_len, mode='edge') #edge
    wavelet_matrix, scales, freqs, coi, fft, fftfreqs = cwt.cwt(padded, dt, dj, s0, J, mother)
    wavelet_matrix = _unpad(wavelet_matrix, padding_len)
    #wavelet_matrix = _unpad(wavelet_matrix, len(params))

    return (wavelet_matrix, scales, freqs, coi, fft, fftfreqs)




def _zero_outside_coi(wavelet_matrix,freqs, rate = 200):
    """Private function to set each elements outside of the Cone Of Influence (coi) to 0.
  
    Parameters
    ----------
    wavelet_matrix: type
        description
    freqs: type
        description

    """
    for i in range(0,wavelet_matrix.shape[0]):
        coi =int(1./freqs[i]*rate)
        wavelet_matrix[i,0:coi] = 0.
        wavelet_matrix[i,-coi:] = 0.
    return wavelet_matrix

def _scale_for_reconstruction(wavelet_matrix,scales, dj, dt,mother="mexican_hat",period=3):
    """ ?

    Parameters
    ----------
    wavelet_matrix: ndarray
    	The wavelet data resulting from the analysis
    scales: arraylike
    	The scale indices corresponding to the wavelet data
    dj: ?
        ?
    dt: ?
        ?
    mother: ?
        ?
    period: ?
        ?

    """
    scaled = array(wavelet_matrix)

    # mexican Hat
    c = dj / (3.541 * 0.867)

    if mother=="morlet":
        cc = 1.83

        #periods 5 and 6 are correct, 3,4 approximate
        if period == 3:
            cc = 1.74
        if period == 4:
            cc = 1.1
        elif period==5:
            cc=0.9484
        elif period==6:
            cc == 0.7784

        c = dj / (cc * pi**(-0.25))
     
    for i in range(0, len(scales)):
        scaled[i]*= c*sqrt(dt)/sqrt(scales[i])
        # substracting the mean should not be necessary?
        scaled[i]-=mean(scaled[i])

    return scaled


def _freq2scale(freq, mother, period = 3.):
    """
    convert frequency to wavelet scale width
    
    Parameters
    ----------
    freq: float
          frequency value in Hz

    mother: string
            name of the mother wavelet ("mexican_hat", "morlet")
    """

    freq = float(freq)
    if mother.lower() == "mexican_hat":
        return (1./freq)/(2. * pi / sqrt(2 + 0.5)) #np.sqrt(2./(2.*k0+1.)));
    if mother.lower() == "morlet":
        return  (1./freq)*(period + sqrt(2. + period**2))/(4 * pi)
    else:
        return (1./freq)/ (4. * pi / (2. * period + 1.))
     
###########################################################################################
# Public routines
###########################################################################################
def combine_scales(wavelet_matrix, slices):
    """Combine the scales of given slices

    Parameters
    ----------
    wavelet_matrix: ndarray
        The wavelet data matrix.
    slices: ndarray
        The slices

    Returns
    -------
    array
    	The combined scales
    """
    combined_scales = []

    for i in range(0, len(slices)):
        combined_scales.append(sum(wavelet_matrix[slices[i][0]:slices[i][1]]))
    return array(combined_scales)


def cwt_analysis(params, mother_name="mexican_hat",num_scales=12, first_scale = None, first_freq = None, scale_distance=1.0, apply_coi=True, period=5, frame_rate = 200):
    """Achieve the continous wavelet analysis of given parameters

    Parameters
    ----------
    params: arraylike
        The parameters to analyze.
    mother_name: string, optional
        The name of the mother wavelet [default: mexican_hat].
    num_scales: int, optional
        The number of scales [default: 12].
    first_scale: int, optional
        The width of the shortest scale
    first_freq: int, optional
        The highest frequency in Hz
    scale_distance: float, optional
        The distance between scales [default: 1.0].
    apply_coi: boolean, optional
        Apply the Cone Of Influence (coi)
    period: int, optional
        The period of the mother wavelet [default: 5].
    frame_rate: int, optional
        The signal frame rate [default: 200].

    Returns
    -------
    wavelet_matrix: ndarray
    	The wavelet data resulting from the analysis
    scales: arraylike
    	The scale indices corresponding to the wavelet data
    """
    # setup wavelet transform
   
    dt = 1. /float(frame_rate)  # frame length

    if not first_scale:
        first_scale = dt # first scale, here frame length
    
    if first_freq:
        first_scale = _freq2scale(first_freq, mother_name, period)
        
    dj = scale_distance  # distance between scales in octaves
    J =  num_scales #  number of scales

    mother = cwt.MexicanHat()

    if str.lower(mother_name) == "morlet":
        mother = cwt.Morlet(period)
    elif str.lower(mother_name) == "paul":
        mother = cwt.Paul(period)

    wavelet_matrix, scales, freqs, coi, fft, fftfreqs = _padded_cwt(params, dt, dj, first_scale, J,mother, 400)
    #wavelet_matrix, scales, freqs, coi, fft, fftfreqs = cwt.cwt(f0_mean_sub, dt, dj, s0, J,mother)

    #wavelet_matrix = abs(wavelet_matrix)
    wavelet_matrix = _scale_for_reconstruction((wavelet_matrix), scales, dj, dt,mother=mother_name,period=period)

    if apply_coi:
        #wavelet_matrix = _zero_outside_coi(wavelet_matrix, scales/dt*0.5)
        wavelet_matrix = _zero_outside_coi(wavelet_matrix, freqs, frame_rate)
    import numpy as np
    np.set_printoptions(precision=3, suppress=True)
    return (wavelet_matrix,scales,freqs)


def cwt_synthesis(wavelet_matrix, mean = 0):
    """Synthesizing a signal given a wavelet dataset

    Parameters
    ----------
    wavelet_matrix: ndarray
        The wavelet data matrix.
    mean: float
        The mean to translate the signal.

    Returns
    -------
    arraylike
    	The generated signal

    """
    return sum(wavelet_matrix[:])+mean

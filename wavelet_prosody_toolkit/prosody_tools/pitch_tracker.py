#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>
    - Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION
    Module which a default pitch tracker

LICENSE
    See https://github.com/asuni/wavelet_prosody_toolkit/blob/master/LICENSE.txt
"""

# Logging
import logging
logger = logging.getLogger(__name__)

import numpy as np
from . import misc, cwt_utils, filter, f0_processing, smooth_and_interp

import sys
from scipy.io import wavfile
import scipy.signal


def _get_f0(spec, energy, min_hz, max_hz, thresh, sil_thresh):
    """
    return frequency bin with maximum energy, if it is over given threshold
    and overall energy of the frame is over silence threshsold
    otherwise return 0 (unvoiced)
    """

    cand = int(min_hz)+np.argmax(spec[int(min_hz):int(max_hz)])
    if spec[cand] > thresh and energy > sil_thresh:
        if cand > 2*min_hz and spec[int(round(cand/2.))] > spec[cand]*0.5:
            return int(round(cand/2.))
        else:
            return cand
    return 0


def _track_pitch(pic, min_hz=50, max_hz=450,thresh=0.1,energy_thresh=1.0, DEBUG=False):
    """
    extract pitch contour from time-frequency image
    bin with maximum energy / frame is chosen as a first f0 estimate,
    following with refinement steps based on the assumption of continuity of the pitch track
    """

    if DEBUG:
        import pylab
        from matplotlib import colors


    pitch = np.zeros(pic.shape[0])

    # calc energy threshold for voicing
    log_energy = np.log(np.sum(pic, axis=1))
    energy_thresh=np.min(smooth_and_interp.smooth(log_energy,20))+energy_thresh
    pic_smooth = pic*scipy.ndimage.gaussian_filter(pic, [2,5])

    if DEBUG:
        pylab.plot(log_energy)
        pylab.plot(np.full(len(log_energy), energy_thresh))
        pylab.show()


    # find frequency bins with max_energy
    for i in range(0, pic_smooth.shape[0]):
        pitch[i] = _get_f0(pic_smooth[i], log_energy[i],min_hz, max_hz, thresh, energy_thresh)



    # second pass with soft constraints
    n_iters = 3
    from scipy.signal import gaussian


    for iter in range(0, n_iters):

        smoothed = f0_processing.process(pitch)
        smoothed = smooth_and_interp.smooth(smoothed, int(200./(iter+1.)))



        # gradually thightening gaussian window centered on current estimate to softly constrain next iteration
        win_len = 800

        g_window = gaussian(win_len, int(np.mean(smoothed)*(1./(iter+1.)**2)))
        #g_window = gaussian(win_len, (1./(iter+2)**2)))

        for i in range(0, pic.shape[0]):
            window=np.zeros(len(pic_smooth[i]))
            st = int(np.max((0, int(smoothed[i]-win_len))))
            end = int(np.min((int(smoothed[i]+win_len*0.5), win_len-st)))
            window[st:end]=g_window[win_len-end:]
            pitch[i] = _get_f0(pic_smooth[i]*window, log_energy[i],min_hz, max_hz, thresh, energy_thresh)

    return pitch





def _assign_to_bins(pic, freqs, mags):
    for i in range(1, freqs.shape[0]-1):
        for j in range(0, freqs.shape[1]):
            try:
                pic[j, int(freqs[i,j])]+=(mags[i,j])
            except:
                pass


def inst_freq_pitch_from_wav(utt_wav, min_hz=50, max_hz=400, acorr_weight=10., voicing_thresh=50., DEBUG=False, target_rate=200):
    # adjust thhresholds
    # the thresholds are empirically set, depends on number of bins, normalization, smoothing etc..


    # read wav file, downsample to 4000Hz and normalize

    (fs, wav_form) = misc.read_wav(utt_wav)

    return inst_freq_pitch(wav_form, fs, min_hz, max_hz, acorr_weight, voicing_thresh, DEBUG, target_rate)

def inst_freq_pitch(wav_form, fs, min_hz=50, max_hz=400, acorr_weight=10., voicing_thresh=50., DEBUG=False, target_rate=200):
    """
    extract f0 track from speech wav file using instanenous frequency calculated from continuous wavelet transform
    """

    voicing_thresh = (voicing_thresh-50.0) / 100.0
    acorr_weight /= 100.
    sample_rate = 4000.0
    tmp_wav_form = misc.resample(wav_form, fs, sample_rate)
    #params = scipy.signal.resample_poly(params, 1., int(round(fs/sample_rate)))
    tmp_wav_form = misc.normalize_std(tmp_wav_form)

    # init instantenous frequency pic, with rather low time and frequency resolution for speed
    # having 1 hz / bin simplifies the implememtation a bit, but treats males and females differently (other vals do not work)
    steps_in_hertz =1.0

    DEC = int(round(sample_rate/target_rate))

    pic = np.zeros(shape=(int(len(tmp_wav_form)/float(DEC)), int(sample_rate/4.0)))


    # use continuous wavelet transform to get instantenous frequencies
    # integrate analyses with morlet mother wavelets with periods = 3,5,7 for good time and frequency resolution

    # setup wavelet
    #dt = 0.2 #4./sample_rate
    s0 = 2./sample_rate

    dj = 0.05 # 20 scales per octave
    J= 120  # six octaves
    dt = 1./sample_rate
    #periods = [3,5,7] #maybe this is too slow to be default
    periods = [5]
    for p in periods:

        (wavelet_matrix,scales,cwt_freqs) = cwt_utils.cwt_analysis(tmp_wav_form, mother_name="morlet",first_scale = s0, num_scales=J, scale_distance=dj, apply_coi=False,period=p, frame_rate = sample_rate)
        # hilbert transform
        phase = np.unwrap(np.angle(wavelet_matrix), axis=1)
        freqs =  np.abs((np.gradient(phase, dt)[1]) / (2. * np.pi))

        freqs = scipy.signal.decimate(freqs, DEC, zero_phase=True)
        mags = scipy.signal.decimate(abs(wavelet_matrix), DEC, zero_phase=True)

        # normalize magnitudes
        mags = (mags-mags.min())/mags.ptp()

        # construct time-frequency image
        _assign_to_bins(pic, freqs, mags)


    # perform frequency domain autocorrelation to enhance f0

    pic= scipy.ndimage.filters.gaussian_filter(pic,[1,1])

    length = np.min((max_hz*3,pic.shape[1])).astype(int)

    for i in range(0, pic.shape[0]): # frame
        acorr1 = np.correlate(pic[i,:length], pic[i,:length], mode='same')
        pic[i, :int(length/2.)] *= acorr1[int(len(acorr1)/2.):]



    # generate pitch track from the image
    logger.debug("tracking pitch..")

    pitch = _track_pitch(pic,min_hz, max_hz, voicing_thresh, DEBUG=DEBUG)

    if DEBUG:
        pylab.imshow(pic[:, 0:].T, interpolation='nearest', origin='lower',aspect='auto')
        pylab.plot(pitch,'black',linewidth=1)
        pylab.show()

    logger.debug("tracking pitch done.")
    return (pitch,pic)

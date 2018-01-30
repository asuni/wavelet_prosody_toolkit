
import matplotlib
import numpy as np
from . import misc, cwt_utils, filter, f0_processing, smooth_and_interp



import sys


from scipy.io import wavfile

import scipy.signal


# Logging
import logging
logger = logging.getLogger(__name__)


def _get_f0(spec, min_hz, max_hz, thresh):

    cand = min_hz+np.argmax(spec[min_hz:max_hz])

    if spec[cand] > thresh:


        if cand > 2*min_hz and spec[int(round(cand/2.))] > thresh*0.75:
             return int(round(cand/2.))
        else:
            return cand

    return 0

# just picks local maximum in spectra
def simple_pitch(pic, min_hz=50, max_hz=450,thresh=50.0, DEBUG=False):
    if DEBUG:
        import pylab

    pitch = np.zeros(pic.shape[0]) #pic.shape[0])

    #pic_smooth = pic
    pic_smooth = pic*scipy.ndimage.gaussian_filter(pic, [2,5])
    if DEBUG:
        #pylab.imshow(pic[:, 0:max_hz].T, interpolation='nearest', origin='lower',aspect='auto')
        pylab.imshow(np.log(pic_smooth[:, min_hz:max_hz].T), interpolation='nearest', origin='lower',aspect='auto')
        pylab.show()
        pylab.figure()

    # tendency to octave jump upward; attenuate harmonics
    #attenuate = np.sqrt(np.linspace(5,.1, pic.shape[1]))
    #pic_smooth*=attenuate

    logger.debug("ok")

    if DEBUG:
        #pylab.imshow(((pic_smooth+1).T), aspect='auto')
        pylab.imshow(np.log(pic_smooth[:, min_hz:max_hz]).T, interpolation='nearest', origin='lower',aspect='auto')
        #pylab.show()
    # get slot with maximum energy / frame
    for i in range(0, pic_smooth.shape[0]):
        pic_smooth[i]=misc.normalize_std(pic_smooth[i])
        pitch[i] = _get_f0(pic_smooth[i], min_hz, max_hz, thresh)


    if DEBUG:
        pylab.plot(pitch-min_hz,color="black")
        pylab.show()

    # second pass with constraints
    n_iters = 3
    constrain=0.5


    from scipy.signal import gaussian
    for iter in range(0, n_iters):
        try:

            smoothed = f0_processing.process(pitch)
            smoothed = smooth_and_interp.smooth(smoothed, int(200./(iter+1.)))
        except:
            return pitch
        win_len=500
        #gaussian window centered on current estimate to softly constrain next iteration
        g_window = gaussian(win_len*2, int(np.mean(smoothed)*0.5))


        for i in range(0, pic.shape[0]):
            window=np.zeros(len(pic_smooth[i]))
            st = int(np.max((0, int(smoothed[i]-win_len))))
            end = int(np.min((int(smoothed[i]+win_len), win_len*2-st)))
            logger.debug("start=%d, end=%d, offset=%d" % (st, end, win_len*2-end))
            window[st:end]=g_window[win_len*2-end:]
            pic_smooth[i] = (pic_smooth[i]*window)
        if DEBUG:
            pylab.figure()
            pylab.imshow((pic_smooth[:, min_hz:max_hz]).T, interpolation='nearest', origin='lower',aspect='auto')
            pylab.plot(smoothed-min_hz)
            pylab.show()

        for i in range(0, pic.shape[0]):
            min = np.max([min_hz, int(smoothed[i]*(1.-constrain))])
            max = np.min([max_hz, int(smoothed[i]*(1.+constrain))])
            pitch[i] = _get_f0(pic_smooth[i], min, max, thresh)
            #pitch[i] = _get_f0(cur_frame, min, max, thresh)

        if DEBUG:
            pylab.imshow((pic_smooth[:, :max_hz]).T, interpolation='nearest', origin='lower',aspect='auto')
            pylab.plot(smoothed*(1.+constrain))
            pylab.plot(smoothed*(1.-constrain))
            pylab.plot(smoothed)
            pylab.plot(pitch, linewidth=1, color="black")
            pylab.show()
    if DEBUG:
        pylab.plot(pitch)
        pylab.show()
    return pitch





def inst_freq_pitch(utt_wav,min_hz=50, max_hz=400, acorr_weight=50., voicing_thresh=50., DEBUG=False, target_rate=200):
    if DEBUG:
        import pylab
    # read wav file and downsample to 4000Hz
    #fs, params = wavfile.read(utt_wav)
    #import siglib.wavio
    (fs, wav) = misc.read_wav(utt_wav)

    params = wav #xwav.data[:,0]


    # slider scale 0 -> 100
    # the threshold is empirically set, depends on number of bins, normalization, smoothing etc..
    voicing_thresh=(voicing_thresh-50.0) / 100.0+0.2
    acorr_weight /=100.
    #voicing_thresh=(voicing_thresh*1000000.)


    #params =  filter.butter_bandpass_filter(params, int(min_hz*0.5), int(2.5*max_hz), fs, order=5)
    #params =  filter.butter_bandpass_filter(params, 30, 1000, fs, order=4)

    sample_rate = 2000.0
    params = scipy.signal.resample_poly(params, 1., int(round(fs/sample_rate)))
    #params = misc.normalize_std(params)
    #params = misc.resample(params, int(len(params)/(fs/sample_rate)))
    logger.info("downsampled")
    # sometimes the final byte sample messes the analysis or what is the reason
    params = params[:len(params)-1]

    # setup wavelet
    #dt = 0.2 #4./sample_rate
    s0 = 2./sample_rate
    dj = 0.25 # five scales per octave
    J= 60  # five octaves


    # frequency resolution, can be coarse for our purposes
    # equalize the number of bins a bit between males and females by max pitch
    #steps_in_hertz = 400.0 / max_hz
    steps_in_hertz =1.

    max_hz = int(max_hz*steps_in_hertz)
    min_hz = int(min_hz*steps_in_hertz)
    DEC = int(round(sample_rate/target_rate))

    pic = np.zeros(shape=(int(len(params)/float(DEC)+1), int(sample_rate/2.0))) #np.log(sample_rate)*bins)))


    # use morlet for good frequency resolution
    logger.info("Compute CWT")


    (wavelet_matrix,scales) = cwt_utils.cwt_analysis(params, mother_name="morlet",first_scale = s0, num_scales=J, scale_distance=dj, apply_coi=False,period=5, frame_rate = sample_rate)

    mags = np.array(scipy.signal.decimate(np.real(wavelet_matrix), DEC, zero_phase=True))
    freqs = np.array(mags)



    for i in range(0, wavelet_matrix.shape[0]):

        # for mexican hat this one
        #h = hilbert(np.real(wavelet_matrix[i]))

        # get instantenous frequency, morlet analysis contains phase, no hilbert necessary
        h = wavelet_matrix[i]
        phase =np.unwrap((np.angle(h)))

        freq = np.diff(phase) / (2.0*np.pi) * sample_rate
        #freq = np.clip(freq,0, int(sample_rate/2.0)-1)


        mag = abs(h)

        freq = scipy.signal.decimate(freq,DEC, zero_phase=True)
        freq = (freq*steps_in_hertz).astype('int')

        mag = scipy.signal.decimate(mag, DEC, zero_phase=True)

        #freq[(freq)<(np.nanmean(freq)-np.nanstd(freq))] = 0
        #freq[(freq)>(np.nanmean(freq)+np.nanstd(freq))] = 0
        #mag[freq==0] = 0
        #pylab.plot(freq)

        #pylab.show()
        # some problem with lengths, temporary fix
        end_point=np.min([len(mags[i]), len(freq), len(mag)])
        mags[i,:end_point] = mag[:end_point]
        freqs[i, :end_point] = freq[:end_point]




    logger.info("Inst. Freq. done")

    # normalize magnitudes
    mags = (mags-mags.min())/mags.ptp()

    # combine frequency and magnitudes to spectrogram-like thing
    for i in range(2, wavelet_matrix.shape[0]-3):
        for j in range(0, len(freqs[i])):
            try:
                pic[j, int(freqs[i,j])]+=(mags[i,j]) #*5000
            except:
                pass




    # map harmonics down
    logger.info("Compute harmonics..")

    #thresh =0.025
    #thresh =0.02
    #tolerance = 3
    # smooth spectrogram a bit
    pic= scipy.ndimage.filters.gaussian_filter(pic,[2,2])
    if DEBUG:

        pylab.imshow(pic[:, 0:max_hz].T, interpolation='nearest', origin='lower',aspect='auto')
        #pylab.ion()
        pylab.figure()


    acorr_plot = []
    length = np.min((max_hz*3,pic.shape[1]))

    for i in range(0, pic.shape[0]): # frame
        acorr1 = np.correlate(pic[i,:length], pic[i,:length], mode='same') #[length-1:]
        pic[i, :int(length/2.)] =  (1.-acorr_weight)*pic[i,:int(length/2.)] + acorr_weight*acorr1[int(len(acorr1)/2.):]

    logger.info("harmonics done..")


    if DEBUG:
        pylab.imshow(pic[:, 0:max_hz].T, interpolation='nearest', origin='lower',aspect='auto')
        pylab.show()
    # get pitch values from framewise maximum energy
    # pic[:,:min_hz] = 0
    # pic[:,max_hz:] = 0
    #pic= scipy.ndimage.filters.gaussian_filter(pic,[1.5,1.5])
    #pic /=np.max(pic)
    logger.info("Track pitch..")
    pitch = simple_pitch(pic,min_hz, max_hz, voicing_thresh, DEBUG=DEBUG)
    if DEBUG:
        pylab.imshow(pic[:, 0:].T, interpolation='nearest', origin='lower',aspect='auto')
        pylab.plot(pitch,'black',linewidth=1)
        #pylab.plot(f0_processing.process(pitch), 'white', linewidth=5, alpha=0.3)
        #return (pitch,pic[:, 0:max_hz].T)
        pylab.show()
    logger.info("Track pitch done")
    return (pitch,pic)



if __name__ == "__main__":
    import sys

    #if __package__ is None:
    #    import sys
    #    from os import path
    #    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ))
    inst_freq_pitch(sys.argv[1],30,400,voicing_thresh=30,DEBUG=True)

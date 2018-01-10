import numpy as np
from . import smooth_and_interp, misc


# Logging
import logging
logger = logging.getLogger(__name__)

#def extract_energy(waveform, min_freq=200, max_freq=3000, method='mag', target_rate=200):

def extract_energy(waveform, fs=16000, min_freq=200, max_freq=3000, method='mag', target_rate=200):
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

    elif method == "mag":
        energy=np.sqrt(lp_waveform**2)
    logger.debug("fs = %d, target_rate = %d, fs/target_rate = %f" % (fs, target_rate, fs/target_rate))
    #scipy complains about too large decimation if done in single step
    energy = scipy.signal.resample_poly(energy, 1., int(round(fs/target_rate)))
    logger.debug("len(energy) = %d, len(energy)/target_rate = %f" % (len(energy), len(energy)/target_rate))
    return energy


def process(energy, voicing=[]):

    energy=smooth_and_interp.peak_smooth(energy, 100, 5, voicing=voicing)
    return energy


if __name__ == "__main__":


    import sys
    import pylab
    hilbert_env = extract_energy(sys.argv[1]) #, min_freq=500, max_freq=4000, method='hilbert')

    pylab.plot(np.log(hilbert_env))
    true_env = extract_energy(sys.argv[1], method='true_envelope')
    mag = extract_energy(sys.argv[1], method='mag')

    pylab.plot(misc.normalize_minmax(hilbert_env), label="hilbert envelope")
    pylab.plot(misc.normalize_minmax(true_env), label="true envelope")
    pylab.plot(misc.normalize_minmax(mag), label="magnitude" )

    pylab.legend()
    pylab.show()

    import f0_processing
    f0 = f0_processing.extract_f0(sys.argv[1])
    f0, true_env= misc.match_length(f0,true_env)
    #true_env[f0<= 0] = 0
    pylab.plot(misc.normalize_minmax(true_env))
    pylab.plot(misc.normalize_minmax(f0_processing.process(f0)))
    pylab.show()
    pylab.ion()
    pylab.plot(hilbert_env)
    """
    hilbert_env[hilbert_env<0] = 0
    pylab.plot((0.15*np.log(hilbert_env+1.)))
    raw_input()

    fixed = f0_processing._remove_outliers(np.log(hilbert_env+1.))
    pylab.clf()
    fixed = np.exp(fixed)-1.
    pylab.plot(fixed)
    pylab.plot(hilbert_env)
    raw_input()

    #pylab.plot(f0_processing._remove_outliers(0.15*np.log(true_env+1.), trace=True))
    """
    pylab.clf()
    import scipy.signal
    pylab.plot(misc.normalize_minmax(hilbert_env))
    pylab.plot(process(misc.normalize_minmax(scipy.signal.medfilt(hilbert_env,5))))
    pylab.show()
    raw_input()

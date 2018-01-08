# -*- coding: utf-8 -*-

import numpy as np


from . import smooth_and_interp, misc

def duration(labels, rate=200,linear=True):
    dur = np.zeros(len(labels))
    params = np.zeros(int(labels[-1][1]*rate))

    for i in range(len(labels)):
        (st,en, unit) = labels[i]
        #print labels[i]
        
        st*=rate
        en*=rate
        dur[i] = en-st
        if not linear:
            dur[i] = np.log(dur[i])
        params[int(st+(en-st)/2.0)] = dur[i]

    params = smooth_and_interp.interpolate_zeros(params) #, 'linear')
    #params = np.diff(params, 1)

    return params


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

        
        
def _cut_boundary_vals(params, num_vals):
    cutted = np.array(params)
    for i in range(num_vals, len(params)-num_vals):
        if params[i] <=0 and params[i+1] > 0:
            for j in range(i, i+num_vals):
                cutted[j] = 0.0
        if params[i] >0 and params[i+1] <= 0:
            for j in range(i-num_vals, i+1):
                cutted[j] = 0.0
    return cutted




def _interpolate_zeros(params, method='pchip', min_val = 0):


    voiced = np.array(params, float)        
    for i in range(0, len(voiced)):
        if voiced[i] == min_val:
            voiced[i] = np.nan
    last_voiced=len(params)-np.nanargmax(params[::-1] >0)
    
    if np.isnan(voiced[-1]):
        voiced[-1] = np.nanmin(voiced)
    if np.isnan(voiced[0]):
        voiced[0] = np.nanmean(voiced)

    not_nan = np.logical_not(np.isnan(voiced))

    indices = np.arange(len(voiced))
    if method == 'spline':
        interp = scipy.interpolate.UnivariateSpline(indices[not_nan],voiced[not_nan],k=2,s=0)
        # return voiced parts intact
        smoothed = interp(indices)
        for i in range(0, len(smoothed)):
            if not np.isnan(voiced[i]) :                    
                smoothed[i] = params[i]
        return smoothed
    elif method =='pchip':
        interp = scipy.interpolate.pchip(indices[not_nan], voiced[not_nan])
    else:
        interp = scipy.interpolate.interp1d(indices[not_nan], voiced[not_nan], method)
    return interp(indices)

def _smooth(params, win, type="HAMMING"):
    
    """
    gaussian type smoothing, convolution with hamming window
    """
    win = int(win+0.5)
    if win >= len(params)-1:
        win = len(params)-1
    if win % 2 == 0:
        win+=1

    s = np.r_[params[win-1:0:-1],params,params[-1:-win:-1]]

    
    if type=="HAMMING":
        w = np.hamming(win)
        #third = int(win/3)
        #w[:third] = 0
    else:
        w = np.ones(win)
        
        
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[int(win/2):-int(win/2)]
    
def _peak_smooth(params, max_iter, win,min_win=2,voicing=[]):

    TRACE = False
    smooth=np.array(params)

    win_reduce =  np.exp(np.linspace(np.log(win),np.log(min_win), max_iter))

    std = np.std(params)

    if TRACE:
        pylab.plot(params, 'black')
    for i in range(0,max_iter):

        smooth = np.maximum(params,smooth)
        if TRACE:
            if i> 0 and i % 5 == 0:
                pass
                pylab.plot(smooth,'gray',linewidth=1)
                nput()

        if len(voicing) >0:
            smooth = _smooth(smooth,int(win+0.5))
            smooth[voicing>0] = params[voicing>0]
        else:
            smooth = _smooth(smooth,int(win+0.5),type='rectangle')

        win = win_reduce[i]
    
    if TRACE:
        pylab.plot(smooth,'red',linewidth=2)
        input()

 
    return smooth

    
def _remove_outliers(lf0, trace=False):
   

    if np.nanmean(lf0[lf0>0])>10:
        raise("logF0 expected")
    fixed = np.array(lf0)


    # remove f0 values from voicing boundaries, if they make a large difference for
    # interpolation
    boundary_cut = smooth_and_interp.interpolate_zeros(_cut_boundary_vals(fixed, 3), 'linear')
    interp = smooth_and_interp.interpolate_zeros(fixed,'linear')
    fixed[abs(interp-boundary_cut)>0.1] = 0
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
    min_win_len = 10 #20
    max_threshold = 3. #threshold with broad window

    min_threshold = 0.5 #threshold with shorted window
    
    if trace:
        import matplotlib
        import pylab
        pylab.rcParams['figure.figsize'] = 20, 5
        pylab.figure()
        pylab.title("outlier removal")

    
    _std = np.std(interp)
    # do not tie fixing to liveliness of the original
    _std = 0.3

    win_len =  np.exp(np.linspace(np.log(max_win_len),np.log(min_win_len), num_iter+1))
    outlier_threshold = np.linspace(_std*max_threshold, _std*min_threshold, num_iter+1)
    for i in range(0, num_iter):

        smooth_contour = smooth_and_interp.smooth(interp, win_len[i])
        low_limit = smooth_contour - outlier_threshold[i]
        hi_limit = smooth_contour + outlier_threshold[i]*1.5 # bit more careful upwards, not to cut emphases


        # octave jump down fix, more harm than good?
        #fixed[interp<smooth_contour-0.45]=interp[interp<smooth_contour-0.45]+0.5
        #fixed[interp>smooth_contour+0.45]=interp[interp>smooth_contour+0.45]-0.5
        fixed[interp>hi_limit] = 0
        fixed[interp<low_limit]= 0

        if trace:
            pylab.clf()
            pylab.title("outlier removal %d" % i)
            #pylab.ylim(3.5,7)
            pylab.plot((low_limit), 'black',linestyle='--')
            pylab.plot((hi_limit), 'black',linestyle='--')
            pylab.plot((smooth_contour), 'black',linestyle='--')
            pylab.plot((interp),linewidth=3)
            pylab.plot(lf0)
            pylab.show()

        interp = smooth_and_interp.interpolate_zeros(fixed,'linear')
   
    if trace:
        raw_input("press any key to continue")

    return fixed



def _interpolate(f0, method="true_envelope"):


    if method == "linear":
        return smooth_and_interp.interpolate_zeros(f0,'linear')
    elif method == "pchip":
        return smooth_and_interp.interpolate_zeros(f0, 'pchip')

    elif method == 'true_envelope':
        interp = smooth_and_interp.interpolate_zeros(f0)
       
        _std = np.std(interp)
        _min = np.min(interp)
        low_limit = smooth_and_interp.smooth(interp, 200)-1.5*_std
        low_limit[low_limit< _min] = _min
        hi_limit = smooth_and_interp.smooth(interp, 100)+2.0*_std
        voicing = np.array(f0)
        constrained = np.array(f0)
        constrained = np.maximum(f0,low_limit)
        constrained = np.minimum(constrained,hi_limit)

        interp = smooth_and_interp.peak_smooth(constrained, 100, 20,voicing=voicing)
        # smooth voiced parts a bit too
        interp =smooth_and_interp.peak_smooth(interp, 3, 2) #,voicing=raw)
        return interp
    else:
        raise("no such interpolation method: %s", method)

def reaper(in_wav_file, waveform, fs, f0_min, f0_max):
    if 1==1:
        import pyreaper
       
        #if len(waveform == 0):
        (fs, waveform) = misc.read_wav(in_wav_file)
       
        #print(pyreaper.reaper(waveform, fs, f0_min, f0_max))
        pm_times, pm,f0_times, f0, corr = pyreaper.reaper(waveform, fs, f0_min, f0_max)
        
    else:
        print("error")
        # use external REAPER pitch extraction binary if pyreaper not found
        import os
        _curr_dir = os.path.dirname(os.path.realpath(__file__))
        _reaper_bin = os.path.realpath(_curr_dir + '/../REAPER/build/reaper')
        out_est_file = "/tmp/tmp.f0"
        os.system(_reaper_bin + " -m %d -x %d -a -u 0.005 -i %s -f %s" % (f0_min, f0_max, in_wav_file, out_est_file))
        f0 = np.loadtxt(out_est_file, skiprows=7, usecols=[2])
        f0[f0<0] = 0.0
    return f0
    
def extract_f0(filename = "", waveform=[], fs=16000, f0_min = 0, f0_max = 0):

    #first determine f0 without limits, then use mean and std of the first estimate
    #to limit search range.  
    
    if f0_min==0 or f0_max == 0:
        f0 = reaper(filename, waveform, fs, 30, 550)
        #import pylab
        # pylab.hist(f0, 100)
        # pylab.show()
        # raw_input()
        mean_f0 = np.mean(f0[f0>0])
        std_f0 = np.std(f0[f0>0])
        f0_min = max((mean_f0 - 3*std_f0,40.0))
        f0_max = mean_f0 + 6*std_f0
        print(f0_min, f0_max)

    f0 = reaper(filename, waveform, fs,f0_min, f0_max)
    return f0
    
        
def process(f0, fix_outliers=True, interpolate=True, do_trace=False):

    lf0 = np.array(f0)
    log_scaled=True
    if np.mean(f0[f0>0]) > 20:
        log_scaled = False
        lf0[f0>0]=np.log(f0[f0>0])
        lf0[f0<=0] = 0

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
            print("reading F0 file", f0_f)
            try:
                # one f0 value per line
                return np.loadtxt(f0_f)
            except:
                # praat matrix
                try:
                    return np.loadtxt(f0_f, skiprows=4)
                except:
                    print("unknown format")


    return None




if __name__ == "__main__":

    import sys
    import pyreaper
    import scipy.io.wavfile
    import pylab
    pylab.ion()
    try:
        f0_glott = np.loadtxt(sys.argv[1]+".F0")
        pylab.plot(f0_glott, label="glott")
        pylab.plot(process(f0_glott, fix_outliers=False),label="glott_fixed", linewidth=2)
    except:
        print("no f0 file found, using reaper")
    import soundfile as wav
    
    (x, fs) = wav.read(sys.argv[1]+".wav")
    if type(x[0]) in[np.float16, np.float32, np.float64]:
        print(type(x[0]))
        x*=16000
        x=x.astype('int16')

    #raw_input()
    #fs, x = scipy.io.wavfile.read(sys.argv[1]+".wav")
    import os
    os.system("play "+sys.argv[1]+".wav")
    pm_times, pm, f0_times, f0_reaper, corr = pyreaper.reaper(x, fs, 50, 450)
    
    

   
    f0_adaptive = extract_f0(sys.argv[1]+".wav")

    #pylab.plot(f0_reaper, label="reaper")
    pylab.plot(f0_adaptive, label="reaper_adapt")

    #pylab.plot(process(f0_reaper, fix_outliers=True), label="reaper_fixed", linewidth=2)
    pylab.plot(process(f0_adaptive, fix_outliers=True, do_trace=False), label="reaper_adapt_fixed", linewidth=2)
    pylab.legend()
    pylab.show()
    
    raw_input()

import pylab
import numpy as np


def read_wav(filename):
    # various packages tried.. difficulties with channels, 24bit files, various dtypes
    # pysoundfile appears to mostly work
    
    import soundfile
    data, samplerate = soundfile.read(filename, dtype='int16', always_2d=True)

    return (samplerate, data[:, 0].copy(order='C'))

    """
    # import wavio
    # wav = wavio.read(filename)
    #print wav.data.shape
    #pylab.plot(wav.data[:,0])
    #return (wav.rate, wav.data[:, 0])


    import scipy.io.wavfile
    try:
        return scipy.io.wavfile.read(filename)
    except Exception as e:
        
        print e
    """
def write_wav(filename, data, sr, format="WAV"):
    import soundfile
    soundfile.write(filename, data, sr, format=format)
    

# scipy.signal.resample takes minutes with certain signal lengths like prime ones, 
# pad to next power of two
# note: in scipy >18 there is fft.pack.next_fast_len
def resample(signal, length):

    from scipy.signal import resample
    y = np.floor(np.log2(len(signal)))
    nextpow2 = np.power(2, y+1)
    padding = nextpow2-len(signal)
    ratio = nextpow2/len(signal)
    new_length = int(length*ratio)
    padding2 = int(new_length-length)

    signal_data  = np.pad(signal , (0,int(nextpow2-len(signal))), mode='constant')

    return resample(signal_data, new_length)[:-padding2]

def upsample(signal, length):
    from . import  smooth_and_interp
    return smooth_and_interp.interpolate_by_factor(signal, length/float(len(signal)))

def match_length(sig_list):
    length = min(map(len,sig_list))
 
    for i in range(0, len(sig_list)):
        sig_list[i] = sig_list[i][:int(length)]
    return sig_list
 

def get_peaks(params, threshold=0.001):
    #peaks = []
    indices = []
    threshold = 0.001
    import numpy as np
    zc = np.where(np.diff(np.sign(np.diff(params))))[0]
    
    indices = (np.diff(np.sign(np.diff(params))) < 0).nonzero()[0] +1
    
    peaks = params[indices]
    return np.array([peaks, indices])

def calc_prominence(params, labels, func=np.max, use_peaks = True, rate=200):
    labelled = []
    norm = params.astype(float)
    for (start, end, segment, word) in labels:
        if use_peaks:
            peaks = []
            (peaks, indices)=get_peaks(params[start*rate-1:end*rate],0.0)

            if len(peaks) >0:
                labelled.append(np.max(peaks))
            else:
                labelled.append(0.0)
        else:
            #labelled.append([word, func(params[start-10:end])])                                                                                              
            labelled.append(func(params[start*rate:end*rate]))
    return labelled



def get_best_scale(wavelet_matrix, num_units):
    best_i = 0
    best = 999
    for i in range(0, wavelet_matrix.shape[0]):
        num_peaks = len(get_peaks(wavelet_matrix[i])[0])

        dist= abs(num_peaks - num_units)
        if dist < best:
            best = dist
            best_i = i

    return best_i

def get_best_scale2(scales,labels):
    mean_length = 0
    num_words = 0
    for l in labels:
        mean_length+=(l[1]-l[0])

    mean_length/=len(labels)
    dist = scales-mean_length
    return np.argmin(np.abs(dist))
    #raw_input()
    #best_scale = 0
    #smallest_distance = 1000
    

    for i in range(len(scales)):
        #print i, scales[i], mean_length
        distance =  abs(scales[i]*3.5 - mean_length) 
        #if (scales[i]*3.0) > mean_length:
        #    print "yes", i-1
        #    return i-1
        if distance < smallest_distance:
            best_scale = i 
            smallest_distance = distance
    #raw_input()
    return best_scale




def play(utt):
    from pylab import ginput
    import os
    import sys
    wavfile = utt+".wav"
    wavfile = wavfile.replace(" ", "\ ") 
    st = 0.2
    end = 1
    i=0
    while (st  > 0.01):
        try:
            pts =ginput(1)
            st = pts[0][0] / 200.0
            end = 1.0
        except:
            continue
        os.system("play %s trim 0:0:%f 0:0:%f " %(wavfile, st, end))    

# between 0 and 1
def normalize2(params):
    return (params-min(params)+0.1)/(max(params)-min(params))

#z-score
def normalize(params, std=0):
    from scipy import stats
    if std ==0:
        std = np.nanstd(params)

    # empty array or all zeros
    #if std==0:
    if std < 0.00001: #np.isclose([std,0]):
        return np.zeros(len(params))

    mean = np.nanmean(params)
    
    return (params - mean) / float(std)


def read_grid(grid_file):
    segments = []
    syllables = []


    import tgt
    tg = tgt.read_textgrid(grid_file)

    try:
        stress_tier = tg.get_tier_by_name("stress")
        segment_tier = tg.get_tier_by_name("segment")
    except:
        stress_tier = tg.get_tier_by_name("Stress")
        segment_tier = tg.get_tier_by_name("Segment")
    

    prev_syl_mid = 0
    for s in segment_tier.annotations:
        if not hasattr(s, "text"):
            continue
        try:
            segments.append([s.start_time,s.end_time,s.text,""])
        except:
            segments.append([s.start_time,s.end_time," ",""])
            # stressed vowel
        stress = stress_tier.get_annotation_by_start_time(s.start_time)
        if hasattr(stress,"text"):

            syl_mid = stress.start_time+(stress.end_time-stress.start_time)/2.0
            start = segments[0][0]

                #end = 0

            if prev_syl_mid > 0:
                start = prev_syl_mid+(syl_mid-prev_syl_mid)/2.0
                syllables[-1][1] = start
                    
            syllables.append([start, 0, stress.text,""])
            prev_syl_mid = syl_mid
    syllables[-1][1] = segments[-1][1]
    return (segments, syllables)

def plot_labels(labels,shift = 0,  fig=pylab, text = True, ypos = 2, color="black", boundary=True,size=12,rate = 200):

    if fig == "":
        fig = pylab
        #print labels
    for (start, end, segment, token) in labels:
        
        if token:
            if boundary:
                fig.axvline(x=start*rate, color='gray',linestyle="-")
                fig.axvline(x=end*rate, color='gray',linestyle="-")
        # fig.axvline(x=end, color='gray',linestyle="--")
        if text: # and str(segment[0]) != "!":
            # fig.text(start+1-shift,5, token) #, color="grey")
            # fig.text(start-1+(end-start)/5-shift,ypos, segment, color=color, fontsize= 15)
            fig.text(int(start*rate)+1,ypos, segment, color=color,fontsize=size) #, color="grey")
            if boundary:
                fig.axvline(x=start*rate, color='gray',linestyle="-")
                fig.axvline(x=end*rate, color='gray',linestyle="-")
        fig.legend()
            

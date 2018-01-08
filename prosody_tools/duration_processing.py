from . import smooth_and_interp, misc
import numpy as np


SIL_SYMBOLS = ["#","!pau", "<s>", "pau", "!sil", "sil", "", " ","<p>", "<p:>", ".", ",","?"]

def get_dur_stats(labels, linear=False, sil_symbols=[]):
    durations = []
    for i in range(len(labels)):
        (st,en, unit) = labels[i]
        if unit.lower() not in sil_symbols:
            dur = en-st
            if not linear:
                
                dur = np.log(dur)
            durations.append(dur)
    durations = np.array(durations)
    return (np.min(durations), np.max(durations), np.mean(durations))


def get_rate(params,p=2,hp=10,lp=150, fig=None):
    """
    estimation of speech rate as a center of gravity of wavelet spectrum
    similar to method described in "Boundary Detection using Continuous Wavelet Analysis" (2016)
    """
    import prosody_tools.cwt_utils as cwt_utils
 
    params = smooth_and_interp.smooth(params, hp)
    params -= smooth_and_interp.smooth(params, lp)
    wavelet_matrix, scales  = cwt_utils.cwt_analysis(params, mother_name="Paul",num_scales=80, scale_distance=0.1, apply_coi=True,period=2)
    wavelet_matrix = abs(wavelet_matrix)

    rate = np.zeros(len(params))

    for i in range(0,wavelet_matrix.shape[1]):
        frame_en = np.sum(wavelet_matrix[:,i])
        # center of gravity
        rate[i] = np.nonzero(wavelet_matrix[:,i].cumsum() >=frame_en*0.5)[0].min()
        # maximum energy scale
        #rate[i]= np.argmax(wavelet_matrix[:,i]) #.astype('float'))

    if fig:
        fig.contourf((wavelet_matrix),50) 
    rate = smooth_and_interp.smooth(rate,30)
    if fig:
        fig.plot(rate,color="black") 

    return rate



def duration(labels, rate=200,linear=False,bump=False, sil_symbols=SIL_SYMBOLS):
    """ 
    construct duration signal from labels
    """
    if isinstance(labels,str):
        pass
        
    dur = np.zeros(len(labels))
    params = np.zeros(int(labels[-1][1]*rate))
    prev_end = 0
    (min_dur, max_dur, mean_dur) = get_dur_stats(labels,linear, sil_symbols)
    bump=True
    for i in range(0,len(labels)):
        
        (st,en, unit) = labels[i]
        st*=rate
        en*=rate
        dur[i] = en-st
        if not linear:
            dur[i] = np.log(dur[i]+1.)
        if unit.lower() in sil_symbols:

            dur[i] = min_dur #np.log(dur[i]+1.) #min_dur
            #continue
          
        if 1==1: #unit.lower() not in sil_symbols:

            params[int(st+(en-st)/2.0)] = dur[i]
            if bump:
                
                try:
                    #params[int(st)]= (dur[i]+dur[i-1])/4. #4 = arbitrary
                    if i > 0 and labels[i]: # not in sil_symbols:
                        params[int(st)]= (dur[i]+dur[i-1])/2.- (abs(dur[i]-dur[i-1])) #4 = arbitrary
                    #else:
                    #    params[int(st)]= (dur[i]+dur[i-1])/2
                except:
                    pass
                #params[int(st)]= 0.001
      
        # handle gaps in labels
        if  st > prev_end and i > 1:
            gap_dur = min_dur
            params[int(prev_end+(st-prev_end)/2.0)] = (gap_dur) #0.001 #-max_dur
        prev_end = en
    params[0] = np.mean(dur)
    params[-1] = np.mean(dur)
    #params = smooth_and_interp.interpolate_zeros(params, 'pchip')
    params = smooth_and_interp.interpolate_zeros(params, 'pchip')
    params = smooth_and_interp.smooth(params, 20)
    #params[1:]=np.diff(params)
    #params[0] = 0

    return params

def get_duration_signal(tiers =[], weights = [], sil_symbols=SIL_SYMBOLS, rate=1):

    durations = []
    lengths  = []
    for t in tiers:
        durations.append(misc.normalize(duration(t, rate=rate, sil_symbols=sil_symbols)))
    durations = misc.match_length(durations)
    sum_durations =np.zeros(len(durations[0]))
                            
    if len(weights)!=len(tiers):
        weights = np.ones(len(tiers))
    for i in range(len(durations)):
        sum_durations+=durations[i]*weights[i]
    #print(np.max(sum_durations))
    #return np.diff(sum_durations)
    return (sum_durations)
    
def get_durations_from_file(lab_file, rate=1, tiers = ["segments", "words"],sil_symbols = SIL_SYMBOLS):


    from . import lab
    labels = []
    if lab_file.lower().endswith("lab"):
        labels = lab.read_htk_label(lab_file)        
    
    elif lab_file.lower().endswith("textgrid"):
        pass
    if not labels:
        print("reading "+lab_file+" failed")
        return 
    durations = []
    lengths = []
    for t in tiers:

        if labels.has_key(t):
            #durations.append(duration(labels[t]))
            durations.append(misc.normalize(duration(labels[t], rate=rate, sil_symbols=sil_symbols)))
    


    if len(tiers) < 2:
        return (labels, durations)

    # if many tiers specified, normalize and sum them

    for d in durations:
        lengths.append(len(d))

    min_length = np.min(lengths)
    
    sum_durations =np.zeros(min_length)
    weights = [1.0, 1.0]
    for i in range(len(durations)):
        sum_durations+=durations[i][:min_length]*weights[i]
    return (labels, sum_durations)
            
    #import pylab
    #pylab.plot(sum_durations)
    #pylab.show()
    
if __name__ == "__main__":

    import sys
    import pylab
    
    labels, durations = get_durations(sys.argv[1])
    import lab
    lab.plot_labels(labels["words"])
    print(labels)
    pylab.plot(durations)
    #pylab.show()
    import cwt_utils
    cwt_matrix, scales = cwt_utils.cwt_analysis(durations, num_scales=12, scale_distance=1.0)
    pylab.contourf(cwt_matrix, 100)
    pylab.show()

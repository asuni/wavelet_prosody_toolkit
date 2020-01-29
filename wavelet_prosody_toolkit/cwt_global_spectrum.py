"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>
    - Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION

usage: cwt_global_spectrum.py input_file


Tool for extracting global wavelet spectrum of speech envelope, 
introduced for second language fluency estimation in the following paper:

@inproceedings{suni2019characterizing,
  title={Characterizing second language fluency with global wavelet spectrum},
  author={Suni, Antti and Kallio, Heini and Benu{\v{s}}, {\v{S}}tefan and {\v{S}}imko, Juraj},
  booktitle={International Congress of Phonetic Sciences},
  pages={1947--1951},
  year={2019},
  organization={Australasian Speech Science and Technology Association Inc.}
}

You should be able to see peak around 4Hz, corresponding to syllable rate.
For longer speech files, lower frequency peaks related to phrasing should appear.
Synthetic test file with 8Hz, 4Hz and 1Hz components is included in sample directory.


LICENSE
    See https://github.com/asuni/wavelet_prosody_toolkit/blob/master/LICENSE.txt

"""



import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
import sys, os
import time

from wavelet_prosody_toolkit.prosody_tools import cwt_utils as cwt_utils
from wavelet_prosody_toolkit.prosody_tools import misc as misc
from wavelet_prosody_toolkit.prosody_tools import energy_processing as energy_processing

PLOT =  True


def get_freq_labels(freqs):
    freq_list = [round(x,3)
                 if (np.isclose(x, round(x)) or
                     (x < 2 and np.isclose(x*100., round(x*100))) or
                     (x < 0.5 and np.isclose(x*10000., round(x*10000))))
                 else ""
                 for x in list(freqs)]
    return freq_list
    
    



def calc_global_spectrum(wav_file):
   

    # Extract signal envelope, scale and normalize
    (fs, waveform) = misc.read_wav(wav_file)
    waveform = misc.resample(waveform,fs, 16000)
    #plt.plot(abs(waveform))
    energy = energy_processing.extract_energy(waveform, min_freq = 30, method="hilbert")
    
    energy[energy<0] = 0
    energy = np.cbrt(energy+0.1)
    params = misc.normalize_std(energy)


    # perform continous wavelet transform on envelope with morlet wavelet

    # increase _period to get sharper spectrum
    _period = 5
    n_scales = 60
    matrix, scales,freq = cwt_utils.cwt_analysis(params, first_freq = 16, num_scales = n_scales, scale_distance  = 0.1,period=_period, mother_name="Morlet",apply_coi=True)


  

    # power, arbitrary scaling to prevent underflow
    p_matrix = (abs(matrix)**2).astype('float32')*1000.0
    power_spec = np.nanmean(p_matrix,axis=1)
   

    if PLOT:
        f, wave_pics = plt.subplots(len(sys.argv)-1,2, gridspec_kw = {'width_ratios':[5, 1]},  sharey=True)
        f.subplots_adjust(hspace=10)
        f.subplots_adjust(wspace=0)
        wave_pics[0].set_ylim(0, n_scales)
        wave_pics[0].set_xlabel("Time(m:s)")
        wave_pics[0].set_ylabel("Frequency(Hz)")
        wave_pics[1].set_xlabel("power")
        wave_pics[1].tick_params(labelright=True)
        fname = os.path.basename(wav_file)
        title = "CWT Morlet(p="+str(_period)+") global spectrum, "+ fname
        wave_pics[0].contourf(p_matrix, 100)
        wave_pics[0].set_title(title, loc="center")
        wave_pics[0].plot(params*3, color="white",alpha=0.5)
        freq_labels =get_freq_labels(freq)
        wave_pics[0].set_yticks(np.linspace(0, len(freq_labels)-1, len(freq_labels)))
        wave_pics[0].set_yticklabels(freq_labels)
        formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 200)))
        wave_pics[0].xaxis.set_major_formatter(formatter)
        wave_pics[1].grid(axis="y")
        wave_pics[1].plot(power_spec,np.linspace(0,len(power_spec), len(power_spec)),"-") 
        plt.show()


    # save spectrum and associated frequencies for further processing
    np.savetxt(wav_file[:-4]+".spec.txt",power_spec, fmt="%.5f", newline= " ")
    np.savetxt(wav_file[:-4]+".freqs.txt",freq, fmt="%.5f", newline= " ")
    

    
if len(sys.argv)==2 and sys.argv[1].lower().endswith(".wav"):
    calc_global_spectrum(sys.argv[1])
else:
    print("usage: cwt_global_spectrum.py <audiofile>")




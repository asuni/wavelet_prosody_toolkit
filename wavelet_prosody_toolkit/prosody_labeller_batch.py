


import sys,os, glob
import argparse
import yaml
from collections import defaultdict
import numpy as np

# - acoustic features
from wavelet_prosody_toolkit.prosody_tools import energy_processing
from wavelet_prosody_toolkit.prosody_tools import f0_processing
from wavelet_prosody_toolkit.prosody_tools import duration_processing

# - helpers
from wavelet_prosody_toolkit.prosody_tools import misc
from wavelet_prosody_toolkit.prosody_tools import smooth_and_interp

# - wavelet transform
from wavelet_prosody_toolkit.prosody_tools import cwt_utils, loma, lab

import matplotlib.pyplot as plt
from joblib import Parallel, delayed

PLOT = True

def analysis(input_file, cfg):
    print("analyzing", input_file)
    import librosa
    orig_sr, sig = misc.read_wav(input_file)
    #sig, orig_sr = librosa.load(input_file, sr=16000)

    

    
    ################
    # extract energy
    energy = energy_processing.extract_energy(sig, orig_sr,
                                              cfg["energy"]["band_min"],
                                              cfg["energy"]["band_max"],
                                              cfg["energy"]["calculation_method"])
    energy = np.cbrt(energy+1)
    if cfg["energy"]["smooth_energy"]:
        energy = smooth_and_interp.peak_smooth(energy, 30, 3)  # FIXME: 30? 3?
        energy = smooth_and_interp.smooth(energy, 10) 

        

    #############    
    # extract f0
    raw_pitch = f0_processing.extract_f0(sig, orig_sr,
                                         f0_min=cfg["f0"]["min_f0"],
                                         f0_max=cfg["f0"]["max_f0"],
                                         voicing=cfg["f0"]["voicing_threshold"],
                                         #harmonics=cfg["f0"]["harmonics"],
                                         configuration=cfg["f0"]["pitch_tracker"])
    # interpolate, stylize
    pitch = f0_processing.process(raw_pitch)

    
    ###################
    # extract speech rate
    rate = np.zeros(len(pitch))
    tiers = []
   
    
    grid = os.path.splitext(input_file)[0]+".TextGrid"
    if os.path.exists(grid):
        tiers = lab.read_textgrid(grid)
    else:
        grid = os.path.splitext(input_file)[0]+".lab"
        tiers = lab.read_htk_label(grid)

    if len(tiers) > 0:
        dur_tiers = []

        for level in cfg["duration"]["duration_tiers"]:

            assert(level.lower() in tiers), level+" not defined in tiers: check that duration_tiers in config match the actual textgrid tiers"
            try:
                dur_tiers.append(tiers[level.lower()])
            except:
                print("\nerror: "+"\""+level+"\"" +" not in labels, modify duration_tiers in config\n\n")
                raise
            
    if cfg["duration"]["acoustic_estimation"]==False:
        rate = duration_processing.get_duration_signal(dur_tiers,
                                                       weights=cfg["duration"]["weights"],
                                                       linear=cfg["duration"]["linear"],
                                                       sil_symbols=cfg["duration"]["silence_symbols"],
                                                       bump = cfg["duration"]["bump"])
        
    else:
        rate = duration_processing.get_rate(energy)
        rate = smooth_and_interp.smooth(rate, 30)

    if cfg["duration"]["delta_duration"]:
            rate = np.diff(rate)


    if PLOT:
        fig, ax =  plt.subplots(5,1, sharex=True)
        ax[0].plot(raw_pitch)
        ax[0].plot(pitch)
        ax[0].set_title("pitch")
        ax[1].plot(energy)
        ax[1].set_title("energy")
        ax[2].plot(rate)
        ax[2].set_title("speech rate")
        
    #################
    # combine signals
    min_length = np.min([len(pitch), len(energy), len(rate)])
    pitch = pitch[:min_length]
    energy = energy[:min_length]
    rate = rate[:min_length]

    if cfg["feature_combination"]["type"] == "product":
        pitch = misc.normalize_minmax(pitch) ** cfg["feature_combination"]["weights"]["f0"]
        energy = misc.normalize_minmax(energy) ** cfg["feature_combination"]["weights"]["energy"]
        rate =  misc.normalize_minmax(rate) ** cfg["feature_combination"]["weights"]["duration"]
        params = pitch * energy * rate

    else:
        params = misc.normalize_std(pitch) * cfg["feature_combination"]["weights"]["f0"] + \
                 misc.normalize_std(energy) * cfg["feature_combination"]["weights"]["energy"] + \
                 misc.normalize_std(rate) * cfg["feature_combination"]["weights"]["duration"]

    if cfg["feature_combination"]["detrend"]:
         params = smooth_and_interp.remove_bias(params, 800)


    params = misc.normalize_std(params)

    if PLOT:
        ax[3].plot(params)
        ax[3].set_title("combined signal")
        import scipy.ndimage
        ax[3].plot(params, "red")
        plt.xlim(0, len(params))
        #plt.show()

    

    ##############
    # CWT analysis
    (cwt, scales, freqs) = cwt_utils.cwt_analysis(params,
                                                  mother_name=cfg["wavelet"]["mother_wavelet"],
                                                  period=cfg["wavelet"]["period"],
                                                  num_scales=cfg["wavelet"]["num_scales"],
                                                  scale_distance=cfg["wavelet"]["scale_distance"],
                                                  apply_coi=True)
    cwt = np.real(cwt)
    scales*=200
    ###########
    # Lines of maximum amplitude

    assert(cfg["labels"]["annotation_tier"].lower() in tiers), \
        cfg["labels"]["annotation_tier"]+" not defined in tiers: check that annotation_tier in config is found in the textgrid tiers"
    labels = tiers[cfg["labels"]["annotation_tier"].lower()]

    n_scales = cfg["wavelet"]["num_scales"]
    scale_dist = cfg["wavelet"]["scale_distance"]
    # get scale corresponding to avg unit length of selected tier
    scales = (1./freqs*200)*0.5
    unit_scale = misc.get_best_scale2(scales, labels)

    # Define the scale information (FIXME: description)
    pos_loma_start_scale = unit_scale + int(cfg["loma"]["prom_start"]/scale_dist)  # three octaves down from average unit length
    pos_loma_end_scale = unit_scale + int(cfg["loma"]["prom_end"]/scale_dist)
    neg_loma_start_scale = unit_scale + int(cfg["loma"]["boundary_start"]/scale_dist)  # two octaves down
    neg_loma_end_scale = unit_scale + int(cfg["loma"]["boundary_end"]/scale_dist)  # one octave up

    pos_loma = loma.get_loma(cwt, scales, pos_loma_start_scale, pos_loma_end_scale)
    neg_loma = loma.get_loma(-cwt, scales, neg_loma_start_scale, neg_loma_end_scale)

    max_loma = loma.get_prominences(pos_loma, labels)
    prominences = np.array(max_loma)
    boundaries = np.array(loma.get_boundaries(max_loma, neg_loma, labels))


    ###############
    # output results
    loma.save_analyses(os.path.splitext(input_file)[0]+".prom",
                       labels,
                       prominences,
                       boundaries)

    if PLOT:
        prom_text =  prominences[:, 1]/(np.max(prominences[:, 1]))*2.5 + 0.5
        lab.plot_labels(labels, ypos=0.5, size=6, prominences=prom_text,fig=ax[4], boundary=True)


        fig, ax =  plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[3, 1]})   
        cwt[cwt>0] = np.log(cwt[cwt>0]+1.)
        cwt[cwt<-0.1] = -0.1
        ax[0].contourf(cwt,100, cmap="jet")
        loma.plot_loma(pos_loma, ax[0], color="black")
        loma.plot_loma(neg_loma, ax[0], color="white")
        

        lab.plot_labels(labels, ypos=0.5, size=6, prominences=prom_text,fig=ax[1], boundary=True)

        for i in range(0, len(labels)):
            ax[1].axvline(x=labels[i][1], color='black',
                          linestyle="-", linewidth=boundaries[i][-1] * 4,
                          alpha=0.5)
        os.system("play "+input_file+" &")
        plt.xlim(0, cwt.shape[1])
        plt.show()



##############################################################################################
# Configuration utilities
##############################################################################################
def apply_configuration(current_configuration, updating_part):
    """Utils to update the current configuration using the updating part

    Parameters
    ----------
    current_configuration: dict
        The current state of the configuration

    updating_part: dict
        The information to add to the current configuration

    Returns
    -------
    dict
       the updated configuration
    """
    if not isinstance(current_configuration, dict):
        return updating_part

    if current_configuration is None:
        return updating_part

    if updating_part is None:
        return current_configuration

    for k in updating_part:
        if k not in current_configuration:
            current_configuration[k] = updating_part[k]
        else:
            current_configuration[k] = apply_configuration(current_configuration[k], updating_part[k])

    return current_configuration



def analysis_wrap(f, configuration):
    #try:
    analysis(f, configuration)
    #except:
    #    print("analysis of "+f+" failed.")
##############################################################################################
# Main routine definition
##############################################################################################
def main():


    parser = argparse.ArgumentParser(description="Command line application to analyze prosody using wavelets.")

    # Load default configuration
    parser.add_argument("-c", "--config", default=None, help="configuration file")
  
    parser.add_argument("input_dir", help="directory with wave files to analyze (a label file with the same basename should be available)")
    # Parsing arguments
    args = parser.parse_args()


    # Load configuration
    configuration = defaultdict()
    with open(os.path.dirname(os.path.realpath(__file__)) + "/configs/default.yaml", 'r') as f:
        configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.load(f)))

    if args.config:
        try:
            with open(args.config, 'r') as f:
                configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.load(f)))
        except IOError as ex:
            print("configuration file " + args.config + " could not be loaded:")

            sys.exit(1)
    print(configuration)
    #analysis("/home/asuni/work/toy_corp/rjs/rjs_01_0003.wav", configuration)
    input_files = glob.glob(args.input_dir+"/*.wav")

    
    num_cores = 16
    if PLOT == True:
        num_cores = 1
    Parallel(n_jobs=num_cores)(delayed(analysis_wrap)(f, configuration) for f in input_files)
    """
    i = 0
    for f in files:
        if i % 1 == 0:
            try:
                analysis(f, configuration)
            except:
                print(f+" failed")
        i+=1
    """
    #analysis("/home/asuni/work/aligner/montreal-forced-aligner/libritts-dev-alignments/1272_128104_000003_000000.wav", configuration)
        # Debug time



###############################################################################
#  Envelopping
###############################################################################
if __name__ == '__main__':
    main()

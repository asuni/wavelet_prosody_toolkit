#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR

    Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION

LICENSE
    This script is in the public domain, free from copyrights or restrictions.
    Created: 27 January 2020
"""

# System/default
import sys
import os
import glob

# Arguments
import argparse

# Messaging/logging
import traceback
import time
import logging
import copy

# Configuration
import yaml
from collections import defaultdict

# Math and plotting
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# Parallel job managment
from joblib import Parallel, delayed

# acoustic features
from wavelet_prosody_toolkit.prosody_tools import energy_processing
from wavelet_prosody_toolkit.prosody_tools import f0_processing
from wavelet_prosody_toolkit.prosody_tools import duration_processing

# helpers
from wavelet_prosody_toolkit.prosody_tools import misc
from wavelet_prosody_toolkit.prosody_tools import smooth_and_interp

# wavelet transform
from wavelet_prosody_toolkit.prosody_tools import cwt_utils, loma, lab

###############################################################################
# global constants
###############################################################################
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]

###############################################################################
# Functions
###############################################################################
def get_logger(verbosity, log_file):

    # create logger and formatter
    logger = logging.getLogger("prosody labeller")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Verbose level => logging level
    log_level = verbosity
    if (log_level >= len(LEVEL)):
        log_level = len(LEVEL) - 1
        logger.setLevel(log_level)
        logging.warning("verbosity level is too high, I'm gonna assume you're taking the highest (%d)" % log_level)
    else:
        logger.setLevel(LEVEL[log_level])

    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    return logger


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



def analysis(input_file, cfg, logger, annotation_dir=None, output_dir=None, plot=False):

    # Load the wave file
    print("Analyzing %s starting..." % input_file)
    orig_sr, sig = misc.read_wav(input_file)

    # extract energy
    energy = energy_processing.extract_energy(sig, orig_sr,
                                              cfg["energy"]["band_min"],
                                              cfg["energy"]["band_max"],
                                              cfg["energy"]["calculation_method"])
    energy = np.cbrt(energy+1)
    if cfg["energy"]["smooth_energy"]:
        energy = smooth_and_interp.peak_smooth(energy, 30, 3)  # FIXME: 30? 3?
        energy = smooth_and_interp.smooth(energy, 10)

    # extract f0
    raw_pitch = f0_processing.extract_f0(sig, orig_sr,
                                         f0_min=cfg["f0"]["min_f0"],
                                         f0_max=cfg["f0"]["max_f0"],
                                         voicing=cfg["f0"]["voicing_threshold"],
                                         #harmonics=cfg["f0"]["harmonics"],
                                         configuration=cfg["f0"]["pitch_tracker"])
    # interpolate, stylize
    pitch = f0_processing.process(raw_pitch)

    # extract speech rate
    rate = np.zeros(len(pitch))


    # Get annotations (if available)
    tiers = []
    if annotation_dir is None:
        annotation_dir = os.path.dirname(input_file)
    basename = os.path.splitext(os.path.basename(input_file))[0]
    grid =  os.path.join(annotation_dir, "%s.TextGrid" % basename)
    if os.path.exists(grid):
        tiers = lab.read_textgrid(grid)
    else:
        grid =  os.path.join(annotation_dir, "%s.lab" % basename)
        if not os.path.exists(grid):
            raise Exception("There is no annotations associated with %s" % input_file)
        tiers = lab.read_htk_label(grid)

    # Extract duration
    if len(tiers) > 0:
        dur_tiers = []
        for level in cfg["duration"]["duration_tiers"]:
            assert(level.lower() in tiers), level+" not defined in tiers: check that duration_tiers in config match the actual textgrid tiers"
            try:
                dur_tiers.append(tiers[level.lower()])
            except:
                print("\nerror: "+"\""+level+"\"" +" not in labels, modify duration_tiers in config\n\n")
                raise

    if not cfg["duration"]["acoustic_estimation"]:
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

    # Combine signals
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


    # CWT analysis
    (cwt, scales, freqs) = cwt_utils.cwt_analysis(params,
                                                  mother_name=cfg["wavelet"]["mother_wavelet"],
                                                  period=cfg["wavelet"]["period"],
                                                  num_scales=cfg["wavelet"]["num_scales"],
                                                  scale_distance=cfg["wavelet"]["scale_distance"],
                                                  apply_coi=False)
    cwt = np.real(cwt)
    scales *= 200 # FIXME: why 200?


    # Compute lines of maximum amplitude
    assert(cfg["labels"]["annotation_tier"].lower() in tiers), \
        cfg["labels"]["annotation_tier"]+" not defined in tiers: check that annotation_tier in config is found in the textgrid tiers"
    labels = tiers[cfg["labels"]["annotation_tier"].lower()]

    # get scale corresponding to avg unit length of selected tier
    n_scales = cfg["wavelet"]["num_scales"]
    scale_dist = cfg["wavelet"]["scale_distance"]
    scales = (1./freqs*200)*0.5 # FIXME: hardcoded vales
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


    # output results
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = os.path.join(output_dir, "%s.prom" % basename)
    print("Saving %s..." % (output_filename))
    loma.save_analyses(output_filename,
                       labels,
                       prominences,
                       boundaries)

    # Plotting
    if plot != 0:
        fig, ax =  plt.subplots(6, 1, sharex=True,
                                figsize=(len(labels) / 10 * 8, 8),
                                gridspec_kw = {'height_ratios':[1, 1, 1, 2, 4, 1.5]})
        plt.subplots_adjust(hspace=0)

        # Plot individual signals
        ax[0].plot(pitch, linewidth=1)
        ax[0].set_ylabel("Pitch", rotation="horizontal", ha="right", va="center")

        ax[1].plot(energy, linewidth=1)
        ax[1].set_ylabel("Energy", rotation="horizontal", ha="right", va="center")

        ax[2].plot(rate, linewidth=1)
        ax[2].set_ylabel("Speech rate", rotation="horizontal", ha="right", va="center")

        # Plot combined signal
        ax[3].plot(params, linewidth=1)
        ax[3].set_ylabel("Combined \n signal", rotation="horizontal", ha="right", va="center")
        plt.xlim(0, len(params))

        # Wavelet and loma
        cwt[cwt>0] = np.log(cwt[cwt>0]+1.)
        cwt[cwt<-0.1] = -0.1
        ax[4].contourf(cwt,100, cmap="inferno")
        loma.plot_loma(pos_loma, ax[4], color="black")
        loma.plot_loma(neg_loma, ax[4], color="white")
        ax[4].set_ylabel("Wavelet & \n LOMA", rotation="horizontal", ha="right", va="center")
        
        # Add labels
        prom_text =  prominences[:, 1]/(np.max(prominences[:, 1]))*2.5 + 0.5
        lab.plot_labels(labels, ypos=0.3, size=6, prominences=prom_text, fig=ax[5], boundary=False, background=False)
        ax[5].set_ylabel("Labels", rotation="horizontal", ha="right", va="center")
        for i in range(0, len(labels)):
            for a in [0, 1, 2, 3, 4, 5]:
                ax[a].axvline(x=labels[i][0], color='black',
                              linestyle="-", linewidth=0.2, alpha=0.5)
                
                ax[a].axvline(x=labels[i][1], color='black',
                              linestyle="-", linewidth=0.2+boundaries[i][-1] * 2,
                              alpha=0.5)

        plt.xlim(0, cwt.shape[1])
    
        # Align ylabels and remove axis
        fig.align_ylabels(ax)
        for i in range(len(ax)-1):
            ax[i].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            ax[i].tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=False) # labels along the bottom edge are off

        ax[len(ax)-1].tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False) # labels along the bottom edge are off

        # Plot
        if plot < 0:
            output_filename = os.path.join(output_dir, "%s.png" % basename)
            logger.info("Save plot %s" % output_filename)
            fig.savefig(output_filename, bbox_inches='tight', dpi=400)
        elif plot > 0:
            plt.show()

def analysis_batch_wrap(input_file, cfg, annotation_dir=None, output_dir=None, plot=0, logger=None):
    # Encapsulate running
    try:
        print(".")
        analysis(input_file, cfg, logger, annotation_dir, output_dir, plot)
    except Exception as ex:
        logging.error(str(ex))
        traceback.print_exc(file=sys.stderr)


###############################################################################
# Main function
###############################################################################
def main():
    """Main entry function
    """
    global args, logger

    # Load configuration
    configuration = defaultdict()
    with open(os.path.dirname(os.path.realpath(__file__)) + "/configs/default.yaml", 'r') as f:
        configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.load(f, Loader=yaml.FullLoader)))

    if args.config:
        try:
            with open(args.config, 'r') as f:
                configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.load(f, Loader=yaml.FullLoader)))
        except IOError as ex:
            print("configuration file " + args.config + " could not be loaded:")

            sys.exit(1)
    logger.debug("Current confirugration:")
    logger.debug(configuration)

    # Get the number of jobs
    nb_jobs = args.nb_jobs

    # Loading files
    if os.path.isfile(args.input):
        input_files = [args.input]
    else:
        input_files = glob.glob(args.input + "/*.wav")
    if len(input_files) == 1:
        nb_jobs = 1

    plot_flag = 0
    if nb_jobs > 1:
        if args.plot:
            plot_flag = -1
        Parallel(n_jobs=nb_jobs, verbose=args.verbosity)(delayed(analysis_batch_wrap)(f, configuration, args.annotation_directory, args.output_directory, plot_flag, logger) for f in input_files)
    else:
        if args.plot:
            plot_flag = 1
        for f in input_files:
            analysis(f, configuration, logger, args.annotation_directory, args.output_directory, plot_flag)


###############################################################################
#  Envelopping
###############################################################################
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Command line application to analyze prosody using wavelets.")

        # Add options
        parser.add_argument("-a", "--annotation_directory", default=None, type=str,
                            help="Annotation directory. If not specified, the tool will by default try to load annotations from the directory containing the wav files")
        parser.add_argument("-j", "--nb_jobs", default=4, type=int,
                            help="Define the number of jobs to run in parallel")
        parser.add_argument("-c", "--config", default=None, type=str,
                            help="configuration file")
        parser.add_argument("-l", "--log_file", default=None, type=str,
                            help="Logger file")
        parser.add_argument("-o", "--output_directory", default=None, type=str,
                            help="The output directory. If not specified, the tool will output the result in a .prom file in the same directory than the wave files")
        parser.add_argument("-p", "--plot", default=False, action="store_true",
                            help="Plot the result (the number of jobs is de facto set to 1 if activated)")
        parser.add_argument("-v", "--verbosity", action="count", default=1,
                            help="increase output verbosity")

        # Add arguments
        parser.add_argument("input", help="directory with wave files or wave file to analyze (a label file with the same basename should be available)")



        # Parsing arguments
        args = parser.parse_args()
        if args.plot:
            args.nb_jobs = 1
        # Get the logger
        logger = get_logger(args.verbosity, args.log_file)

        # Debug time
        start_time = time.time()
        logger.info("start time = " + time.asctime())

        # Running main function <=> run application
        main()

        # Debug time
        logger.info("end time = " + time.asctime())
        logger.info('TOTAL TIME IN MINUTES: %02.2f' %
                     ((time.time() - start_time) / 60.0))

        # Exit program
        sys.exit(0)
    except KeyboardInterrupt as e:  # Ctrl-C
        raise e
    except SystemExit:  # sys.exit()
        pass
    except Exception as e:
        logging.error('ERROR, UNEXPECTED EXCEPTION')
        logging.error(str(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(-1)

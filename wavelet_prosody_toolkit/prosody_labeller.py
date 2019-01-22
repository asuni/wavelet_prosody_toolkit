#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>
    - Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION

usage: prosody_labeller.py [-h] [-v] [-H] [-P] [-l LEVEL] [-L LABEL]
                           [-n NB_SCALES] [-d SCALE_DIST] [-f SCALE_FACTOR]
                           input_file output_file

Prosody events labeller tool based on wavelet transformation

positional arguments:
  - input_file            input wave file to analyze (a label file with the same basename should be available)
  - output_file           Output file which contains the events in a csv format

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbosity       Increase output verbosity
  -H, --with-header     Also print the header
  -P, --plot            Plot the results
  -l LEVEL, --level LEVEL
                        The analyzed level
  -L LABEL, --label LABEL
                        Alternative label filename
  -n NB_SCALES, --nb-scales NB_SCALES
                        The number of scales for the cwt
  -d SCALE_DIST, --scale-dist SCALE_DIST
                        The distance between scales
  -f SCALE_FACTOR, --scale-factor SCALE_FACTOR
                        Scaling factor

LICENSE
    See https://github.com/seblemaguer/wavelet_prosody_toolkit/blob/master/LICENSE.txt
"""

# System
import sys
import os
import os.path
import warnings

# Debug
import traceback

# Arguments
import argparse

# Logging
import time
import logging
logging.basicConfig(level=logging.WARN)

# extraction and preprocessing of prosodic signals
from wavelet_prosody_toolkit.prosody_tools import f0_processing, energy_processing, duration_processing, smooth_and_interp

# labels
from wavelet_prosody_toolkit.prosody_tools import lab

# wavelet
from wavelet_prosody_toolkit.prosody_tools import cwt_utils, loma, misc

import numpy as np

try:
    import pylab
except Exception as ex:
    logging.info("pylab is not available, so plotting is not available")


# List of logging levels used to setup everything using verbose option
LOG_LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]


###############################################################################
# Functions
###############################################################################
def extract_signal_prosodic_feature(input_file):
    """Extract energy (smoothed) and pitch from an input wav file.

    Parameters
    ----------
    input_file: string
        Filename of the input wav file

    """
    # read waveform
    orig_sr, sig = misc.read_wav(input_file)

    # extract energy
    energy = energy_processing.extract_energy(sig, orig_sr, 300, 5000)
    energy_smooth = smooth_and_interp.peak_smooth(energy, 30, 3)

    # extract f0
    #raw_pitch = f0_processing.extract_f0(sig, orig_sr)
    raw_pitch = f0_processing.extract_f0(input_file) #, orig_sr)
    pitch = f0_processing.process(raw_pitch)

    return (energy_smooth, pitch)


def extract_speech_rate(labels):
    """Extract speech rate for a give list of labels.

    Parameters
    ----------
    labels: list of tuple (float, float, string)
        list of labels which are lists of 3 elements [start, end, description]

    """
     # extract speech rate (from signal)
    try:
        rate = duration_processing.get_duration_signal([labels]) #, labels['segments']])
    except:
        rate = duration_processing.get_rate(energy)
        rate = smooth_and_interp.smooth(rate, 30)
    rate = np.diff(rate)

    return rate


def extract_params(input_file, labels):
    """Extract prosodic params from wav file and corresponding labels.

    Parameters
    ----------
    input_file: string
        Filename of the input wav file
    labels: list of tuple (float, float, string)
        list of labels which are lists of 3 elements [start, end, description]

    """
     # Extract acoustic part from input wav file.
    (energy_smooth, pitch) = extract_signal_prosodic_feature(input_file)

    # Extract speech rate from labels
    rate = extract_speech_rate(labels)

    # combine feats
    (pitch, energy_smooth, rate) = misc.match_length([pitch, energy_smooth, rate])
    params = misc.normalize_std(pitch)+ \
             misc.normalize_std(energy_smooth)+ \
             misc.normalize_std(rate)
    params = misc.normalize_std(params)

    return (params, pitch, energy_smooth, rate)


def label_prosody(scales, cwt, labels):
    """Label prosody event based on wavelet transform

    Parameters
    ----------
    scales: vector of float (?)
        The wavelet scales
    cwt: matrix of float (?)
        The wavelet coefficients
    labels: list of tuple (float, float, string)
        List of labels which are lists of 3 elements [start, end, description]

    """
    # get scale corresponding to word length
    level_scale = misc.get_best_scale(np.real(cwt), len(labels))


    pos_loma = loma.get_loma(np.real(cwt),scales, level_scale-4, level_scale+4)
    neg_loma = loma.get_loma(-np.real(cwt),scales, level_scale-4, level_scale+4)

    prominences = loma.get_prominences(pos_loma, labels)
    boundaries = loma.get_boundaries(prominences, neg_loma, labels)

    return (prominences, boundaries, pos_loma, neg_loma)


def plot(labels, rate, energy_smooth, pitch, params, cwt, boundaries, prominences, pos_loma, neg_loma):
    """Plot all the elements

    Parameters
    ----------
    labels: list of tuple (float, float, string)
        List of labels which are lists of 3 elements [start, end, description]
    rate: type
        description
    energy_smooth: type
        description
    pitch: type
        description
    params: type
        description
    cwt: type
        description
    boundaries: type
        description
    prominences: type
        description
    pos_loma: type
        description
    neg_loma: type
        description


    """
    f, axarr = pylab.subplots(2, sharex=True)
    axarr[0].set_title("Acoustic Features")
    shift = 0
    axarr[0].plot(params, label="combined")

    shift = 4
    axarr[0].plot(misc.normalize_std(rate)+shift, label="rate (shift=%d)" % shift)

    shift = 7
    axarr[0].plot(misc.normalize_std(energy_smooth)+shift, label="energy(shift=%d)" % shift)

    shift = 10
    axarr[0].plot(misc.normalize_std(pitch)+shift, label="f0 (shift=%d)" % shift)
    axarr[0].set_xlim(0,len(params))
    l = axarr[0].legend(fancybox=True)
    l.get_frame().set_alpha(0.75)

    axarr[1].set_title("Continuous Wavelet Transform")
    axarr[1].contourf(cwt, 100)
    loma.plot_loma(pos_loma, color='black', fig=axarr[1])
    loma.plot_loma(neg_loma, color='white', fig=axarr[1])

    lab.plot_labels(labels, ypos=1., prominences= np.array(prominences)[:,1],  fig=axarr[1])
    pylab.show()


###############################################################################
# Main function
###############################################################################
def run():
    """Main function which wraps the prosody labeller tool

    """
    global args

    # Extract labels
    if args.label:
        lab_f = args.label
    else:
        lab_f = os.path.splitext(args.input_file)[0]+".lab"

    if os.path.exists(lab_f):
        labels = lab.read_htk_label(lab_f)
        labels = labels[args.level] # Filter by level
    else:
        logging.error("Label file \"%s\" doesn't exist" % lab_f)
        sys.exit(-1)

    # Extract parameters
    (params, pitch, energy_smooth, rate) = extract_params(args.input_file, labels)

    # perform wavelet transform
    (cwt,scales) = cwt_utils.cwt_analysis(params, mother_name="mexican_hat", period=2,
                                          num_scales=args.nb_scales, scale_distance=args.scale_dist,
                                          apply_coi=True)
    scales *= args.scale_factor

    # Labelling prominences and boundarys
    (prominences, boundaries, pos_loma, neg_loma) = label_prosody(scales, cwt, labels)

    print("========================================================")
    print("label\tprominence\tboundary")
    for i in range(0, len(labels)):
        print("%s\t%f\t%f" %(labels[i][-1], prominences[i][-1], boundaries[i][-1]))

    if args.plot:
        warnings.simplefilter("ignore", np.ComplexWarning) # Plotting can't deal with complex, but we don't care
        plot(labels, rate, energy_smooth, pitch, params, cwt, boundaries, prominences, pos_loma, neg_loma)


###############################################################################
#  Envelopping
###############################################################################
def main():
    """Entry point for the prosody labeller tool

    This function is a wrapper to deal with arguments and logging.
    """
    global args

    try:
        parser = argparse.ArgumentParser(description="Prosody events labeller tool based on wavelet transformation")

        # General options
        parser.add_argument("-v", "--verbosity", action="count", default=0,
                            help="Increase output verbosity")
        parser.add_argument("-H", "--with-header", action="store_true",
                            help="Also print the header")
        parser.add_argument("-P", "--plot", action="store_true",
                            help="Plot the results")
        parser.add_argument("-l", "--level", default="words", help="The analyzed level")
        parser.add_argument("-L", "--label", help="Alternative label filename")

        # Scales options
        parser.add_argument("-n", "--nb-scales", default=20, type=int,
                            help="The number of scales for the cwt")
        parser.add_argument("-d", "--scale-dist", default=0.5, type=float,
                            help="The distance between scales")
        parser.add_argument("-f", "--scale-factor", default=200, type=int,
                            help="Scaling factor")

        # Add arguments
        parser.add_argument("input_file", help="input wave file to analyze (a label file with the same basename should be available)")
        parser.add_argument("output_file", help="Output file which contains the events in a csv format")

        # Parsing arguments
        args = parser.parse_args()

        # Verbose level => logging level
        log_level_idx = args.verbosity
        if (args.verbosity > len(LOG_LEVEL)):
            logging.warning("verbosity level is too high, I'm gonna assume you're taking the highes ")
            log_level_idx = len(LOG_LEVEL) - 1
            print(LOG_LEVEL[log_level_idx])
        logging.getLogger().setLevel(LOG_LEVEL[log_level_idx])

        # Debug time
        start_time = time.time()
        logging.info("start time = " + time.asctime())

        # Running main function <=> run application
        run()

        # Debug time
        logging.info("end time = " + time.asctime())
        logging.info('TOTAL TIME IN MINUTES: %02.2f' %
                     ((time.time() - start_time) / 60.0))

        # Exit program
        sys.exit(0)
    except KeyboardInterrupt as e:  # Ctrl-C
        raise e
    except SystemExit as e:  # sys.exit()
        pass
    except Exception as e:
        logging.error('ERROR, UNEXPECTED EXCEPTION')
        logging.error(str(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(-1)


if __name__ == '__main__':
    main()


# prosody_labeller_command_line.py ends here

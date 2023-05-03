#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>
    - Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION

usage: cwt_analysis_synthesis.py [-h] [-v] [-M MODE] [-m MEAN_F0] [-o OUTPUT]
                                 [-P]
                                 input_file

Tool for CWT analysis/synthesis of the F0

positional arguments:
  input_file            Input signal or F0 file

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbosity       increase output verbosity
  -M MODE, --mode MODE  script mode: 0=analysis, 1=synthesis, 2=analysis/synthesis
  -m MEAN_F0, --mean_f0 MEAN_F0
                        Mean f0 needed for synthesis (unsed for analysis modes)
  -o OUTPUT, --output OUTPUT
                        output directory for analysis or filename for synthesis.
                        (Default: input_file directory [Analysis] or <input_file>.f0 [Synthesis])
  -P, --plot            Plot the results


LICENSE
    See https://github.com/asuni/wavelet_prosody_toolkit/blob/master/LICENSE.txt
"""

import sys
import os
import traceback
import argparse
import time
import logging

import yaml

# Collections
from collections import defaultdict

import warnings

# Wavelet import
from wavelet_prosody_toolkit.prosody_tools import misc
from wavelet_prosody_toolkit.prosody_tools import cwt_utils
from wavelet_prosody_toolkit.prosody_tools import f0_processing

import numpy as np

# List of logging levels used to setup everything using verbose option
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]

# FIXME: be more specific!
warnings.simplefilter("ignore", np.ComplexWarning)  # Plotting can't deal with complex, but we don't care


###############################################################################
# Functions
###############################################################################
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


def load_f0(input_file, binary_mode=False, configuration=None):
    """Load the f0 from a text file or extract it from a wav file

    Parameters
    ----------
    input_file: string
        The input file name.

    Returns
    -------
    1D arraylike
       the raw f0 values
    """
    if input_file.lower().endswith(".csv"):
        if binary_mode:
            raise Exception("cannot have a csv file in binary mode")
        else:
            raw_f0 = np.loadtxt(input_file)
    if input_file.lower().endswith(".f0"):
        if binary_mode:
            raw_f0 = np.fromfile(input_file, dtype=np.float32)
        else:
            raw_f0 = np.loadtxt(input_file)
    elif input_file.lower().endswith(".lf0"):
        if binary_mode:
            raw_f0 = np.fromfile(input_file, dtype=np.float32)
        else:
            raw_f0 = np.loadtxt(input_file)
        raw_f0 = np.exp(raw_f0)
    elif input_file.lower().endswith(".wav"):
        logging.info("Extracting the F0 from the signal")
        (fs, wav_form) = misc.read_wav(input_file)
        raw_f0 = f0_processing.extract_f0(wav_form, fs,
                                          configuration["f0"]["min_f0"],
                                          configuration["f0"]["max_f0"])


    return raw_f0


###############################################################################
# Main function
###############################################################################
def run():
    """Main entry function

    This function contains the code needed to achieve the analysis and/or the synthesis
    """
    global args

    warnings.simplefilter("ignore", FutureWarning)     # Plotting can't deal with complex, but we don't care

    # Loading default configuration
    configuration = defaultdict()
    with open(os.path.dirname(os.path.realpath(__file__)) + "/configs/default.yaml", 'r') as f:
        configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.safe_load(f)))
        logging.debug("default configuration")
        logging.debug(configuration)

    # Loading dedicated analysis.synthesis configuration
    with open(os.path.dirname(os.path.realpath(__file__)) + "/configs/synthesis.yaml", 'r') as f:
        configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.safe_load(f)))
        logging.debug("configuration filled with synthesis part")
        logging.debug(configuration)

    # Loading user configuration
    if args.configuration_file:
        try:
            with open(args.configuration_file, 'r') as f:
                configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.safe_load(f)))
                logging.debug("configuration filled with user part")
                logging.debug(configuration)
        except IOError as ex:
            logging.error("configuration file " + args.config + " could not be loaded:")
            logging.error(ex.msg)
            sys.exit(1)

    # Analysis Mode
    if args.mode == 0:
        raw_f0 = load_f0(args.input_file, args.binary_mode, configuration)

        logging.info("Processing f0")
        f0 = f0_processing.process(raw_f0)
        # FIXME: reintegrated
        if args.plot:
            # Plotting
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors

            plt.title("F0 preprocessing and interpolation")
            plt.plot(f0, color="red", alpha=0.5, linewidth=3)
            plt.plot(raw_f0, color="gray", alpha=0.5)
            plt.show()

        # # FIXME: read this?
        # logging.info("writing interpolated lf0\t" + output_file + ".interp")
        # np.savetxt(output_file + ".interp", f0.astype('float'),
        #            fmt="%f", delimiter="\n")

        # Perform continuous wavelet transform of mean-substracted f0 with 12 scales, one octave apart
        logging.info("Starting analysis with (num_scale=%d, scale_distance=%f, mother_name=%s)" %
                     (configuration["wavelet"]["num_scales"], configuration["wavelet"]["scale_distance"], configuration["wavelet"]["mother_wavelet"]))
        full_scales, widths, _ = cwt_utils.cwt_analysis(f0 - np.mean(f0),
                                                        mother_name=configuration["wavelet"]["mother_wavelet"],
                                                        period=configuration["wavelet"]["period"],
                                                        num_scales=configuration["wavelet"]["num_scales"],
                                                        scale_distance=configuration["wavelet"]["scale_distance"],
                                                        apply_coi=False)
        full_scales = np.real(full_scales)
        # SSW parameterization, adjacent scales combined (with extra scales to handle long utterances)
        scales = cwt_utils.combine_scales(np.real(full_scales), configuration["wavelet"]["combined_scales"])
        for i in range(0, len(scales)):
            logging.debug("Mean scale[%d]: %s" % (i, str(np.mean(scales[i]))))

        # Saving matrix
        logging.info("writing wavelet matrix in \"%s\"" % args.output_file)
        if args.binary_mode:
            with open(args.output_file, "wb") as f_out:
                scales.T.astype(np.float32).tofile(f_out)
        else:
            np.savetxt(args.output_file, scales.T.astype('float'), fmt="%f", delimiter=",")

    # Synthesis mode
    if args.mode == 1:
        if args.binary_mode:
            scales = np.fromfile(args.input_file, dtype=np.float32)
            scales = scales.reshape(-1, len(configuration["wavelet"]["combined_scales"])).T
        else:
            scales = np.loadtxt(args.input_file, delimiter=",").T  # FIXME: hardcoded

        rec = cwt_utils.cwt_synthesis(scales, args.mean_f0)

        logging.info("Save reconstructed f0 in %s" % args.output_file)
        if args.binary_mode:
            with open(args.output_file, "wb") as f_out:
                rec.astype(np.float32).tofile(f_out)
        else:
            np.savetxt(args.output_file, rec, fmt="%f")

    # Debugging /plotting part
    if args.plot:
        nb_sub = 2
        if args.mode == 0:
            nb_sub = 3

        ax = plt.subplot(nb_sub, 1, 1)
        # pylab.title("CWT decomposition to % scales and reconstructed signal" % len(configuration["wavelet"]["combined_scales"]))

        if args.mode == 0:
            plt.plot(f0, linewidth=1, color="red")
            rec = cwt_utils.cwt_synthesis(scales, np.mean(f0))

        plt.plot(rec, color="blue", alpha=0.3)

        plt.subplot(nb_sub, 1, 2, sharex=ax)
        for i in range(0, len(scales)):
            plt.plot(scales[i] + max(rec)*1.5 + i*75,
                     color="blue", alpha=0.5)
            #plt.plot(scales[len(scales)-i-1] + max(rec)*1.5 + i*75,



        if args.mode == 0:
            plt.subplot(nb_sub, 1, 3, sharex=ax)
            plt.contourf(np.real(full_scales), 100,
                         norm=colors.SymLogNorm(linthresh=0.2, linscale=0.05,
                                                vmin=np.min(full_scales), vmax=np.max(full_scales)),cmap="jet")
        plt.show()


###############################################################################
#  Envelopping
###############################################################################
def main():
    """Entry point for CWT analysis/synthesis tool

    This function is a wrapper to deal with arguments and logging.
    """
    global args

    try:
        parser = argparse.ArgumentParser(description="Tool for CWT analysis/synthesis of the F0")

        # Add options
        parser.add_argument("-B", "--binary-mode", action="store_true",
                            help="Activate binary mode, else files are assumed to be a csv for the f0/wavelet part")
        parser.add_argument("-c", "--configuration-file", default=None, help="configuration file")
        parser.add_argument("-M", "--mode", type=int, default=0,
                            help="script mode: 0=analysis, 1=synthesis")
        parser.add_argument("-m", "--mean_f0", type=float, default=100,
                            help="Mean f0 needed for synthesis (unsed for analysis modes)")
        parser.add_argument("-P", "--plot", action="store_true",
                            help="Plot the results")
        parser.add_argument("-v", "--verbosity", action="count", default=0,
                            help="increase output verbosity")

        # Add arguments
        parser.add_argument("input_file", help="Input signal or F0 file")
        parser.add_argument("output_file",
                            help="output directory for analysis or filename for synthesis. " +
                            "(Default: input_file directory [Analysis] or <input_file>.f0 [Synthesis])")

        # Parsing arguments
        args = parser.parse_args()

        # Verbose level => logging level
        log_level = args.verbosity
        if (args.verbosity >= len(LEVEL)):
            log_level = len(LEVEL) - 1
            logging.basicConfig(level=LEVEL[log_level])
            logging.warning("verbosity level is too high, I'm gonna assume you're taking the highest (%d)" % log_level)
        else:
            logging.basicConfig(level=LEVEL[log_level])

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

# cwt_analysis_synthesis.py ends here

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>
    - Sébastien Le Maguer <lemagues@helsinki.fi>

DESCRIPTION

usage: cwt_global_spectrum.py [-h] [-v] [-o OUTPUT]
                              [-P]
                              input_file


Tool for extracting global wavelet spectrum of speech envelope
introduced for second language fluency estimation in the following paper:

@inproceedings{suni2019characterizing,
  title={Characterizing second language fluency with global wavelet spectrum},
  author={Suni, Antti and Kallio, Heini and Benu{\v{s}}, {\v{S}}tefan and {\v{S}}imko, Juraj},
  booktitle={International Congress of Phonetic Sciences},
  pages={1947--1951},
  year={2019},
  organization={Australasian Speech Science and Technology Association Inc.}
}

positional arguments:
  input_file            Input signal or F0 file

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbosity       increase output verbosity
  -o OUTPUT, --output OUTPUT
                        output directory for analysis or filename for synthesis.
                        (Default: input_file directory [Analysis] or <input_file>.f0 [Synthesis])
  -P, --plot            Plot the results


You should be able to see peak around 4Hz, corresponding to syllable rate.
For longer speech files, lower frequency peaks related to phrasing should appear.
Synthetic test file with 8Hz, 4Hz and 1Hz components is included in sample directory.


LICENSE
    See https://github.com/asuni/wavelet_prosody_toolkit/blob/master/LICENSE.txt

"""

# Core Python
import sys
import os
import argparse

# Messaging/logging
import logging
from logging.config import dictConfig
try:
    import pythonjsonlogger
    JSON_LOGGER = True
except Exception:
    JSON_LOGGER = False


# Math/plot
import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt

# Libraries
from wavelet_prosody_toolkit.prosody_tools import cwt_utils as cwt_utils
from wavelet_prosody_toolkit.prosody_tools import misc as misc
from wavelet_prosody_toolkit.prosody_tools import energy_processing as energy_processing

###############################################################################
# global constants
###############################################################################
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]

###############################################################################
# Functions
###############################################################################
def configure_logger(args) -> logging.Logger:
    """Setup the global logging configurations and instanciate a specific logger for the current script

    Parameters
    ----------
    args : dict
        The arguments given to the script

    Returns
    --------
    the logger: logger.Logger
    """
    # create logger and formatter
    logger = logging.getLogger()

    # Verbose level => logging level
    log_level = args.verbosity
    if args.verbosity >= len(LEVEL):
        log_level = len(LEVEL) - 1
        # logging.warning("verbosity level is too high, I'm gonna assume you're taking the highest (%d)" % log_level)

    # Define the default logger configuration
    logging_config = dict(
        version=1,
        disable_existing_logger=True,
        formatters={
            "f": {
                "format": "[%(asctime)s] [%(levelname)s] — [%(name)s — %(funcName)s:%(lineno)d] %(message)s",
                "datefmt": "%d/%b/%Y: %H:%M:%S ",
            }
        },
        handlers={
            "h": {
                "class": "logging.StreamHandler",
                "formatter": "f",
                "level": LEVEL[log_level],
            }
        },
        root={"handlers": ["h"], "level": LEVEL[log_level]},
    )

    # Add file handler if file logging required
    if args.log_file is not None:
        cur_formatter_key = "f"
        if JSON_LOGGER:
            logging_config["formatters"]["j"] = {
                '()': 'pythonjsonlogger.json.JsonFormatter',
                'fmt': '%(asctime)s %(levelname)s %(filename)s %(lineno)d %(message)s',
                'rename_fields': {'asctime': 'time', 'levelname': 'level', 'lineno': 'line_number'}
            }
            cur_formatter_key = "j"

        logging_config["handlers"]["f"] = {
            "class": "logging.FileHandler",
            "formatter": cur_formatter_key,
            "level": LEVEL[log_level],
            "filename": args.log_file,
        }
        logging_config["root"]["handlers"] = ["h", "f"]

    # Setup logging configuration
    dictConfig(logging_config)

    # Retrieve and return the logger dedicated to the script
    logger = logging.getLogger(__name__)
    return logger


def define_argument_parser() -> argparse.ArgumentParser:
    """Defines the argument parser

    Returns
    --------
    The argument parser: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="")

    # Add logging options
    parser.add_argument("-l", "--log_file", default=None, help="Logger file")
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity",
    )

    # Add performative options
    parser.add_argument("-o", "--output_dir", default=None, type=str,
                        help="The output directory (if not defined, use the same directory than the wave file)")
    parser.add_argument("-P", "--plot", default=False, action="store_true",
                        help="Plot the results")

    # Add arguments
    parser.add_argument("wav_file", help="The input wave file")

    # Return parser
    return parser

###############################################################################
# Functions
###############################################################################
def calc_global_spectrum(wav_file, period=5, n_scales=60, plot=False):
    """
    """

    # Extract signal envelope, scale and normalize
    (fs, waveform) = misc.read_wav(wav_file)
    waveform = misc.resample(waveform, fs, 16000)
    energy = energy_processing.extract_energy(waveform, min_freq=30, method="hilbert")
    energy[energy<0] = 0
    energy = np.cbrt(energy+0.1)
    params = misc.normalize_std(energy)


    # perform continous wavelet transform on envelope with morlet wavelet

    # increase _period to get sharper spectrum
    matrix, scales, freq = cwt_utils.cwt_analysis(params, first_freq = 16, num_scales = n_scales, scale_distance  = 0.1,period=period, mother_name="Morlet",apply_coi=True)


    # power, arbitrary scaling to prevent underflow
    p_matrix = (abs(matrix)**2).astype('float32')*1000.0
    power_spec = np.nanmean(p_matrix,axis=1)

    if plot:
        f, wave_pics = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[5, 1]},  sharey=True)
        f.subplots_adjust(hspace=10)
        f.subplots_adjust(wspace=0)
        wave_pics[0].set_ylim(0, n_scales)
        wave_pics[0].set_xlabel("Time(m:s)")
        wave_pics[0].set_ylabel("Frequency(Hz)")
        wave_pics[1].set_xlabel("power")
        wave_pics[1].tick_params(labelright=True)

        fname = os.path.basename(wav_file)
        title = "CWT Morlet(p="+str(period)+") global spectrum, "+ fname
        wave_pics[0].contourf(p_matrix, 100)
        wave_pics[0].set_title(title, loc="center")
        wave_pics[0].plot(params*3, color="white",alpha=0.5)

        freq_labels =  [round(x,3)
                        if (np.isclose(x, round(x)) or
                            (x < 2 and np.isclose(x*100., round(x*100))) or
                            (x < 0.5 and np.isclose(x*10000., round(x*10000))))
                        else ""
                        for x in list(freq)]

        wave_pics[0].set_yticks(np.linspace(0, len(freq_labels)-1, len(freq_labels)))
        wave_pics[0].set_yticklabels(freq_labels)
        formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 200)))
        wave_pics[0].xaxis.set_major_formatter(formatter)
        wave_pics[1].grid(axis="y")
        wave_pics[1].plot(power_spec,np.linspace(0,len(power_spec), len(power_spec)),"-")
        plt.show()


    return (power_spec, freq)

###############################################################################
# Entry point
###############################################################################
def main():
    # Initialization of the argument parser and the logger
    arg_parser = define_argument_parser()
    args = arg_parser.parse_args()
    logger = configure_logger(args)


    # Compute the global spectrum
    (power_spec, freq) = calc_global_spectrum(wav_file=args.wav_file, plot=args.plot)

    # save spectrum and associated frequencies for further processing
    output_dir = os.path.dirname(args.wav_file)
    if args.output_dir is not None:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.join(output_dir, os.path.splitext(os.path.basename(args.wav_file))[0])
    np.savetxt(basename+".spec.txt", power_spec, fmt="%.5f", newline= " ")
    np.savetxt(basename+".freqs.txt", freq, fmt="%.5f", newline= " ")


###############################################################################
# Wrapping for directly calling the scripts
###############################################################################
if __name__ == "__main__":
    main()

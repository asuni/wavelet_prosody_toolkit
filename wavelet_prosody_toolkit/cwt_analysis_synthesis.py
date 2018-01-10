#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR

DESCRIPTION

LICENSE
"""

import sys
import os
import traceback
import argparse
import time
import logging

import warnings

import pylab

from wavelet_prosody_toolkit.prosody_tools import f0_processing, cwt_utils

import numpy as np

LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]

###############################################################################
# Functions
###############################################################################
def load_f0(input_file):
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
    if input_file.lower().endswith(".f0"):
        raw_f0 = np.loadtxt(input_file)
    elif input_file.lower().endswith(".wav"):
        logging.info("Extracting the F0 from the signal")
        raw_f0 = f0_processing.extract_f0(input_file)

    return raw_f0

###############################################################################
# Main function
###############################################################################
def run():
    """Main entry function
    """
    global args
    scales = None

    # Dealing with output
    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.dirname(args.input_file)
    basename = os.path.basename(args.input_file)
    output_file = os.path.join(output_dir, basename)

    # FIXME: be more specific!
    warnings.simplefilter("ignore", np.ComplexWarning) # Plotting can't deal with complex, but we don't care

    # Analysis
    if (args.mode % 2) == 0:

        raw_f0 = load_f0(args.input_file)
        logging.debug(raw_f0)

        logging.info("Processing f0")
        f0 = f0_processing.process(raw_f0)
        if args.plot:
            pylab.title("F0 preprocessing and interpolation")
            pylab.plot(f0, color="red", alpha=0.5, linewidth=3)
            pylab.plot(raw_f0,color="gray", alpha=0.5)


        logging.info("writing interpolated lf0\t" + output_file + ".interp")
        np.savetxt(output_file + ".interp", f0.astype('float'), fmt="%f", delimiter="\n")

        # Perform continuous wavelet transform of mean-substracted f0 with 12 scales, one octave apart
        scales, widths = cwt_utils.cwt_analysis(f0-np.mean(f0), num_scales=12, scale_distance=1.0, mother_name="mexican_hat", apply_coi=False)

        # SSW parameterization, adjacent scales combined (with extra scales to handle long utterances)
        scales = cwt_utils.combine_scales(scales, [(0,2),(2,4),(4,6),(6,8),(8,12)])
        for i in range(0,len(scales)):
            logging.debug("Mean scale[%d]: %s" % (i, str(np.mean(scales[i]))))

        logging.info("writing wavelet matrix \"%s.cwt\"" % output_file)
        np.savetxt(output_file + ".cwt", scales[:].T.astype('float'), fmt="%f", delimiter="\n")

        # for individual training of scales
        for i in range(0, 5):
            logging.info("writing scale \"%s.cwt.%d\"" % (output_file, i))
            np.savetxt(output_file+".cwt."+str(i+1), scales[i].astype('float'),fmt="%f", delimiter="\n")



    # then add deltas etc, train and generate
    # then synthesis by the following, voicing and mean value
    # have to come from other sources

    # Synthesis mode
    if args.mode >= 1 or args.plot:
        if scales is None:
            scales = np.loadtxt(args.input_file).reshape(-1,5).T
        if args.mode == 1:
            rec = cwt_utils.cwt_synthesis(scales, args.mean_f0)
        else:
            rec = cwt_utils.cwt_synthesis(scales, np.mean(f0))
        #rec = exp(cwt_utils.cwt_synthesis(scales)+mean(lf0))
        # rec[f0==0] = 0

    if args.mode >= 1:
        if args.mode == 1:
            if output_file is not None:
                output_file = args.input_file + "_rec.f0"
            else:
                output_file = args.output
        else:
            output_file += "_rec.f0"

        logging.info("Save reconstructed f0 in %s" % output_file)
        np.savetxt(output_file, rec.astype('float'), fmt="%f", delimiter="\n")


    if args.plot:
        pylab.figure()
        pylab.title("CWT decomposition to 5 scales and reconstructed signal")
        pylab.plot(rec, linewidth=5, color="blue", alpha=0.3)
        if (args.mode % 2) == 0:
            pylab.plot(f0, linewidth=1, color="red")
        for i in range(0,len(scales)):
            pylab.plot(scales[len(scales)-i-1]+max(rec)*1.5+i*75, color="blue", alpha=0.5, linewidth=2)

        pylab.show()

###############################################################################
#  Envelopping
###############################################################################
def main():
    global args

    try:
        parser = argparse.ArgumentParser(description="")

        # Add options
        parser.add_argument("-v", "--verbosity", action="count", default=0,
                            help="increase output verbosity")
        parser.add_argument("-M", "--mode", type=int, default=0,
                            help="script mode: 0=analysis, 1=synthesis, 2=analysis/synthesis")
        parser.add_argument("-m", "--mean_f0", type=float, default=100,
                            help="Mean f0 needed for synthesis (unsed for analysis modes)")
        parser.add_argument("-o", "--output",
                            help="output directory for analysis or filename for synthesis. (Default: input_file directory [Analysis] or <input_file>.f0 [Synthesis])")
        parser.add_argument("-P", "--plot", action="store_true",
                            help="Plot the results")

        # Add arguments
        parser.add_argument("input_file", help="Input signal or F0 file")

        # Parsing arguments
        args = parser.parse_args()

        # Verbose level => logging level
        log_level = args.verbosity
        if (args.verbosity > len(LEVEL)):
            logging.warning("verbosity level is too high, I'm gonna assume you're taking the highes ")
            log_level = len(LEVEL) - 1
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

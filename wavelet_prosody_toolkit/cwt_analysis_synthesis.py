from . import cwt_utils, f0_processing

from numpy import loadtxt,savetxt, mean, std, array,real,log, exp, sqrt

import os.path
import sys

# Logging
import logging
logger = logging.getLogger(__name__)

FIX_OUTLIERS = 1
INTERPOLATE = 1
CWT_ANALYSIS = 1
CWT_SYNTHESIS = 0

if __name__=="__main__":


    input_file = sys.argv[1]
    f0 = loadtxt(sys.argv[1])

    # operate on log-domain
    lf0 = array(f0)
    if mean(f0[f0>0]) > 20:

        lf0[f0>0]=log(f0[f0>0])
        lf0[f0<=0] = 0
    if FIX_OUTLIERS:
        lf0 = f0_processing.remove_outliers(lf0, trace=False)
    if INTERPOLATE:

        lf0 =f0_processing.interpolate(lf0,'true_envelope')
        logging.info("writing interpolated lf0\t%s.interp" % input_file)
        savetxt(input_file+".interp", lf0.astype('float'),fmt="%f", delimiter="\n")

    if CWT_ANALYSIS:
        # all scales
        scales = cwt_utils.cwt_analysis(lf0, num_scales=12, scale_distance=1.0)

        # SSW parameterization, adjacent scales combined (with extra scales to handle long utterances)
        scales = cwt_utils.combine_scales(scales,[(0,2),(2,4),(4,6),(6,8),(8,12)])
        logging.info("writing wavelet matrix\t%s.cwt" % input_file)
        savetxt(input_file+".cwt", scales[:].T.astype('float'), fmt="%f", delimiter="\n")

        # for individual training of scales
        for i in range(0, 5):
            logging.info("writing scale\t%s.cwt.%d" % (input_file, i))
            savetxt(input_file+".cwt."+str(i+1), scales[i].astype('float'),fmt="%f", delimiter="\n")



    # then add deltas etc, train and generate
    # then synthesis by the following, voicing and mean value
    # have to come from other sources


    if CWT_SYNTHESIS:
        cwt = loadtxt(input_file+".cwt").reshape(-1,5).T
        rec = exp(cwt_utils.cwt_synthesis(scales)+mean(lf0))
        rec[f0==0] = 0

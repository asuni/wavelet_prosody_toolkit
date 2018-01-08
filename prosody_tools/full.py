import cwt_utils


import sys
import soundfile as wav
import pylab

import scipy.signal
import numpy as np
(sig,fs) = wav.read(sys.argv[1])
sig = scipy.signal.decimate(sig, 2, zero_phase=True)
(cwt,scales) = cwt_utils.cwt_analysis(sig,mother_name='Morlet', period=7, num_scales=100, scale_distance=0.2)

cwt2, scales  =  cwt_utils.cwt_analysis(np.sqrt(sig*sig),mother_name='mexican_hat', period=2, num_scales=100, scale_distance=0.2, apply_coi=False)
cwt=abs(cwt)
cwt+=cwt2.real
import matplotlib.colors as colors
#pylab.contourf(np.flipud(cwt),100, cmap="jet")
print np.nanmin(cwt), np.nanmax(cwt)
pylab.contourf(np.flipud(cwt),100,  norm=colors.SymLogNorm(linthresh=0.002), cmap="jet")
#pylab.contourf(cwwt,100,  norm=colors.SymLogNorm(linthresh=0.03, linscale=0,vmin=0, vmax=0.003), cmap="jet")    
pylab.show()


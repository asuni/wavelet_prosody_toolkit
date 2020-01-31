#!/bin/bash

# Check default script
python3 wavelet_prosody_toolkit/cwt_analysis_synthesis.py -v samples/01l_fact_0001.wav 01l_fact_0001.cwt
diff 01l_fact_0001.cwt test/resources/01l_fact_0001.cwt
ret=$?
if [ $ret != 0 ]; then
    exit $ret
fi

# Check prosody labeller
python3 wavelet_prosody_toolkit/prosody_labeller.py -v -o test_libri -c wavelet_prosody_toolkit/configs/libritts.yaml samples/libritts
python3 test/diff_num.py  test_libri test/resources/libritts
#diff -r test_libri test/resources/libritts
ret=$?
if [ $ret != 0 ]; then
    exit $ret
fi

# Check global spectrum extractor
python3 wavelet_prosody_toolkit/cwt_global_spectrum.py -v -o test_spectrum samples/8hz_4hz_1hz.wav
diff -r test_spectrum/ test/resources/test_spectrum
ret=$?
if [ $ret != 0 ]; then
    exit $ret
fi

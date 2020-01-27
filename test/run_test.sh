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
diff -r test_libri test/resources/libritts
ret=$?
if [ $ret != 0 ]; then
    exit $ret
fi

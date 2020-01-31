# travis test for comparing prominence and boundary values across versions.
# allow for minor differences in values 
import sys, glob
import numpy as np
ref_files = sorted(glob.glob(sys.argv[1]+"/*.prom"))
test_files = sorted(glob.glob(sys.argv[2]+"/*.prom"))

for i in range(len(ref_files)):
    ref = (open(ref_files[i], "r")).readlines()
    test = (open(test_files[i], "r")).readlines()

    val_ref = []
    val_test = []
    # compare prominence and boundary values with some tolerance
    for l in ref:
        val_ref.append(float(l.strip().split("\t")[-1]))
        val_ref.append(float(l.strip().split("\t")[-2]))
    for l in test:
        val_test.append(float(l.strip().split("\t")[-1]))
        val_test.append(float(l.strip().split("\t")[-2]))


    assert np.allclose(np.array(val_ref), np.array(val_test), atol=0.3), \
        ref_files[i]+" and "+test_files[i]+ " differ too much!"

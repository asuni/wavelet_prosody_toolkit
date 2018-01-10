from numpy import array,concatenate, sqrt, pad, mean, std, real, nan, zeros, nanmean, nanstd

#from pycwt import cwt

import pycwt as cwt


def _unpad(matrix, num):
    unpadded = matrix[:,num:len(matrix[0])-num]
    return unpadded


def _padded_cwt(params, dt, dj, s0, J,mother, padding_len):
    #padded = concatenate([params,params,params])
    padded = pad(params, padding_len, mode='edge') #edge
    wavelet_matrix, scales, freqs, coi, fft, fftfreqs = cwt.cwt(padded, dt, dj, s0, J,mother)
    wavelet_matrix = _unpad(wavelet_matrix, padding_len)
    #wavelet_matrix = _unpad(wavelet_matrix, len(params))

    return (wavelet_matrix, scales, freqs, coi, fft, fftfreqs)
def _zero_outside_coi(wavelet_matrix,scales):
    for i in range(0,wavelet_matrix.shape[0]):
        coi =int((scales[i]))
        wavelet_matrix[i,0:coi] = 0.
        wavelet_matrix[i,-coi:] = 0.
    return wavelet_matrix

def _scale_for_reconstruction(wavelet_matrix,scales, dj, dt,mother="mexican_hat",period=3):
    scaled = array(wavelet_matrix)

    # mexican Hat
    c = dj / (3.541 * 0.867)


    if mother=="morlet":
        from numpy import pi

        cc = 1.83
        #periods 5 and 6 are correct, 3,4 approximate
        if period == 3:
            cc = 1.74
        if period == 4:
            cc = 1.1
        elif period==5:
            cc=0.9484
        elif period==6:
            cc == 0.7784


        c = dj / (cc * pi**(-0.25))
        #for i in range(0, len(scales)):
        #    scaled[i]*= 1./(i+1)  # c*sqrt(dt)/sqrt(scales[i])

    for i in range(0, len(scales)):
        scaled[i]*= c*sqrt(dt)/sqrt(scales[i])
        # substracting the mean should not be necessary?
        scaled[i]-=mean(scaled[i])
    return scaled


def combine_scales(wavelet_matrix, slices):
    combined_scales = []

    for i in range(0, len(slices)):
        combined_scales.append(sum(wavelet_matrix[slices[i][0]:slices[i][1]]))
    return array(combined_scales)

def cwt_analysis(params, mother_name="mexican_hat",num_scales=12, first_scale = None, scale_distance=1.0, apply_coi=True,period=5, frame_rate = 200):


    # setup wavelet transform
    dt = 1. /float(frame_rate)  # frame length
    dj = scale_distance  # distance between scales in octaves
    if first_scale == None:
        first_scale = dt # first scale, here frame length
    J =  num_scales #  number of scales

    mother = cwt.MexicanHat()

    if str.lower(mother_name) == "morlet":
        mother = cwt.Morlet(period)
    elif str.lower(mother_name) == "paul":
        mother = cwt.Paul(period)

    wavelet_matrix, scales, freqs, coi, fft, fftfreqs = _padded_cwt(params, dt, dj, first_scale, J,mother, 400)
    #wavelet_matrix, scales, freqs, coi, fft, fftfreqs = cwt.cwt(f0_mean_sub, dt, dj, s0, J,mother)

    #wavelet_matrix = abs(wavelet_matrix)
    wavelet_matrix = _scale_for_reconstruction((wavelet_matrix), scales, dj, dt,mother=mother_name,period=period)
    if apply_coi:
        wavelet_matrix = _zero_outside_coi(wavelet_matrix, scales/dt*0.5)


    return (wavelet_matrix,scales)


def cwt_synthesis(wavelet_matrix, mean = 0):
    return sum(wavelet_matrix[:])+mean

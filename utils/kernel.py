import numpy as np
import math
import pandas as pd
# importing brain template shape
from utils.template import pad_shape

def kernel_calc(fwhm, dims):
    #Conversion from fwhm to sigma based on affine
    s = ((fwhm / 2) / math.sqrt(8*math.log(2)) + np.finfo(float).eps)**2

    rs = math.ceil(3.5*math.sqrt(s)) # Half of required kernel length
    x = list(range(-rs, rs + 1))
    rk = np.exp(-0.5 * np.multiply(x,x) / s) / math.sqrt(2*math.pi*s)
    rk = rk / sum(rk)
    
    # Convolution of 1D Gaussians to create 3D Gaussian
    gkern3d = rk[:,None,None] * rk[None,:,None] * rk[None,None,:]
    
    # Padding to get matrix of desired size
    pad_size = int((dims - len(x)) / 2)
    gkern3d = np.pad(gkern3d, ((pad_size,pad_size),
                               (pad_size,pad_size),
                               (pad_size,pad_size)),'constant', constant_values=0)
    return gkern3d

def kernel_conv(peaks, kernel):
    data = np.zeros(pad_shape)
    for peak in peaks:
        x = (peak[0],peak[0]+31)
        y = (peak[1],peak[1]+31)
        z = (peak[2],peak[2]+31)
        data[x[0]:x[1], y[0]:y[1], z[0]:z[1]] = np.maximum(data[x[0]:x[1], y[0]:y[1], z[0]:z[1]], kernel)
    return data[15:data.shape[0]-15,15:data.shape[1]-15, 15:data.shape[2]-15]
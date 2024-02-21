import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import norm
from joblib import Parallel, delayed
from utils.template import shape, pad_shape, prior, affine, sample_space
from utils.kernel import kernel_conv


EPS =  np.finfo(float).eps


def compute_ma(peaks, kernels):
    ma = np.zeros((len(kernels), shape[0], shape[1], shape[2]))
    for i, kernel in enumerate(kernels):
        ma[i, :] = kernel_conv(peaks = peaks[i], 
                               kernel = kernel)
        
    return ma


def compute_hx(ma, bin_edges):
    hx = np.zeros((ma.shape[0], len(bin_edges)))
    for i in range(ma.shape[0]):
        data = ma[i, :]
        bin_idxs, counts = np.unique(np.digitize(data[prior], bin_edges),return_counts=True)
        hx[i,bin_idxs] = counts
    return hx


def compute_ale(ma):
    return 1-np.prod(1-ma, axis=0)


def compute_hx_conv(hx, bin_centers, step):    
    ale_hist = hx[0,:]
    for x in range(1,hx.shape[0]):
        v1 = ale_hist
        # save bins, which there are entries in the combined hist
        da1 = np.where(v1 > 0)[0]
        # normalize combined hist to sum to 1
        v1 = ale_hist/np.sum(v1)
        
        v2 = hx[x,:]
        # save bins, which there are entries in the study hist
        da2 = np.where(v2 > 0)[0]
        # normalize study hist to sum to 1
        v2 = hx[x,:]/np.sum(v2)
        ale_hist = np.zeros((len(bin_centers),))
        #iterate over bins, which contain values
        for i in range(len(da2)):
            p = v2[da2[i]]*v1[da1]
            score = 1-(1-bin_centers[da2[i]])*(1-bin_centers[da1])
            ale_bin = np.round(score*step).astype(int)
            ale_hist[ale_bin] = np.add(ale_hist[ale_bin], p)
    last_used = np.where(ale_hist>0)[0][-1]
    hx_conv = np.flip(np.cumsum(np.flip(ale_hist[:last_used+1])))
    
    return hx_conv


def compute_z(ale, hx_conv, step):    
    # computing the corresponding histogram bin for each ale value
    ale_step = np.round(ale*step).astype(int)
    # replacing histogram bin number with corresponding histogram value (= p-value)
    p = np.array([hx_conv[i] for i in ale_step])
    p[p < EPS] = EPS
    # calculate z-values by plugging 1-p into a probability density function
    z = norm.ppf(1-p)
    
    return z


def compute_cluster(z, thresh, cut_cluster=None):    
    # disregard all voxels that feature a z-value of lower than some threshold (approx. 3 standard deviations aways from the mean)
    # this serves as a preliminary thresholding
    sig_arr = np.zeros(shape)
    sig_arr[z > norm.ppf(1-thresh)] = 1
    # find clusters of significant z-values
    labels, cluster_count = ndimage.label(sig_arr)
    # save number of voxels in biggest cluster
    try:
        max_clust = np.max(np.bincount(labels[labels>0]))
    except ValueError:
        max_clust = 0

    if cut_cluster is not None:
        # check significance of cluster against the 95th percentile of the null distribution cluster size
        sig_clust = np.where(np.bincount(labels[labels > 0]) > cut_cluster)[0]
        # z-value array that only features values for voxels that belong to significant clusters
        z = z*np.isin(labels, sig_clust)
        return z, max_clust
    
    return max_clust

    
def compute_null_cutoffs(s0, sample_space, num_peaks, kernels, step=10000, thresh=0.001, target_n=None,
                          hx_conv=None, bin_edges=None, bin_centers=None, tfce_enabled=True):
    if target_n:
        s0 = np.random.permutation(s0)
        s0 = s0[:target_n]
    # compute ALE values based on random peak locations sampled from a give sample_space
    # sample space could be all grey matter or only foci reported in brainmap
    null_ma, null_ale = compute_null_ale(s0, sample_space, num_peaks, kernels)
    # Peak ALE threshold
    null_max_ale = np.max(null_ale)
    if hx_conv is None:
        null_hx = compute_hx(null_ma, bin_edges)
        hx_conv = compute_hx_conv(null_hx, bin_centers, step)
    null_z = compute_z(null_ale, hx_conv, step)
    # Cluster level threshold
    null_max_cluster = compute_cluster(null_z, thresh, sample_space)
    if tfce_enabled:
        tfce = compute_tfce(null_z)
        # TFCE threshold
        null_max_tfce = np.max(tfce)
        return null_max_ale, null_max_cluster, null_max_tfce
        
    return null_max_ale, null_max_cluster, None

def compute_null_cluster(s0, sample_space, num_peaks, kernels, step, cluster_thresh, bin_centers, bin_edges, target_n):
    srandom = np.random.permutation(s0)[:target_n]

    null_peaks = np.array([sample_space[:,np.random.randint(0,sample_space.shape[1], num_peaks[i])].T for i in srandom], dtype=object)
    null_ma = np.zeros((len(srandom), shape[0], shape[1], shape[2]))
    for i, kernel in enumerate(kernels.loc[srandom]):
        null_ma[i, :] = kernel_conv(peaks = null_peaks[i], 
                                    kernel = kernel)

    null_ale = compute_ale(null_ma)
    null_hx = compute_hx(null_ma, bin_edges)
    hx_conv = compute_hx_conv(null_hx, bin_centers, step)
    null_z = compute_z(null_ale, hx_conv, step)
    null_max_cluster = compute_cluster(null_z, cluster_thresh)
    return null_max_cluster


def create_samples(N, target_n, num_samples):
    possible_combinations = math.factorial(N) / (math.factorial(target_n)*math.factorial(N-target_n))
    if possible_combinations < num_samples:
        num_samples = possible_combinations
                                    
    subsamples = set()
    while len(subsamples)<num_samples:
        perm = np.random.permutation(range(N))
        subsamples.add(tuple(sorted(perm[:target_n]))) # `tuple` because `list`s are not hashable. right Beni?

    return list(subsamples)

def compute_sub_ale(subsample, ma, hx, bin_centers, cut_cluster, step=10000, cluster_thresh=0.001):
    hx_conv = compute_hx_conv(hx[subsample], bin_centers, step)
    ale = compute_ale(ma[subsample])
    z = compute_z(ale, hx_conv, step)
    z, max_cluster = compute_cluster(z, cluster_thresh, cut_cluster=cut_cluster)
    z[z > 0] = 1
    return z

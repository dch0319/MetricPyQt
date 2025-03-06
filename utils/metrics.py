import multiprocessing
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import EuclideanTransform, warp

from utils.imtools import rgb2gray, torch2np


# def ssim(img1, img2, cs_map=False):
#     if isinstance(img1, torch.Tensor):
#         img1 = img1.squeeze()
#         img2 = img2.squeeze()
#         img1 = img1.cpu().numpy()
#         img2 = img2.cpu().numpy()
#     if np.max(img1) < 2:
#         img1 = img1 * 255
#         img2 = img2 * 255
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     size = 11
#     sigma = 1.5
#     window = fspecial_gauss(size, sigma)
#     K1 = 0.01
#     K2 = 0.03
#     L = 255  # bitdepth of image
#     C1 = (K1 * L) ** 2
#     C2 = (K2 * L) ** 2
#     mu1 = signal.fftconvolve(window, img1, mode='valid')
#     mu2 = signal.fftconvolve(window, img2, mode='valid')
#     mu1_sq = mu1 * mu1
#     mu2_sq = mu2 * mu2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
#     sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
#     sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
#     if cs_map:
#         return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                              (sigma1_sq + sigma2_sq + C2)),
#                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
#     else:
#         ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#         return ssim.mean()


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


### Parallel Computing of PSNR by Best Matching with Rotation
def aver_bmp_psnr_ssim_rot_par(img1, img2, num_cores=None, bd_cut=15, maxshift=10, shift_inter=1, angle_inter=0.1,
                               maxangle=0.5, ssim_compute=True, show_aligned=False, verbose=False):
    ''' Parallel computing is applied for images'''
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    im_len = len(img1)
    for i in range(im_len):
        img1[i] = np.squeeze(torch2np(img1[i]))
        img2[i] = np.squeeze(torch2np(img2[i]))

        img1[i][img1[i] < 0] = 0
        img2[i][img2[i] < 0] = 0
        img1[i][img1[i] > 1] = 1
        img2[i][img2[i] > 1] = 1

        img1[i] = np.around(img1[i] * 255).astype(int) / 255
        img2[i] = np.around(img2[i] * 255).astype(int) / 255

    if num_cores == 0:
        Results = comp_upto_shif_rot_algn_color(img1[0], img2[0], cut=bd_cut, maxshift=maxshift, maxangle=maxangle,
                                                ssim_compute=ssim_compute, shift_inter=shift_inter,
                                                angle_inter=angle_inter)
    else:
        Results = Parallel(n_jobs=num_cores)(
            delayed(comp_upto_shif_rot_algn_color)(
                img1[i].copy(),  # 显式创建可写副本
                img2[i].copy(),  # 显式创建可写副本
                cut=bd_cut,
                maxshift=maxshift,
                maxangle=maxangle,
                ssim_compute=ssim_compute,
                shift_inter=shift_inter,
                angle_inter=angle_inter
            ) for i in range(im_len)
        )

    PSNR = np.zeros((im_len, 1))
    for ii in range(im_len):
        try:
            PSNR[ii] = Results[ii][0]
        except:
            PSNR[ii] = Results[ii]

    output = {}
    PSNR_mean = np.mean(PSNR)
    output['PSNR_mean'] = PSNR_mean
    if ssim_compute:
        SSIM = np.zeros((im_len, 1))
        for ii in range(im_len):
            SSIM[ii] = Results[ii][1]
        SSIM_mean = np.mean(SSIM)
        output['SSIM_mean'] = SSIM_mean
    if show_aligned:
        I1_matched = [None] * im_len
        I2_matched = [None] * im_len
        for ii in range(im_len):
            I1_matched[ii] = Results[ii][2]
            I2_matched[ii] = Results[ii][3]
        output['I1_matched'] = I1_matched
        output['I2_matched'] = I2_matched
    if verbose:
        output['PSNR'] = PSNR
        if ssim_compute:
            output['SSIM'] = SSIM
    return output


def comp_upto_shif_rot_algn_color(I1, I2, cut=15, maxshift=10, maxangle=0.5, shift_inter=1, angle_inter=0.1,
                                  ssim_compute=True):
    '''
    Compute the PSNR and SSIM for color image aligned by best matching principle using Euclidean transformation
        I1: Deblurred results
        I2: Sharp image
        cut: The boundary cut
    '''
    I1 = np.array(I1, copy=True).astype(np.float32)
    I2 = np.array(I2, copy=True).astype(np.float32)
    I1.setflags(write=True)
    I2.setflags(write=True)

    I1_gray = rgb2gray(I1)
    I2_gray = rgb2gray(I2)
    # Search the best matching principle with relatively small search scope
    x_shift = np.arange(-maxshift, maxshift + shift_inter, shift_inter)
    y_shift = np.arange(-maxshift, maxshift + shift_inter, shift_inter)
    r_shift = np.arange(-maxangle, maxangle + angle_inter, angle_inter)
    r_shift = r_shift / 180 * np.pi

    I2_gray_cut = I2_gray[cut:-cut, cut:-cut]
    ssdes = []
    trans = []
    for (x, y, r) in product(x_shift, y_shift, r_shift):
        model = EuclideanTransform(translation=[x, y], rotation=r)
        I1_gray_warped = warp(I1_gray, model.inverse)
        I1_gray_warped = I1_gray_warped[cut:-cut, cut:-cut]
        ssdes.append(np.sum((I1_gray_warped - I2_gray_cut) ** 2))
        trans.append((x, y, r))

    idx = np.argmin(ssdes)
    bx, by, br = trans[idx]
    model = EuclideanTransform(translation=[bx, by], rotation=br)
    I1_warped = warp(I1, model.inverse)
    I1_warped_cut = I1_warped[cut:-cut, cut:-cut, :]
    psnr_metric = psnr(I1_warped_cut, I2[cut:-cut, cut:-cut, :])
    ssim_metric = None

    # ssim is also computed in grayscale.
    if ssim_compute:
        ssim_metric = ssim(rgb2gray(I1_warped_cut), I2_gray_cut, data_range=1)
    return psnr_metric, ssim_metric, I1_warped_cut, I2[cut:-cut, cut:-cut, :].copy()


def comp_upto_shif_rot_algn_color_multiprocess(I1, I2, cut=15, maxshift=10, maxangle=0.5, shift_inter=1,
                                               angle_inter=0.1,
                                               ssim_compute=True):
    '''
    Compute the PSNR and SSIM for color image aligned by best matching principle using Euclidean transformation
        I1: Deblurred results
        I2: Sharp image
        cut: The boundary cut
    '''

    I1 = np.array(I1, copy=True).astype(np.float32)
    I2 = np.array(I2, copy=True).astype(np.float32)
    I1.setflags(write=True)
    I2.setflags(write=True)

    I1_gray = rgb2gray(I1)
    I2_gray = rgb2gray(I2)

    I2_gray_cut = I2_gray[cut:-cut, cut:-cut]

    x_shift = np.arange(-maxshift, maxshift + shift_inter, shift_inter)
    y_shift = np.arange(-maxshift, maxshift + shift_inter, shift_inter)
    r_shift = np.arange(-maxangle, maxangle + angle_inter, angle_inter)
    r_shift = r_shift / 180 * np.pi

    params = list(product(x_shift, y_shift, r_shift))

    def compute_ssd(param):
        x, y, r = param
        model = EuclideanTransform(translation=[x, y], rotation=r)
        I1_gray_warped = warp(I1_gray, model.inverse)
        I1_gray_warped = I1_gray_warped[cut:-cut, cut:-cut]
        ssd_val = np.sum((I1_gray_warped - I2_gray_cut) ** 2)
        return (ssd_val, (x, y, r))

    # 使用所有可用的CPU核心并行计算
    results = Parallel(n_jobs=-1, backend='threading', verbose=0)(
        delayed(compute_ssd)(param) for param in params
    )

    ssdes, trans = zip(*results) if results else ([], [])
    if not ssdes:
        raise ValueError("No transformations were computed.")

    idx = np.argmin(ssdes)
    bx, by, br = trans[idx]
    model = EuclideanTransform(translation=[bx, by], rotation=br)
    I1_warped = warp(I1, model.inverse)
    I1_warped_cut = I1_warped[cut:-cut, cut:-cut, :]
    psnr_metric = psnr(I1_warped_cut, I2[cut:-cut, cut:-cut, :])
    ssim_metric = None

    if ssim_compute:
        ssim_metric = ssim(rgb2gray(I1_warped_cut), I2_gray_cut, data_range=1)

    return psnr_metric, ssim_metric, I1_warped_cut, I2[cut:-cut, cut:-cut, :].copy()

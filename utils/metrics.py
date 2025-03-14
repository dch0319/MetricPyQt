import multiprocessing
from itertools import product
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import map_coordinates
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.transform import EuclideanTransform, warp
from utils.imtools import rgb2gray, torch2np


def comp_upto_shift(I1, I2, maxshift=10):
    """
    Computes PSNR, SSIM, and the shifted version of I1 that best aligns with I2.

    Parameters:
        I1 (numpy.ndarray): First input image.
        I2 (numpy.ndarray): Second input image to compare.
        maxshift (int): Maximum allowed shift in pixels.

    Returns:
        psnrs (float): PSNR value between best-shifted I1 and I2.
        ssims (float): SSIM value between best-shifted I1 and I2.
        tI1 (numpy.ndarray): The best-shifted version of I1.
    """
    # Step 1: Crop images to avoid boundary issues
    I1 = np.array(I1, copy=True).astype(np.float32)
    I2 = np.array(I2, copy=True).astype(np.float32)
    I1.setflags(write=True)
    I2.setflags(write=True)

    I1 = rgb2gray(I1)
    I2 = rgb2gray(I2)
    I2 = I2[16:-15, 16:-15]
    I1 = I1[16 - maxshift:-15 + maxshift, 16 - maxshift:-15 + maxshift]
    N1, N2 = I2.shape
    # Step 2: Generate coordinate grids
    shifts = np.arange(-maxshift, maxshift + 0.25, 0.25)
    gx, gy = np.meshgrid(np.arange(1 - maxshift, N2 + maxshift + 1),
                         np.arange(1 - maxshift, N1 + maxshift + 1))
    gx0, gy0 = np.meshgrid(np.arange(1, N2 + 1), np.arange(1, N1 + 1))

    # Step 3: Initialize variables for SSD computation
    ssdem = np.full((len(shifts), len(shifts)), np.inf)

    # Step 4: Evaluate SSD for all shift combinations
    for i, shift_x in enumerate(shifts):
        for j, shift_y in enumerate(shifts):
            gxn = gx0 + shift_x
            gyn = gy0 + shift_y
            tI1 = map_coordinates(I1, [gyn.ravel(), gxn.ravel()], order=1, mode='constant').reshape(N1, N2)
            ssdem[i, j] = np.sum((tI1 - I2) ** 2)

    # Step 5: Find the best shift
    min_ssd_idx = np.unravel_index(np.argmin(ssdem), ssdem.shape)
    best_shift_x, best_shift_y = shifts[min_ssd_idx[0]], shifts[min_ssd_idx[1]]

    # Step 6: Recompute the best-shifted image
    gxn = gx0 + best_shift_x
    gyn = gy0 + best_shift_y
    tI1 = map_coordinates(I1, [gyn.ravel(), gxn.ravel()], order=1, mode='constant').reshape(N1, N2)

    # Step 7: Compute PSNR and SSIM
    psnr = peak_signal_noise_ratio(I2 * 255, tI1 * 255, data_range=255)
    ssim = structural_similarity(I2 * 255, tI1 * 255, data_range=255)

    return psnr, ssim


def comp_upto_shift_multiprocess(I1, I2, cut=15, maxshift=10, shift_inter=0.25):
    """
    Computes PSNR, SSIM, and the shifted version of I1 that best aligns with I2.
    Multi-core accelerated version with joblib.

    Parameters:
        I1 (numpy.ndarray): First input image.
        I2 (numpy.ndarray): Second input image to compare.
        maxshift (int): Maximum allowed shift in pixels.

    Returns:
        psnr (float): PSNR value between best-shifted I1 and I2.
        ssim (float): SSIM value between best-shifted I1 and I2.
        tI1 (numpy.ndarray): The best-shifted version of I1.
    """
    # Step 1: Crop images with explicit copy
    I1 = rgb2gray(I1)
    I2 = rgb2gray(I2)
    I2 = np.array(I2[cut:-cut, cut:-cut], copy=True).astype(np.float32)
    I1 = np.array(I1[cut - maxshift:-cut + maxshift, cut - maxshift:-cut + maxshift], copy=True).astype(np.float32)
    I1.setflags(write=True)
    I2.setflags(write=True)

    N1, N2 = I2.shape
    shifts = np.arange(-maxshift, maxshift + shift_inter, shift_inter)

    # Precompute coordinate grids
    gy0, gx0 = np.mgrid[1:N1 + 1, 1:N2 + 1]

    # Generate all shift combinations
    shift_combinations = list(product(shifts, shifts))

    # Step 2: Parallel SSD computation
    def compute_shift_ssd(shift_pair):
        shift_x, shift_y = shift_pair
        gxn = gx0 + shift_x
        gyn = gy0 + shift_y

        # Create local writable copies
        I1_local = np.array(I1, copy=True)
        I1_local.setflags(write=True)

        tI1 = map_coordinates(I1_local, [gyn.ravel(), gxn.ravel()], order=1, mode='constant')
        tI1 = tI1.reshape(N1, N2)
        ssd = np.sum((tI1 - I2) ** 2)
        return (ssd, (shift_x, shift_y))

    # Use threading backend to avoid read-only issues
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(compute_shift_ssd)(pair) for pair in shift_combinations
    )

    # Step 3: Find optimal shift
    ssd_values, shift_pairs = zip(*results)
    ssd_matrix = np.array(ssd_values).reshape(len(shifts), len(shifts))
    min_idx = np.unravel_index(np.argmin(ssd_matrix), ssd_matrix.shape)
    best_shift_x, best_shift_y = shifts[min_idx[0]], shifts[min_idx[1]]

    # Step 4: Apply best shift
    gxn = gx0 + best_shift_x
    gyn = gy0 + best_shift_y
    tI1 = map_coordinates(I1, [gyn.ravel(), gxn.ravel()], order=1, mode='constant').reshape(N1, N2)

    # Step 5: Compute metrics
    psnr_val = peak_signal_noise_ratio(I2 * 255, tI1 * 255, data_range=255)
    ssim_val = structural_similarity(I2 * 255, tI1 * 255, data_range=255)

    return psnr_val, ssim_val, tI1


def comp_upto_shift_rgb(I1, I2, maxshift=10):
    """
    Computes PSNR, SSIM, and the shifted version of RGB image I1 that best aligns with RGB image I2.

    Parameters:
        I1 (numpy.ndarray): First input RGB image (shape: H x W x 3).
        I2 (numpy.ndarray): Second input RGB image to compare (shape: H x W x 3).
        maxshift (int): Maximum allowed shift in pixels.

    Returns:
        psnrs (float): PSNR value between best-shifted I1 and I2 (calculated for all channels).
        ssims (float): SSIM value between best-shifted I1 and I2 (calculated for all channels).
        tI1 (numpy.ndarray): The best-shifted version of I1 (shape: H x W x 3).
    """
    assert I1.shape == I2.shape, "Input images must have the same dimensions"
    assert I1.shape[-1] == 3, "Input images must be RGB (3 channels)"
    from scipy.ndimage import map_coordinates
    H, W, C = I1.shape

    # Crop images to avoid boundary issues
    I2 = I2[16:-15, 16:-15, :]
    I1 = I1[16 - maxshift:-15 + maxshift, 16 - maxshift:-15 + maxshift, :]
    N1, N2, _ = I2.shape

    # Generate coordinate grids
    shifts = np.arange(-maxshift, maxshift + 0.25, 0.25)
    gx0, gy0 = np.meshgrid(np.arange(1, N2 + 1), np.arange(1, N1 + 1))

    # Initialize variables for SSD computation
    ssdem = np.full((len(shifts), len(shifts)), np.inf)

    # Evaluate SSD for all shift combinations
    for i, shift_x in enumerate(shifts):
        for j, shift_y in enumerate(shifts):
            gxn = gx0 + shift_x
            gyn = gy0 + shift_y
            # Interpolate for all channels at once
            tI1 = np.stack([
                map_coordinates(I1[:, :, c], [gyn.ravel(), gxn.ravel()], order=1, mode='constant').reshape(N1, N2)
                for c in range(C)
            ], axis=-1)
            # Compute SSD across all channels
            ssdem[i, j] = np.sum((tI1 - I2) ** 2)

    # Find the best shift
    min_ssd_idx = np.unravel_index(np.argmin(ssdem), ssdem.shape)
    best_shift_x, best_shift_y = shifts[min_ssd_idx[0]], shifts[min_ssd_idx[1]]

    # Recompute the best-shifted image
    gxn = gx0 + best_shift_x
    gyn = gy0 + best_shift_y
    tI1 = np.stack([
        map_coordinates(I1[:, :, c], [gyn.ravel(), gxn.ravel()], order=1, mode='constant').reshape(N1, N2)
        for c in range(C)
    ], axis=-1)

    # Compute PSNR and SSIM for the entire RGB image
    psnr = peak_signal_noise_ratio(I2 * 255, tI1 * 255, data_range=255)
    ssim = structural_similarity(I2 * 255, tI1 * 255, multichannel=True, data_range=255, channel_axis=2)

    return psnr, ssim


##### for nonuniform blur #####
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


##### for nonuniform blur #####
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
    psnr_metric = peak_signal_noise_ratio(I1_warped_cut, I2[cut:-cut, cut:-cut, :])
    ssim_metric = None

    # ssim is also computed in grayscale.
    if ssim_compute:
        ssim_metric = structural_similarity(rgb2gray(I1_warped_cut), I2_gray_cut, data_range=1)
    return psnr_metric, ssim_metric, I1_warped_cut, I2[cut:-cut, cut:-cut, :].copy()


##### for nonuniform blur #####
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
    psnr_metric = peak_signal_noise_ratio(I1_warped_cut, I2[cut:-cut, cut:-cut, :])
    ssim_metric = None

    if ssim_compute:
        ssim_metric = structural_similarity(rgb2gray(I1_warped_cut), I2_gray_cut, data_range=1)

    return psnr_metric, ssim_metric, I1_warped_cut, I2[cut:-cut, cut:-cut, :].copy()

def comp_upto_shif_algn_color_multiprocess(I1, I2, cut=15, maxshift=10, shift_inter=1.0, ssim_compute=True):
    '''
    Compute the PSNR and SSIM for color image aligned by best matching principle using translation only
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

    # 仅生成平移参数组合
    params = list(product(x_shift, y_shift))

    def compute_ssd(param):
        x, y = param  # 移除了旋转参数
        model = EuclideanTransform(translation=[x, y])  # 仅应用平移
        I1_gray_warped = warp(I1_gray, model.inverse)
        I1_gray_warped = I1_gray_warped[cut:-cut, cut:-cut]
        ssd_val = np.sum((I1_gray_warped - I2_gray_cut) ** 2)
        return (ssd_val, (x, y))

    # 并行计算
    results = Parallel(n_jobs=-1, backend='threading', verbose=0)(
        delayed(compute_ssd)(param) for param in params
    )

    ssdes, trans = zip(*results) if results else ([], [])
    if not ssdes:
        raise ValueError("No transformations were computed.")

    idx = np.argmin(ssdes)
    bx, by = trans[idx]  # 仅获取平移参数
    model = EuclideanTransform(translation=[bx, by])
    I1_warped = warp(I1, model.inverse)
    I1_warped_cut = I1_warped[cut:-cut, cut:-cut, :]
    psnr_metric = peak_signal_noise_ratio(I1_warped_cut, I2[cut:-cut, cut:-cut, :])
    ssim_metric = None

    if ssim_compute:
        ssim_metric = structural_similarity(rgb2gray(I1_warped_cut), I2_gray_cut, data_range=1)

    return psnr_metric, ssim_metric, I1_warped_cut, I2[cut:-cut, cut:-cut, :].copy()
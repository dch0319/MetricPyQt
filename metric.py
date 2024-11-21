import numpy as np
from scipy.ndimage import map_coordinates
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def comp_upto_shift(I1, I2, maxshift=5):
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
    psnrs = psnr(I2 * 255, tI1 * 255, data_range=255)
    ssims = ssim(I2 * 255, tI1 * 255, data_range=255)

    return psnrs, ssims, tI1

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


def comp_upto_shift_rgb(I1, I2, maxshift=5):
    """
    Computes PSNR, SSIM, and the shifted version of RGB image I1 that best aligns with RGB image I2.

    Parameters:
        I1 (numpy.ndarray): First input RGB image (shape: H x W x 3).
        I2 (numpy.ndarray): Second input RGB image to compare (shape: H x W x 3).
        maxshift (int): Maximum allowed shift in pixels.

    Returns:
        psnrs (list of float): PSNR values for R, G, B channels.
        ssims (list of float): SSIM values for R, G, B channels.
        tI1 (numpy.ndarray): The best-shifted version of I1 (shape: H x W x 3).
    """
    assert I1.shape == I2.shape, "Input images must have the same dimensions"
    assert I1.shape[-1] == 3, "Input images must be RGB (3 channels)"

    H, W, _ = I1.shape
    psnrs = []
    ssims = []
    tI1 = np.zeros_like(I1)

    # Process each channel independently
    for channel in range(3):
        # Extract individual channel
        I1_channel = I1[:, :, channel]
        I2_channel = I2[:, :, channel]

        # Crop images to avoid boundary issues
        I2_cropped = I2_channel[16:-15, 16:-15]
        I1_cropped = I1_channel[16 - maxshift:-15 + maxshift, 16 - maxshift:-15 + maxshift]
        N1, N2 = I2_cropped.shape

        # Generate coordinate grids
        shifts = np.arange(-maxshift, maxshift + 0.25, 0.25)
        gx, gy = np.meshgrid(np.arange(1 - maxshift, N2 + maxshift + 1),
                             np.arange(1 - maxshift, N1 + maxshift + 1))
        gx0, gy0 = np.meshgrid(np.arange(1, N2 + 1), np.arange(1, N1 + 1))

        # Initialize variables for SSD computation
        ssdem = np.full((len(shifts), len(shifts)), np.inf)

        # Evaluate SSD for all shift combinations
        for i, shift_x in enumerate(shifts):
            for j, shift_y in enumerate(shifts):
                gxn = gx0 + shift_x
                gyn = gy0 + shift_y
                tI1_channel = map_coordinates(I1_cropped, [gyn.ravel(), gxn.ravel()], order=1, mode='constant').reshape(
                    N1, N2)
                ssdem[i, j] = np.sum((tI1_channel - I2_cropped) ** 2)

        # Find the best shift
        min_ssd_idx = np.unravel_index(np.argmin(ssdem), ssdem.shape)
        best_shift_x, best_shift_y = shifts[min_ssd_idx[0]], shifts[min_ssd_idx[1]]

        # Recompute the best-shifted image
        gxn = gx0 + best_shift_x
        gyn = gy0 + best_shift_y
        tI1_channel = map_coordinates(I1_cropped, [gyn.ravel(), gxn.ravel()], order=1, mode='constant').reshape(N1, N2)

        # Store the aligned channel
        tI1[16:-15, 16:-15, channel] = tI1_channel

        # Compute PSNR and SSIM
        psnrs.append(psnr(I2_cropped * 255, tI1_channel * 255, data_range=255))
        ssims.append(ssim(I2_cropped * 255, tI1_channel * 255, data_range=255))

        psnr_value = np.mean(psnrs)
        ssim_value = np.mean(ssims)

    return psnr_value, ssim_value, tI1


def comp_upto_shift_rgb2(I1, I2, maxshift):
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

    H, W, C = I1.shape

    # Crop images to avoid boundary issues
    I2 = I2[16:-15, 16:-15, :]
    I1 = I1[16 - maxshift:-15 + maxshift, 16 - maxshift:-15 + maxshift, :]
    N1, N2, _ = I2.shape

    # Generate coordinate grids
    shifts = np.arange(-maxshift, maxshift + 0.25, 0.25)
    gx, gy = np.meshgrid(np.arange(1 - maxshift, N2 + maxshift + 1),
                         np.arange(1 - maxshift, N1 + maxshift + 1))
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
    psnrs = psnr(I2 * 255, tI1 * 255, data_range=255)
    ssims = ssim(I2 * 255, tI1 * 255, multichannel=True, data_range=255, channel_axis=2)

    return psnrs, ssims, tI1

# blurry_image_path = r"E:\CV\Code\diffusion-deconv\对比试验\afhq\kernel=31\flickr_dog_000454_dog\vdip-deconv-extreme\x.png"
# gt_image_path = r"E:\CV\Code\diffusion-deconv\对比试验\afhq\kernel=31\flickr_dog_000454_dog\gt\flickr_dog_000454_dog.png"
# # Load images
# blurry_image = io.imread(blurry_image_path)[:, :, :3]  # RGBA -> RGB
# gt_image = io.imread(gt_image_path)[:, :, :3]
# blurry_image = rgb2gray(img_as_float32(blurry_image))
# gt_image = rgb2gray(img_as_float32(gt_image))
# psnr_value, ssim_value, _ = comp_upto_shift(gt_image, blurry_image, maxshift=5)
# print(f'PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.2f}')
#
# blurry_image = io.imread(blurry_image_path)[:, :, :3]  # RGBA -> RGB
# gt_image = io.imread(gt_image_path)[:, :, :3]
# blurry_image = (img_as_float32(blurry_image))
# gt_image = (img_as_float32(gt_image))
# psnr_value, ssim_value, _ = comp_upto_shift_rgb(gt_image, blurry_image, maxshift=5)
# print(f'PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.2f}')
# psnr_value, ssim_value, _ = comp_upto_shift_rgb2(gt_image, blurry_image, maxshift=5)
# print(f'PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.2f}')

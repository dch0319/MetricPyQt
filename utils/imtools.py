''' Image Tools '''
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import filters
from scipy.signal import fftconvolve


def torch2np(x_tensor):
    if isinstance(x_tensor, np.ndarray):
        return x_tensor
    elif not x_tensor.is_cuda:
        x = x_tensor.numpy()
        return x
    else:
        x = x_tensor.detach().cpu().numpy()
        return x


def imshow(x_in, str, dir='tmp/'):
    x = torch2np(x_in)
    x = np.squeeze(x)
    if len(x.shape) == 2:
        x[x > 1] = 1
        x[x < 0] = 0
        x_int = np.uint8(np.around(x * 255))
        Image.fromarray(x_int, 'L').save(dir + str + '.png')
    elif len(x.shape) == 3:
        if x.shape[0] == 3:
            x = x.transpose(1, 2, 0)
        x[x > 1] = 1
        x[x < 0] = 0
        x_int = np.uint8(np.around(x * 255))
        Image.fromarray(x_int, 'RGB').save(dir + str + '.png')


import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def imshown(x_in):
    x = torch2np(x_in)
    x = np.squeeze(x)
    if x.shape[0] == 3:
        x = np.stack((x[0,], x[1,], x[2,]), axis=2)
        plt.imshow(x, vmin=0, vmax=1)
    else:
        plt.imshow(x, vmin=0, vmax=1, cmap='gray')
    plt.show()


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def for_fft(ker, shape):
    ker_mat = np.zeros(shape, dtype=np.float32)
    ker_shape = np.asarray(np.shape(ker))
    circ = np.ndarray.astype(-np.floor((ker_shape) / 2), dtype=np.int)
    ker_mat[:ker_shape[0], :ker_shape[1]] = ker
    ker_mat = np.roll(ker_mat, circ, axis=(0, 1))
    return ker_mat


def fspecial(type, *args):
    dtype = np.float32
    if type == 'average':
        siz = (args[0], args[0])
        h = np.ones(siz) / np.prod(siz)
        return h.astype(dtype)
    elif type == 'gaussian':
        p2 = args[0]
        p3 = args[1]
        siz = np.array([(p2[0] - 1) / 2, (p2[1] - 1) / 2])
        std = p3
        x1 = np.arange(-siz[1], siz[1] + 1, 1)
        y1 = np.arange(-siz[0], siz[0] + 1, 1)
        x, y = np.meshgrid(x1, y1)
        arg = -(x * x + y * y) / (2 * std * std)
        h = np.exp(arg)
        sumh = sum(map(sum, h))
        if sumh != 0:
            h = h / sumh
        return h.astype(dtype)
    elif type == 'motion':
        p2 = args[0]
        p3 = args[1]
        len = max(1, p2)
        half = (len - 1) / 2
        phi = np.mod(p3, 180) / 180 * np.pi

        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xsign = np.sign(cosphi)
        linewdt = 1

        eps = np.finfo(float).eps
        sx = np.fix(half * cosphi + linewdt * xsign - len * eps)
        sy = np.fix(half * sinphi + linewdt - len * eps)

        x1 = np.arange(0, sx + 1, xsign)
        y1 = np.arange(0, sy + 1, 1)
        x, y = np.meshgrid(x1, y1)

        dist2line = (y * cosphi - x * sinphi)
        rad = np.sqrt(x * x + y * y)

        lastpix = np.logical_and(rad >= half, np.abs(dist2line) <= linewdt)
        lastpix.astype(int)
        x2lastpix = half * lastpix - np.abs((x * lastpix + dist2line * lastpix * sinphi) / cosphi)
        dist2line = dist2line * (-1 * lastpix + 1) + np.sqrt(dist2line ** 2 + x2lastpix ** 2) * lastpix
        dist2line = linewdt + eps - np.abs(dist2line)
        logic = dist2line < 0
        dist2line = dist2line * (-1 * logic + 1)

        h1 = np.rot90(dist2line, 2)
        h1s = np.shape(h1)
        h = np.zeros(shape=(h1s[0] * 2 - 1, h1s[1] * 2 - 1))
        h[0:h1s[0], 0:h1s[1]] = h1
        h[h1s[0] - 1:, h1s[1] - 1:] = dist2line
        h = h / sum(map(sum, h)) + eps * len * len

        if cosphi > 0:
            h = np.flipud(h)

        return h.astype(dtype)


def cconv_np(data, ker):
    return filters.convolve(data, ker, mode='wrap')


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def cconv_torch(x, ker):
    with torch.no_grad():
        x_h, x_v = x.size()
        conv_ker = np.flip(np.flip(ker, 0), 1)
        ker = torch.FloatTensor(conv_ker.copy()).cuda()
        k_h, k_v = ker.size()
        k2_h = k_h // 2
        k2_v = k_v // 2
        x = torch.cat((x[-k2_h:, :], x, x[0:k2_h, :]), dim=0).cuda()
        x = torch.cat((x[:, -k2_v:], x, x[:, 0:k2_v]), dim=1).cuda()
        x = x.unsqueeze(0).cuda()
        x = x.unsqueeze(1).cuda()
        ker = ker.unsqueeze(0).cuda()
        ker = ker.unsqueeze(1).cuda()
        y1 = F.conv2d(x, ker).cuda()
        y1 = torch.squeeze(y1)
        y = y1[-x_h:, -x_v:]
    return y


def pad_for_kernel(img, kernel, mode):
    p = [(d - 1) // 2 for d in kernel.shape]
    # padding = [p, p] + (img.ndim - 2) * [(0, 0)]
    padding = [p, p]
    img_pad = np.pad(img, padding, mode)

    return img_pad


def edgetaper_alpha(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, axis=1 - i), img_shape[i] - 1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z / np.max(z))
    return np.outer(*v)


def edgetaper(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[..., np.newaxis]
        alpha = alpha[..., np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img, _kernel, 'wrap'), kernel, mode='valid')
        img = alpha * img + (1 - alpha) * blurred
    return img


def rgb2ycbcr(im_rgb, to_255=False):
    '''Imitattion MATLAB function rgb2ycbcr '''
    try:
        assert np.max(im_rgb) <= 1.05
    except:
        raise ('The input image should be scale to 0-1')

    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0  # to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0  # to [16/255, 240/255]

    if to_255:
        im_ycbcr = np.round((im_ycbcr * 255)).astype(int)

    return im_ycbcr


from skimage.feature import corner_harris, corner_peaks, plot_matches
from skimage.measure import ransac
from skimage.transform import warp
from skimage.transform import EuclideanTransform


def alignment(src_img, dst_img, max_translation=20, max_angle=2):
    '''
        Perform Euclidean Transformation to facilitate PSNR and SSIM computation. The features of each image is obtained
        via corner detection and the transformation matrix is obtained by the ransac algorithms

    '''
    src_gray = rgb2gray(src_img)
    dst_gray = rgb2gray(dst_img)
    src_gray = src_gray.unsqueeze(0).unsqueeze(0)
    dst_gray = dst_gray.unsqueeze(0).unsqueeze(0)
    src_gray = F.interpolate(src_gray, scale_factor=1 / 8).squeeze()
    dst_gray = F.interpolate(dst_gray, scale_factor=1 / 8).squeeze()

    def get_coords(im, exclude_border=20, num_peaks=50):
        corner_tmp = corner_harris(im)
        coords = corner_peaks(corner_tmp[exclude_border:-exclude_border + 1, exclude_border:-exclude_border + 1],
                              min_distance=10, num_peaks=num_peaks)
        coords += np.array([exclude_border, exclude_border])
        return coords

    coords_src = get_coords(src_gray, num_peaks=50)
    coords_dst = get_coords(dst_gray, num_peaks=np.inf)

    ###
    plt.rcParams["figure.figsize"] = [40, 20]
    fig, ax = plt.subplots()
    plt.gray()
    idx = np.array([True])
    inlier_idxs = np.nonzero(idx)[0]
    # plot_matches(ax, src_img, dst_img, src, dst,
    # np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
    plot_matches(ax, src_gray, dst_gray, coords_src, coords_dst,
                 np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')

    ax.axis('off')
    ax.set_title('Correct correspondences')

    plt.savefig("tmp/image1.png")

    src = []
    dst = []

    def gaussian_weights(window_ext, sigma=1):
        y, x = np.mgrid[-window_ext:window_ext + 1, -window_ext:window_ext + 1]
        g = np.zeros(y.shape, dtype=np.double)
        g[:] = np.exp(-0.5 * (x ** 2 / sigma ** 2 + y ** 2 / sigma ** 2))
        g /= 2 * np.pi * sigma * sigma
        return g

    def match_corner(coord, window_ext=5, max_d=20):
        r, c = np.round(coord).astype(np.intp)
        window_orig = dst_img[r - window_ext:r + window_ext + 1, c - window_ext:c + window_ext + 1, :]  # img_orig

        weights = gaussian_weights(window_ext, 3)
        weights = np.dstack((weights, weights, weights))

        SSDs = []
        coords_dst_neighbor = []
        d = max_d
        for cr, cc in coords_src:
            if np.abs(cr - r) <= d and np.abs(cc - c) <= d:
                window_warped = src_img[cr - window_ext:cr + window_ext + 1, cc - window_ext:cc + window_ext + 1,
                                :]  # img_warped
                SSD = np.sum(weights * (window_orig - window_warped) ** 2)
                SSDs.append(SSD)
                coords_dst_neighbor.append([cr, cc])

        # use corner with minimum SSD as correspondence
        try:
            min_idx = np.argmin(SSDs)
            return coords_dst_neighbor[min_idx]
        except:
            return None

    for coord in coords_dst:
        matched_coord = match_corner(coord, window_ext=10, max_d=8)
        if matched_coord is not None:
            dst.append(coord)
            src.append(matched_coord)
    src = np.array(src)
    dst = np.array(dst)

    model_robust, inliers = ransac((np.flip(src, axis=-1), np.flip(dst, axis=-1)), EuclideanTransform,
                                   min_samples=2, residual_threshold=5, stop_sample_num=50,
                                   max_trials=1000)  # residual_threshold=5
    # if np.abs(np.arcsin(model_robust.rotation)) >= max_angle or np.max(np.abs(model_robust.translation)) >= max_translation \
    #     or np.sum(inliers == True ) < 2:
    #     return None
    # else:
    src_warped = warp(src_img, model_robust.inverse)
    #     return src_warped
    # imshow(src_warped, 'src_warped')
    plt.rcParams["figure.figsize"] = [40, 20]

    fig, ax = plt.subplots()
    plt.gray()

    inlier_idxs = np.nonzero(inliers)[0]
    # plot_matches(ax, src_img, dst_img, src, dst,
    # np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
    plot_matches(ax, src_img, dst_img, coords_src, coords_dst,
                 np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')

    ax.axis('off')
    ax.set_title('Correct correspondences')

    plt.savefig("tmp/image.png")

    imshow(src_img, 'src')
    imshow(dst_img, 'dst')
    imshow(src_warped, 'src_warped')

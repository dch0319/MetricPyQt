import time

from skimage import io
from skimage.util import img_as_float32

from utils.metrics import comp_upto_shif_rot_algn_color_multiprocess


def adjust_image_size(image, target_shape):
    """
    调整图像尺寸，通过裁剪较大图像的边缘使其与目标尺寸一致。

    :param image: 需要调整的图像 (numpy array)
    :param target_shape: 目标图像的形状 (height, width, channels)
    :return: 调整后的图像
    """
    height, width, channels = target_shape
    current_height, current_width, _ = image.shape

    # 计算需要裁剪的像素数
    crop_height = (current_height - height) // 2
    crop_width = (current_width - width) // 2

    # 裁剪图像
    if crop_height > 0 and crop_width > 0:
        return image[crop_height:crop_height + height, crop_width:crop_width + width, :]
    else:
        return image


restored_image_path = r"C:\Users\Admin\Downloads\media_images_image_4900_c208be9f2343fb25df19.png"
gt_image_path = r"E:\CV\dataset\deblur\Lai\synthetic_dataset\ground_truth\manmade_01.png"
restored_image = io.imread(restored_image_path)[:, :, :3]  # RGBA -> RGB
gt_image = io.imread(gt_image_path)[:, :, :3]
# Ensure images have the same dimensions
if restored_image.shape != gt_image.shape:
    # 获取较小的图像尺寸
    target_shape = min(restored_image.shape, gt_image.shape, key=lambda x: x[0] * x[1])

    # 调整较大图像的尺寸
    if restored_image.shape != target_shape:
        restored_image = adjust_image_size(restored_image, target_shape)
    if gt_image.shape != target_shape:
        gt_image = adjust_image_size(gt_image, target_shape)
        # Normalize images to [0, 1]
restored_image = img_as_float32(restored_image)
gt_image = img_as_float32(gt_image)
rc = [restored_image]
sp = [gt_image]
time_start = time.time()
# output = aver_bmp_psnr_ssim_rot_par(rc, sp, bd_cut=15, maxshift=14, maxangle=0.5, num_cores=30,
#                                                     shift_inter=1, angle_inter=0.1, ssim_compute=True,
#                                                     verbose=True, show_aligned=True)
# psnr_value, ssim_value = output['PSNR_mean'], output['SSIM_mean']
output = comp_upto_shif_rot_algn_color_multiprocess(restored_image, gt_image, cut=15, maxshift=14, maxangle=0.5,
                                                    shift_inter=1,
                                                    angle_inter=0.1,
                                                    ssim_compute=True)
psnr_value, ssim_value = output[0], output[1]
print(f'PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}')
print(f'Time elapsed: {time.time() - time_start:.2f}s')

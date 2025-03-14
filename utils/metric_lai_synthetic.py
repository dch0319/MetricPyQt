import time

from skimage import io
from skimage.util import img_as_float32

from metrics import comp_upto_shift_multiprocess, comp_upto_shif_algn_color_multiprocess

# 构建文件路径
restored_image_path = r"D:\Workspace\CV\low-level\deblur\MetricPyQt\synthetic\lai\vdip_sparse_ours\restored\text_03_kernel_02.png"
gt_image_path = r"D:\Workspace\CV\low-level\deblur\MetricPyQt\synthetic\lai\ground_truth\text_03.png"

restored_image = img_as_float32(io.imread(restored_image_path)[:, :, :3])
gt_image = img_as_float32(io.imread(gt_image_path)[:, :, :3])

start_time = time.time()
result = comp_upto_shift_multiprocess(restored_image, gt_image)
print(time.time() - start_time)
print(result[0], result[1])
start_time = time.time()
result = comp_upto_shif_algn_color_multiprocess(restored_image, gt_image, shift_inter=0.25)
print(time.time() - start_time)
print(result[0], result[1])

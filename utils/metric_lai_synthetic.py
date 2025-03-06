import os
import re

import pandas as pd
from skimage import io, color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import img_as_float32
from tqdm import tqdm

from metric import comp_upto_shift

folder_path = r"C:\Users\Admin\Desktop\sparse-pre-kernel"
gt_path = r'E:\CV\对比试验\lai\synthetic\ground_truth'
output_excel = 'lai_synthetic.xlsx'

# 获取最后一级目录作为新的sheet名称
sheet_name = os.path.basename(os.path.normpath(folder_path))

# 初始化DataFrame
df = pd.DataFrame(columns=['File Name', 'PSNR', 'SSIM'])

# 遍历文件夹中的所有文件
data_list = []
# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)

# 使用tqdm包装迭代器，显示进度条
for file_name in tqdm(file_names, desc="Processing files", unit="file"):
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # 跳过非图像文件

    if '_k.png' in file_name:
        continue  # 跳过文件名包含'_k'的文件

    file_path = os.path.join(folder_path, file_name)
    # Load and preprocess images
    blurry_image = img_as_float32(io.imread(file_path))[:, :, :3]
    gt_image = img_as_float32(io.imread(os.path.join(gt_path, re.sub(r'_kernel_0\d', '', file_name))))[:, :, :3]

    blurry_image = color.rgb2gray(blurry_image)
    gt_image = color.rgb2gray(gt_image)
    psnr_value, ssim_value = comp_upto_shift(gt_image, blurry_image, maxshift=5)
    # psnr_value = peak_signal_noise_ratio(gt_image, restored_image)
    # ssim_value = structural_similarity(gt_image, restored_image, multichannel=True, channel_axis=2, data_range=1)

    # 去掉文件名中的扩展名和'_x'
    clean_file_name = os.path.splitext(file_name)[0].replace('_x', '')
    # 添加数据到列表中
    data_list.append({
        'File Name': clean_file_name,
        'PSNR': psnr_value,
        'SSIM': ssim_value
    })

# 将列表中的数据转换为DataFrame
df = pd.DataFrame(data_list)

# 使用ExcelWriter来写入Excel文件
with pd.ExcelWriter(output_excel, engine='openpyxl', mode='a') as writer:
    # 写入新的工作表
    df.to_excel(writer, sheet_name=sheet_name, index=False)

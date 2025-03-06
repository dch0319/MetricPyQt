import os

import pandas as pd
import torch
from pyiqa import create_metric
from skimage import io
from skimage.util import img_as_float32

folder_path = r'E:\CV\Code\diffusion-deconv\对比试验\lai\real\selfdeblur'
output_excel = 'lai_real.xlsx'

# 获取最后一级目录作为新的sheet名称
sheet_name = os.path.basename(os.path.normpath(folder_path))

# 初始化DataFrame
df = pd.DataFrame(columns=['File Name', 'NIQE', 'BRISQUE', 'PIQE'])

# 遍历文件夹中的所有文件
data_list = []
for file_name in os.listdir(folder_path):
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # 跳过非图像文件

    if '_k.png' in file_name:
        continue  # 跳过文件名包含'_k'的文件

    file_path = os.path.join(folder_path, file_name)
    # Load and preprocess images
    blurry_image = img_as_float32(io.imread(file_path))[:, :, :3]
    blurry_tensor = torch.tensor(blurry_image).permute(2, 0, 1).unsqueeze(0).cuda()

    # Calculate NIQE, BRISQUE, PIQE
    niqe_metric = create_metric('niqe').cuda()
    niqe_value = niqe_metric(blurry_tensor).item()

    brisque_metric = create_metric('brisque').cuda()
    brisque_value = brisque_metric(blurry_tensor).item()

    piqe_metric = create_metric('piqe').cuda()
    piqe_value = piqe_metric(blurry_tensor).item()

    # 去掉文件名中的扩展名和'_x'
    clean_file_name = os.path.splitext(file_name)[0].replace('_x', '')
    # 添加数据到列表中
    data_list.append({
        'File Name': clean_file_name,
        'NIQE': niqe_value,
        'BRISQUE': brisque_value,
        'PIQE': piqe_value
    })

# 将列表中的数据转换为DataFrame
df = pd.DataFrame(data_list)

# 使用ExcelWriter来写入Excel文件
with pd.ExcelWriter(output_excel, engine='openpyxl', mode='a') as writer:
    # 写入新的工作表
    df.to_excel(writer, sheet_name=sheet_name, index=False)

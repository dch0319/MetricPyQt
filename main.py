import os
import sys

import lpips
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QGuiApplication
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QMessageBox, QHBoxLayout, QCheckBox
)
from pyiqa import create_metric
from skimage import io
from skimage.util import img_as_float32

from metric import comp_upto_shift_rgb, comp_upto_shift_rgb_gpu


class ImageComparisonApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Quality Assessment Tool')

        # Layouts
        self.layout = QVBoxLayout()

        # Labels for displaying images
        self.label_blurry = QLabel("Image for Quality Assessment")
        self.label_gt = QLabel("Ground Truth Image")
        self.label_blurry.setAlignment(Qt.AlignCenter)
        self.label_gt.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_blurry)
        self.label_blurry_dir = QLabel("Blurry Image Directory: N/A")
        self.layout.addWidget(self.label_blurry_dir)
        self.layout.addWidget(self.label_gt)
        self.label_gt_dir = QLabel("Ground Truth Directory: N/A")
        self.layout.addWidget(self.label_gt_dir)
        # Buttons to load images
        self.button_load_blurry = QPushButton('Load Image for Assessment')
        self.button_load_gt = QPushButton('Load Ground Truth Image')
        self.button_load_blurry.clicked.connect(self.load_blurry_image)
        self.button_load_gt.clicked.connect(self.load_gt_image)
        self.layout.addWidget(self.button_load_blurry)
        self.layout.addWidget(self.button_load_gt)

        # Button to calculate metrics
        self.button_calculate = QPushButton('Calculate Metrics')
        self.button_calculate.clicked.connect(self.calculate_metrics)
        self.layout.addWidget(self.button_calculate)

        self.lpips_checkbox = QCheckBox("Compute LPIPS")
        self.lpips_checkbox.setChecked(True)
        self.layout.addWidget(self.lpips_checkbox)

        self.unmetrics_checkbox = QCheckBox("Compute Unsupervised Metrics")
        self.unmetrics_checkbox.setChecked(True)
        self.layout.addWidget(self.unmetrics_checkbox)

        # Results with copy buttons
        self.result_layout_psnr = self.create_result_row("PSNR", self.copy_psnr)
        self.result_layout_ssim = self.create_result_row("SSIM", self.copy_ssim)
        self.result_layout_lpips = self.create_result_row("LPIPS", self.copy_lpips)
        self.result_layout_niqe = self.create_result_row("NIQE", self.copy_niqe)
        self.result_layout_brisque = self.create_result_row("BRISQUE", self.copy_brisque)
        self.result_layout_piqe = self.create_result_row("PIQE", self.copy_piqe)

        # Add "Copy to LaTeX" button
        self.button_copy_latex = QPushButton('Copy to LaTeX')
        self.button_copy_latex.clicked.connect(self.copy_to_latex)
        self.layout.addWidget(self.button_copy_latex)

        self.layout.addLayout(self.result_layout_psnr)
        self.layout.addLayout(self.result_layout_ssim)
        self.layout.addLayout(self.result_layout_lpips)
        self.layout.addLayout(self.result_layout_niqe)
        self.layout.addLayout(self.result_layout_brisque)
        self.layout.addLayout(self.result_layout_piqe)

        self.setLayout(self.layout)

        # Image paths
        self.blurry_image_path = None
        self.gt_image_path = None

        # Metrics
        self.psnr_value = None
        self.ssim_value = None
        self.lpips_value = None
        self.niqe_value = None
        self.brisque_value = None
        self.piqe_value = None

    def create_result_row(self, label_text, copy_action):
        """Helper to create a row for a metric with its label and copy button."""
        row_layout = QHBoxLayout()
        result_label = QLabel(f"{label_text}: N/A")
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(copy_action)
        row_layout.addWidget(result_label)
        row_layout.addWidget(copy_button)

        if label_text == "PSNR":
            self.result_psnr_label = result_label
        elif label_text == "SSIM":
            self.result_ssim_label = result_label
        elif label_text == "LPIPS":
            self.result_lpips_label = result_label
        elif label_text == "NIQE":
            self.result_niqe_label = result_label
        elif label_text == "BRISQUE":
            self.result_brisque_label = result_label
        elif label_text == "PIQE":
            self.result_piqe_label = result_label

        return row_layout

    def load_blurry_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load Image for Assessment', '', 'Images (*.png *.jpg *.jpeg)')
        if file_path:
            self.blurry_image_path = file_path
            pixmap = QPixmap(file_path)
            self.label_blurry.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            # 显示当前文件夹的名称
            folder_name = os.path.basename(os.path.dirname(file_path))
            self.label_blurry_dir.setText(f"Blurry Image Directory: {folder_name}")

    def load_gt_image(self):
        import os  # 确保导入 os
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load Ground Truth Image', '', 'Images (*.png *.jpg *.jpeg)')
        if file_path:
            self.gt_image_path = file_path
            pixmap = QPixmap(file_path)
            self.label_gt.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            # 显示当前文件夹的名称
            folder_name = os.path.basename(os.path.dirname(file_path))
            self.label_gt_dir.setText(f"Ground Truth Directory: {folder_name}")

    def adjust_image_size(self, image, target_shape):
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

    def calculate_metrics(self):
        if not self.blurry_image_path:
            QMessageBox.warning(self, "Warning", "Please load an image for assessment.")
            return

        # Load and preprocess images
        blurry_image = img_as_float32(io.imread(self.blurry_image_path))[:, :, :3]
        blurry_tensor = torch.tensor(blurry_image).permute(2, 0, 1).unsqueeze(0).cuda()

        if self.unmetrics_checkbox.isChecked():
            # Calculate NIQE, BRISQUE, PIQE
            niqe_metric = create_metric('niqe').cuda()
            self.niqe_value = niqe_metric(blurry_tensor).item()

            brisque_metric = create_metric('brisque').cuda()
            self.brisque_value = brisque_metric(blurry_tensor).item()

            piqe_metric = create_metric('piqe').cuda()
            self.piqe_value = piqe_metric(blurry_tensor).item()
        else:
            self.niqe_value = self.brisque_value = self.piqe_value = None

        if self.gt_image_path:
            # Load images
            blurry_image = io.imread(self.blurry_image_path)[:, :, :3]  # RGBA -> RGB
            gt_image = io.imread(self.gt_image_path)[:, :, :3]

            # Ensure images have the same dimensions
            if blurry_image.shape != gt_image.shape:
                # 获取较小的图像尺寸
                target_shape = min(blurry_image.shape, gt_image.shape, key=lambda x: x[0] * x[1])

                # 调整较大图像的尺寸
                if blurry_image.shape != target_shape:
                    blurry_image = self.adjust_image_size(blurry_image, target_shape)
                if gt_image.shape != target_shape:
                    gt_image = self.adjust_image_size(gt_image, target_shape)

                # 如果调整后尺寸仍然不一致，弹出警告
                if blurry_image.shape != gt_image.shape:
                    QMessageBox.warning(self, "Warning", "无法将图像调整为相同尺寸。")
                    return

            # Normalize images to [0, 1]
            blurry_image = img_as_float32(blurry_image)
            gt_image = img_as_float32(gt_image)

            # Compute PSNR and SSIM using comp_upto_shift
            self.psnr_value, self.ssim_value = comp_upto_shift_rgb_gpu(gt_image, blurry_image, maxshift=5)

            if self.lpips_checkbox.isChecked():
                loss_fn = lpips.LPIPS(net='vgg').cuda()
                # 转为 PyTorch tensor
                gt_tensor = torch.tensor(gt_image).permute(2, 0, 1).unsqueeze(0).cuda()
                blurry_tensor = torch.tensor(blurry_image).permute(2, 0, 1).unsqueeze(0).cuda()
                # 计算 LPIPS 值
                lpips_value = loss_fn(gt_tensor, blurry_tensor)
                self.lpips_value = lpips_value.item()
            else:
                self.lpips_value = None

        # Update UI
        self.result_psnr_label.setText(f"PSNR: {self.psnr_value:.2f}" if self.psnr_value else "PSNR: N/A")
        self.result_ssim_label.setText(f"SSIM: {self.ssim_value:.3f}" if self.ssim_value else "SSIM: N/A")
        self.result_lpips_label.setText(f"LPIPS: {self.lpips_value:.3f}" if self.lpips_value else "LPIPS: N/A")
        self.result_niqe_label.setText(f"NIQE: {self.niqe_value:.4f}")
        self.result_brisque_label.setText(f"BRISQUE: {self.brisque_value:.4f}")
        self.result_piqe_label.setText(f"PIQE: {self.piqe_value:.4f}")

    def copy_psnr(self):
        if self.psnr_value is not None:
            QGuiApplication.clipboard().setText(f"{self.psnr_value:.2f}")
            QMessageBox.information(self, "Copied", "PSNR value copied to clipboard.")

    def copy_ssim(self):
        if self.ssim_value is not None:
            QGuiApplication.clipboard().setText(f"{self.ssim_value:.3f}")
            QMessageBox.information(self, "Copied", "SSIM value copied to clipboard.")

    def copy_lpips(self):
        if self.lpips_value is not None:
            QGuiApplication.clipboard().setText(f"{self.lpips_value:.3f}")
            QMessageBox.information(self, "Copied", "LPIPS value copied to clipboard.")

    def copy_niqe(self):
        if self.niqe_value is not None:
            QGuiApplication.clipboard().setText(f"{self.niqe_value:.4f}")
            QMessageBox.information(self, "Copied", "NIQE value copied to clipboard.")

    def copy_brisque(self):
        if self.brisque_value is not None:
            QGuiApplication.clipboard().setText(f"{self.brisque_value:.4f}")
            QMessageBox.information(self, "Copied", "BRISQUE value copied to clipboard.")

    def copy_piqe(self):
        if self.piqe_value is not None:
            QGuiApplication.clipboard().setText(f"{self.piqe_value:.4f}")
            QMessageBox.information(self, "Copied", "PIQE value copied to clipboard.")

    def copy_to_latex(self):
        latex_table = (
                (f"{self.psnr_value:.2f}" if self.psnr_value is not None else "N/A") + " & " +
                (f"{self.ssim_value:.3f}" if self.ssim_value is not None else "N/A") + " & " +
                (f"{self.lpips_value:.3f}" if self.lpips_value is not None else "N/A") + " & " +
                (f"{self.niqe_value:.4f}" if self.niqe_value is not None else "N/A") + " & " +
                (f"{self.brisque_value:.4f}" if self.brisque_value is not None else "N/A") + " & " +
                (f"{self.piqe_value:.4f}" if self.piqe_value is not None else "N/A")
        )
        QGuiApplication.clipboard().setText(latex_table)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageComparisonApp()
    ex.show()
    sys.exit(app.exec_())

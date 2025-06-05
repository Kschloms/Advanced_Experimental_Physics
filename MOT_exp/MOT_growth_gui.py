import sys
import os
import numpy as np
import cv2 as cv
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QTabWidget,
    QPushButton, QLineEdit, QFileDialog, QMessageBox, QScrollArea, QGridLayout, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])

def import_images(image_folder):
    image_files = [
        os.path.join(image_folder, f)
        for f in sorted(os.listdir(image_folder))
        if f.lower().endswith(('.tif', '.tiff'))
    ]
    images = []
    for file in image_files:
        img = cv.imread(file)
        if img is not None:
            images.append(img)
    return images

def contour_touches_border(contour, img_shape):
    h, w = img_shape[:2]
    for point in contour:
        x, y = point[0]
        if x <= 1 or y <= 1 or x >= w-2 or y >= h-2:
            return True
    return False

def calculate_mot_size(images, contour_threshold, area_threshold=50):
    mot_sizes = []
    contours_list = []
    for img in images:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, binary_mask = cv.threshold(gray_img, contour_threshold, 255, 0)
        contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if not contour_touches_border(c, img.shape)]
        if valid_contours:
            largest_contour = max(valid_contours, key=cv.contourArea)
            area = cv.contourArea(largest_contour)
            if area > area_threshold:
                mot_sizes.append(area)
                contours_list.append(largest_contour)
            else:
                mot_sizes.append(0)
                contours_list.append(None)
        else:
            mot_sizes.append(0)
            contours_list.append(None)
    return mot_sizes, contours_list

def cvimg_to_qpixmap(img):
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

class MotGrowthGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOT Growth Analysis")
        self.images = []
        self.folder = ""
        self.contour_threshold = 10
        self.area_threshold = 50
        self.mot_sizes = []
        self.contours_list = []
        # Default time points
        self.time_start = 0.1
        self.time_end = 5.0
        self.time_step = 0.2
        self.time_points = np.arange(self.time_start, self.time_end, self.time_step)
        self.cooling_power = 20  # Default value
        self.detuning_latex = ""  # For LaTeX display in title
        self.detuning_mhz = ""    # For MHz value if needed
        self.detuning_dict = {
            "": "",           # For "no detuning" option
            "0Γ": r"$0\Gamma$",
            "1Γ": r"$1\Gamma$",
            "2Γ": r"$2\Gamma$",
            "3Γ": r"$3\Gamma$",
            "4Γ": r"$4\Gamma$"
        }
        self.detuning_mhz_dict = {
            "": "",
            "0Γ": "105.9 MHz",
            "1Γ": "102.9 MHz",
            "2Γ": "99.84 MHz",
            "3Γ": "96.80 MHz",
            "4Γ": "93.77 MHz"
        }
        self.pixel_size = 4.8e-6  # Pixel size in meters (4.8 microns)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Cooling power, detuning, and pixel size input
        cp_layout = QHBoxLayout()
        cp_layout.addWidget(QLabel("Cooling Power (%):"))
        self.cp_input = QLineEdit(str(self.cooling_power))
        self.cp_input.setFixedWidth(60)
        self.cp_input.editingFinished.connect(self.update_cooling_power)
        cp_layout.addWidget(self.cp_input)

        cp_layout.addWidget(QLabel("Detuning:"))
        self.detuning_combo = QComboBox()
        for key in self.detuning_dict:
            self.detuning_combo.addItem(key)
        self.detuning_combo.currentIndexChanged.connect(self.update_detuning_combo)
        cp_layout.addWidget(self.detuning_combo)

        cp_layout.addWidget(QLabel("Pixel Size (m):"))
        self.pixel_size_input = QLineEdit(str(self.pixel_size))
        self.pixel_size_input.setFixedWidth(80)
        self.pixel_size_input.editingFinished.connect(self.update_pixel_size)
        cp_layout.addWidget(self.pixel_size_input)

        cp_layout.addStretch()
        layout.addLayout(cp_layout)

        # Time points input
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Start Time (s):"))
        self.time_start_input = QLineEdit(str(self.time_start))
        self.time_start_input.setFixedWidth(60)
        self.time_start_input.editingFinished.connect(self.update_time_points)
        time_layout.addWidget(self.time_start_input)

        time_layout.addWidget(QLabel("End Time (s):"))
        self.time_end_input = QLineEdit(str(self.time_end))
        self.time_end_input.setFixedWidth(60)
        self.time_end_input.editingFinished.connect(self.update_time_points)
        time_layout.addWidget(self.time_end_input)

        time_layout.addWidget(QLabel("Step Size (s):"))
        self.time_step_input = QLineEdit(str(self.time_step))
        self.time_step_input.setFixedWidth(60)
        self.time_step_input.editingFinished.connect(self.update_time_points)
        time_layout.addWidget(self.time_step_input)

        time_layout.addStretch()
        layout.addLayout(time_layout)

        # Folder input
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Enter folder path or browse...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_folder)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_images)
        folder_layout.addWidget(QLabel("Image Folder:"))
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(browse_btn)
        folder_layout.addWidget(load_btn)
        layout.addLayout(folder_layout)

        # Contour threshold slider
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setValue(self.contour_threshold)
        self.slider.valueChanged.connect(self.update_threshold)
        self.slider_label = QLabel(f"Contour Threshold: {self.contour_threshold}")
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.slider)
        layout.addLayout(slider_layout)

        # Tabs
        self.tabs = QTabWidget()
        # Tab 1: Images
        self.images_tab = QWidget()
        self.images_layout = QVBoxLayout(self.images_tab)
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.grid_layout = QGridLayout(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)
        self.images_layout.addWidget(self.scroll_area)
        self.tabs.addTab(self.images_tab, "Images + Contour")

        # Tab 2: Plot
        self.plot_tab = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_tab)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)
        # Save plot button
        save_btn = QPushButton("Save Plot")
        save_btn.clicked.connect(self.save_plot)
        self.plot_layout.addWidget(save_btn)
        # Export data button
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        self.plot_layout.addWidget(export_btn)
        self.tabs.addTab(self.plot_tab, "MOT Size vs Time")

        # Tab 3: Intensity
        self.intensity_tab = QWidget()
        self.intensity_layout = QVBoxLayout(self.intensity_tab)
        self.intensity_figure, self.intensity_ax = plt.subplots()
        self.intensity_canvas = FigureCanvas(self.intensity_figure)
        self.intensity_layout.addWidget(self.intensity_canvas)
        self.tabs.addTab(self.intensity_tab, "MOT Intensity vs Time")

        # Save plot button for intensity
        self.intensity_save_btn = QPushButton("Save Plot")
        self.intensity_save_btn.clicked.connect(self.save_intensity_plot)
        self.intensity_layout.addWidget(self.intensity_save_btn)

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def update_cooling_power(self):
        try:
            self.cooling_power = float(self.cp_input.text())
        except ValueError:
            self.cooling_power = 20  # fallback
            self.cp_input.setText(str(self.cooling_power))
        self.update_plot_tab()

    def update_detuning_combo(self):
        key = self.detuning_combo.currentText()
        self.detuning_latex = self.detuning_dict.get(key, "")
        self.detuning_mhz = self.detuning_mhz_dict.get(key, "")
        self.update_plot_tab()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.folder_input.setText(folder)

    def load_images(self):
        folder = self.folder_input.text().strip()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Error", "Invalid folder path.")
            return
        self.folder = folder
        self.images = import_images(folder)
        if not self.images:
            QMessageBox.warning(self, "Error", "No images found in folder.")
            return
        self.update_analysis()

    def update_threshold(self):
        self.contour_threshold = self.slider.value()
        self.slider_label.setText(f"Contour Threshold: {self.contour_threshold}")
        if self.images:
            self.update_analysis()

    def update_time_points(self):
        try:
            self.time_start = float(self.time_start_input.text())
            self.time_end = float(self.time_end_input.text())
            self.time_step = float(self.time_step_input.text())
            if self.time_step <= 0 or self.time_end <= self.time_start:
                raise ValueError
            self.time_points = np.arange(self.time_start, self.time_end, self.time_step)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid time input values.")
            # Reset to previous valid values
            self.time_start_input.setText(str(self.time_start))
            self.time_end_input.setText(str(self.time_end))
            self.time_step_input.setText(str(self.time_step))
            return
        self.update_plot_tab()

    def update_pixel_size(self):
        try:
            self.pixel_size = float(self.pixel_size_input.text())
        except ValueError:
            self.pixel_size = 4.8e-6  # fallback
            self.pixel_size_input.setText(str(self.pixel_size))
        self.update_plot_tab()

    def calculate_mot_intensity(self): #Average greyscale brightness of MOT within contours
        intensities = []
        for img, contour in zip(self.images, self.contours_list):
            if contour is not None:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv.drawContours(mask, [contour], -1, 255, -1)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                mot_pixels = gray[mask == 255]
                if mot_pixels.size > 0:
                    intensities.append(np.mean(mot_pixels))
                else:
                    intensities.append(0)
            else:
                intensities.append(0)
        return intensities

    def update_analysis(self):
        self.mot_sizes, self.contours_list = calculate_mot_size(
            self.images, self.contour_threshold, self.area_threshold
        )
        self.update_images_tab()
        self.update_plot_tab()
        self.update_intensity_tab()  

    def update_images_tab(self):
        # Remove old widgets
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        # Show images with contours
        cols = 5
        for idx, img in enumerate(self.images):
            img_disp = img.copy()
            contour = self.contours_list[idx]
            if contour is not None:
                cv.drawContours(img_disp, [contour], -1, (0, 255, 0), 2)
            pixmap = cvimg_to_qpixmap(img_disp)
            label = QLabel()
            label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
            self.grid_layout.addWidget(label, idx // cols, idx % cols)

    def update_plot_tab(self):
        self.ax.clear()
        # Convert mot_sizes (pixels) to mm^2
        mot_sizes_real = [area * (self.pixel_size ** 2) * 1e6 for area in self.mot_sizes]  # mm^2
        self.ax.plot(self.time_points[:len(mot_sizes_real)], mot_sizes_real, marker='o')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('MOT Area (mm$^2$)')
        # Title logic with LaTeX
        title = f'MOT Growth, Cooling Power: {self.cooling_power:.0f}%'
        if self.detuning_latex:
            title += f', Detuning: {self.detuning_latex}, AOM Frequency: {self.detuning_mhz}'
        self.ax.set_title(title)
        self.ax.grid(True)
        self.canvas.draw()

    def update_intensity_tab(self):
        intensities = self.calculate_mot_intensity()
        self.intensity_ax.clear()
        self.intensity_ax.plot(self.time_points[:len(intensities)], intensities, marker='o')
        self.intensity_ax.set_xlabel('Time (s)')
        self.intensity_ax.set_ylabel('Mean MOT Intensity')
        self.intensity_ax.set_title('MOT Intensity vs Time')
        self.intensity_ax.grid(True)
        self.intensity_canvas.draw()

    def save_plot(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)")
        if file_path:
            self.figure.savefig(file_path)

    def export_data(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(
            self, "Export Data", "", "NumPy Archive (*.npz);;All Files (*)"
        )
        if file_path:
            mot_sizes_pixels = np.array(self.mot_sizes)
            time_points = np.array(self.time_points[:len(mot_sizes_pixels)])
            np.savez(file_path, time=time_points, mot_area_pixels=mot_sizes_pixels)

    def save_intensity_plot(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save Intensity Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)")
        if file_path:
            self.intensity_figure.savefig(file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MotGrowthGUI()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())
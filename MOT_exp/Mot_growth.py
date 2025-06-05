import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
# Analysis of images to find growth rate of Magneto optical trap
def import_images(image_folder):
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.tif', '.tiff'))
    ]
    images = []
    for file in image_files:
        img = cv.imread(file)
        images.append(img)
    return images

def contour_touches_border(contour, img_shape):
    h, w = img_shape[:2]
    for point in contour:
        x, y = point[0]
        if x <= 1 or y <= 1 or x >= w-2 or y >= h-2:
            return True
    return False


def calculate_mot_size(images):
    mot_sizes = []
    area_threshold = 50  # Set threshold as needed
    contour_threshold = 30
    for img in images:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, binary_mask = cv.threshold(gray_img, contour_threshold, 255, 0)
        contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Filter out border-touching contours
        valid_contours = [c for c in contours if not contour_touches_border(c, img.shape)]
        if valid_contours:
            largest_contour = max(valid_contours, key=cv.contourArea)
            area = cv.contourArea(largest_contour)
            if area > area_threshold:
                mot_sizes.append(area)
            else:
                mot_sizes.append(0)
        else:
            mot_sizes.append(0)

    # Plot all images with contours on single figure
    _, axs = plt.subplots(int(5), int(len(images)/5), layout='tight', figsize=(15, 5))
    axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration
    for i, img in enumerate(images):
        axs[i].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # axs[i].set_title(f'Image {i+1}')
        axs[i].axis('off')
        if mot_sizes and mot_sizes[i] > area_threshold:
            grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, binary_mask = cv.threshold(grey_img, contour_threshold, 255, 0)
            contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if not contour_touches_border(c, img.shape)]
            if valid_contours:
                largest_contour = max(valid_contours, key=cv.contourArea)
                cv.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
        axs[i].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    
    return mot_sizes

def plot_growth_rate(ax : plt.axes, mot_sizes, time_points):
    ax.plot(time_points, mot_sizes, marker='o')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MOT Area (pixels)')
    ax.grid(True)


if __name__ == "__main__":
    # CP_20_images_folder = r"Data\Day2\first_10_s_CP_100\CP_20"
    CP_images_folder = input("Enter the folder path containing MOT images (e.g., 'Data\\Day2\\first_10_s_CP_100\\CP_20'): ")
    # Get all .tif or .tiff files in the folder

    cooling_power = CP_images_folder.split('\\')[-1]  # Extract cooling power from folder name
    cooling_power = cooling_power.split('_')[1]  # Assuming the format is 'CP_20'
    cooling_power = int(cooling_power)  # Convert to integer
    print(f"Analyzing images for cooling power: {cooling_power} %")
    # Get user input for the folder containing images
    
    CP_images = import_images(CP_images_folder)
    # images taken at 0.2 second intervals except for first image at 0.1 seconds
    time_points = np.arange(0.1, 5, 0.2)  
    # time_points = np.insert(time_points, 0, 0.1)  # Insert the first time point at 0 seconds
    print(time_points)

    CP_mot_sizes = calculate_mot_size(CP_images)
    fig, ax = plt.subplots(layout='tight')
    ax.set_title('MOT Growth, Cooling Power: {}%'.format(cooling_power))
    plot_growth_rate(ax, CP_mot_sizes, time_points)
    plt.show()


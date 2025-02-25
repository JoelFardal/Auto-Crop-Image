from time import time
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from fractions import Fraction

def auto_crop(file_name, output_folder, unique_id):
    """
    Input arguments: 
    - file_name (Image file name, e.g., 'rose.tif')
    - output_folder (Folder to save the cropped images)
    - unique_id (Unique integer ID for the output file name)
    
    This function will auto crop the given image using image processing technique and add margins to make it 1000x1000 pixels.
    
    Output: ROI
    """
    # Start timer
    start_time = time()
    # Read an image
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    height = img.shape[0]
    width = img.shape[1]
    # Check image is grayscale or not 
    if len(img.shape) == 2:
        gray_img = img.copy()
    else:
        # Convert bgr image to grayscale image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # To find upper threshold, we need to apply Otsu's thresholding
    upper_thresh, _ = cv2.threshold(gray_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Calculate lower threshold
    lower_thresh = int(0.5 * upper_thresh)
    # Apply canny edge detection
    canny = cv2.Canny(img, lower_thresh, upper_thresh)
    # Finding the non-zero points of canny
    pts = np.argwhere(canny > 0)
    # Finding the min and max points
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)
    # Crop ROI from the given image
    roi_img = img[y1:y2, x1:x2]
    
    # Calculate the required padding to make the image 1000x1000
    roi_height, roi_width = roi_img.shape[:2]
    top_padding = max((1000 - roi_height) // 2, 0)
    bottom_padding = max(1000 - roi_height - top_padding, 0)
    left_padding = max((1000 - roi_width) // 2, 0)
    right_padding = max(1000 - roi_width - left_padding, 0)
    
    # Add padding to the ROI image
    padded_img = cv2.copyMakeBorder(roi_img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Ensure the padded image is 1000x1000 pixels
    padded_height, padded_width = padded_img.shape[:2]
    if padded_height < 1000 or padded_width < 1000:
        padded_img = cv2.copyMakeBorder(padded_img, 0, 1000 - padded_height, 0, 1000 - padded_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Printing image dimensions, execution time
    print(f'Original image dimensions: {width}x{height}')
    print(f'Execution time: {time() - start_time} sec')
    
    # Load the image metadata
    def get_exif_data(image_path):
        image = Image.open(image_path)
        exif_data = {}
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
        return exif_data

    exif_data = get_exif_data(file_name)

    # Extract required metadata
    date_time = exif_data.get('DateTimeOriginal', 'N/A')
    exposure_time = exif_data.get('ExposureTime', 'N/A')
    if isinstance(exposure_time, tuple) and len(exposure_time) == 2:
        exposure_time = f'{exposure_time[0]}/{exposure_time[1]}'
    else:
        exposure_time = f'1/{int(1/float(exposure_time))}' if exposure_time != 'N/A' else 'N/A'
    f_stop = exif_data.get('FNumber', 'N/A')
    focal_length = exif_data.get('FocalLength', 'N/A')
    iso = exif_data.get('ISOSpeedRatings', 'N/A')

    # Add metadata text to the image
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.5  # Increased font scale to make the text 3 times as big
    color = (255, 255, 255)  # White color
    thickness = 2  # Increased thickness for better visibility

    text_top_left = f'Date/Time: {date_time}\nExposure: {exposure_time}\nF-Stop: {f_stop}\nFocal Length: {focal_length}\nISO: {iso}'
    text_bottom_right = f'Image Number: {os.path.basename(file_name)}'

    # Add text to the top left corner
    y0, dy = 60, 60  # Adjusted y0 and dy to accommodate larger text
    for i, line in enumerate(text_top_left.split('\n')):
        y = y0 + i * dy
        cv2.putText(padded_img, line, (30, y), font, font_scale, color, thickness, cv2.LINE_AA)
        print(f'Added text "{line}" at position (30, {y})')

    # Add text to the bottom right corner
    text_size = cv2.getTextSize(text_bottom_right, font, font_scale, thickness)[0]
    text_x = max(padded_img.shape[1] - text_size[0] - 30, 0)
    text_y = max(padded_img.shape[0] - 30, text_size[1])
    cv2.putText(padded_img, text_bottom_right, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    print(f'Added text "{text_bottom_right}" at position ({text_x}, {text_y})')

    # Save the cropped and padded image with text
    output_file_name = f'sunspot_md_{unique_id:06d}.jpg'
    output_path = os.path.join(output_folder, output_file_name)
    cv2.imwrite(output_path, padded_img)
    print(f'Cropped and padded image saved to: {output_path}')

    return padded_img

def image_attributes(img):
    """
    Input argument: img (Opencv mat)
    
    This function will display the following image attributes:
    - Height and Width
    - Color Channel
    - DPI
    - Max/Min/Average Intensity Values

    Output: image attributes
    """
    height = img.shape[0]
    width = img.shape[1]
    if len(img.shape) == 2:
        no_of_channels = 1
    else:
        no_of_channels = img.shape[2]
    bit_depth = no_of_channels*8
    storage_size = int((height*width*bit_depth)/8)
    # Calculate intensity value
    min_intensity = img.min(axis=0).min(axis=0)
    max_intensity = img.max(axis=0).max(axis=0)
    average_intensity = img.mean(axis=0).mean(axis=0).astype(int)

    print(f'- Image dimensions: {width}x{height}')
    print(f'- Height (rows): {height} pixels')
    print(f'- Width (columns): {width} pixels')
    print(f'- No. of pixels: {height*width}')
    print(f'- Color channels: {no_of_channels}')
    print(f'- Bit depth: {bit_depth}')
    print(f'- Storage size (without compression)): {storage_size} bytes')
    print('- Intensity Values')
    if no_of_channels == 1:
        print(f'\tMin Intensity: {min_intensity}')
        print(f'\tMax Intensity: {max_intensity}')
        print(f'\tAverage Intensity: {average_intensity}')
    elif no_of_channels == 3:
        print(f'\tMin Intensity (Blue): {min_intensity[0]}')
        print(f'\tMax Intensity (Blue): {max_intensity[0]}')
        print(f'\tAverage Intensity (Blue): {average_intensity[0]}')
        print(f'\tMin Intensity (Green): {min_intensity[1]}')
        print(f'\tMax Intensity (Green) {max_intensity[1]}')
        print(f'\tAverage Intensity (Green): {average_intensity[1]}')
        print(f'\tMin Intensity (Red): {min_intensity[2]}')
        print(f'\tMax Intensity (Red) {max_intensity[2]}')
        print(f'\tAverage Intensity (Red): {average_intensity[2]}')

def image_features(img):
    """
    Input argument: img (Opencv mat)
    
    This function will display the following image features:
    - Edges 
    - Corners / interest points
    - Blobs / regions of interest points
    - Ridges

    Output: display image features
    """
    # Apply canny edge detection
    canny_img = cv2.Canny(img, 50, 200)
    # Check image is grayscale or not 
    if len(img.shape) == 2:
        gray_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        # Convert bgr image to grayscale image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get heatmap image
    heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)

    # Corners detection
    corner_img = img.copy()
    gray_img_float = np.float32(gray_img)
    dst = cv2.cornerHarris(gray_img_float,2,3,0.04)
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value
    corner_img[dst>0.01*dst.max()]=[0,0,255]

    ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
    ridges_img = ridge_filter.getRidgeFilteredImage(img)

    # Display image features
    fig = plt.figure()
    fig.suptitle('Image Features', fontsize=16)
    
    fig.add_subplot(2,2, 1).set_title('Edges')
    plt.imshow(cv2.cvtColor(canny_img, cv2.COLOR_BGR2RGB))
    
    fig.add_subplot(2,2, 2).set_title('Corners')
    plt.imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))

    fig.add_subplot(2,2, 3).set_title('Ridges')
    plt.imshow(cv2.cvtColor(ridges_img, cv2.COLOR_BGR2RGB))
    
    fig.add_subplot(2,2, 4).set_title('Heatmap')
    plt.imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
    plt.show()

def get_exif_date_time(file_path):
    try:
        image = Image.open(file_path)
        exif_data = image._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                decoded = TAGS.get(tag, tag)
                if decoded == 'DateTimeOriginal':
                    return value
    except Exception as e:
        print(f"Error reading EXIF data from {file_path}: {e}")
    return None

def process_images(input_folder, output_folder):
    """
    Input arguments:
    - input_folder (Folder containing images to be cropped)
    - output_folder (Folder to save the cropped images)
    
    This function will process all images in the input folder and save the cropped images to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    
    # Sort image files by DateTimeOriginal EXIF metadata
    image_files.sort(key=lambda f: get_exif_date_time(os.path.join(input_folder, f)) or '')

    for unique_id, file_name in enumerate(image_files, start=1):
        file_path = os.path.join(input_folder, file_name)
        auto_crop(file_path, output_folder, unique_id)

if __name__ == '__main__':
    input_folder = 'D:\\Footage Storage\\FOOTAGE\\0. The Least Final Experiment Ever\\Dave McKeegan\\Sunspot Jpegs-20250103T123748Z-001\\Sunspot Jpegs'
    output_folder = 'D:\\Footage Storage\\FOOTAGE\\0. The Least Final Experiment Ever\\Dave McKeegan\\Sunspot Jpegs-20250103T123748Z-001\\Cropped sunspots metadata'
    process_images(input_folder, output_folder)


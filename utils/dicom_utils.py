import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers import convert_color_space
from PIL import Image
from utils.constants import *
import matplotlib.pyplot as plt
from pathlib import Path
import random
random.seed(17)

'''
Preprocess DICOM files
'''
def change_dicom_color(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    pixels = ds.pixel_array
    if ds.PhotometricInterpretation=='MONOCHROME2':
        pixels = np.stack((pixels,)*3,axis=-1)
    elif ds.PhotometricInterpretation in ['YRB_FULL','YBR_FULL_422']:
        pixels = convert_color_space(pixels,ds.PhotometricInterpretation,'RGB')
        if len(pixels.shape)<4:
            ecg_mask = np.logical_and(pixels[:,:,1]>200,pixels[:,:,0]<100)
            pixels[ecg_mask,:] = 0
    elif ds.PhotometricInterpretation=='RGB':
        if len(pixels.shape)<4:
            ecg_mask = np.logical_and(pixels[:,:,1]>200,pixels[:,:,0]<100)
            pixels[ecg_mask,:] = 0
    else:
        print('Unsupported photometric interpretation:',ds.PhotometricInterpretation)
    return pixels 

def convert_image_dicom(pixel_array,n=224):
    pixel_array = (pixel_array / np.max(pixel_array) * 255).astype(np.uint8)
    try:
        image = Image.fromarray(pixel_array)
    except: 
        print('Incompatible data type - ',pixel_array.shape)
        return 
    image = image.resize((n,n))
    image_tensor = torch.tensor(np.array(image),dtype=torch.float)
    image_tensor = image_tensor.permute(-1,0,1)
    image_tensor = image_tensor / 255. 
    return image_tensor

def convert_video_dicom(pixel_array,n=112):
    og_shape = pixel_array.shape
    h0,w0 = og_shape[1],og_shape[2]
    pixel_tensor = torch.from_numpy(pixel_array).float() # F,H,W,C
    pixel_tensor = pixel_tensor.permute(0,-1,1,2) # F,C,H,W
    resizer = torchvision.transforms.Resize((n,n))
    avi_tensor = resizer(pixel_tensor)
    avi_tensor = torch.tensor(avi_tensor).squeeze() # torch.tensor(resize_array).squeeze()
    return avi_tensor,h0,w0

def pull_first_frame(avi_tensor,n=224):
    resizer = torchvision.transforms.Resize((n,n))
    avi_tensor = resizer(avi_tensor)
    first_frame = avi_tensor[0,:,:,:]
    first_frame = first_frame.float()
    first_frame -= VIEW_MEAN.reshape((3,1,1))
    first_frame /= VIEW_STD.reshape((3,1,1))
    return first_frame

def pull_random_frame(avi_tensor,n=224):
    resizer = torchvision.transforms.Resize((n,n))
    avi = resizer(avi_tensor)
    n_frames = avi.shape[0]
    random_idx = random.randint(0,n_frames-1)
    frame = avi[random_idx,:,:,:]
    frame = frame.float()
    frame /= 255.
    return frame

'''
    Extract BSA from metadata
'''
def get_bsa(dcm_path): 
    ds = pydicom.dcmread(dcm_path)
    try: 
        height = ds[(0x0010,0x1020)].value # inches
        weight = ds[(0x0010,0x1030)].value # lbs
        if height > weight: # Switched entry of height and weight
            weight = ds[(0x0010,0x1020)].value
            height = ds[(0x0010,0x1030)].value
        kg = weight/2.2
        cm = height*2.54
        # 0.007184 × W^0.425 × H^0.725
        bsa = 0.007184*kg**0.425*cm**0.725
        return bsa
    except:
        ## Default 
        return 1.

'''
    Helper functions for Doppler measurements with EchoNet-Measurements
'''

def get_coordinates_from_dicom(
    dicom: pydicom.Dataset,
) -> tuple[tuple[int, int, int, int], tuple]:
    """
    Looks through ultrasound region tags in the DICOM file. Usually, 
    there are two regions, and the doppler image is the lower one. 
    Returns the coordinates of this region's bounding box.
    """

    REGION_COORD_SUBTAGS = [
        REGION_X0_SUBTAG,
        REGION_Y0_SUBTAG,
        REGION_X1_SUBTAG,
        REGION_Y1_SUBTAG,
    ]

    if ULTRASOUND_REGIONS_TAG in dicom:
        all_regions = dicom[ULTRASOUND_REGIONS_TAG].value
        regions_with_coords = []
        for region in all_regions:
            region_coords = []
            for coord_subtag in REGION_COORD_SUBTAGS:
                if coord_subtag in region:
                    region_coords.append(region[coord_subtag].value)
                else:
                    region_coords.append(None)

            # Keep only regions that have a full set of 4 coordinates
            if all([c is not None for c in region_coords]):
                regions_with_coords.append((region, region_coords))

        # We sort regions by their y0 coordinate, as the Doppler region we want should be the lowest region
        regions_with_coords = list(
            sorted(regions_with_coords, key=lambda x: x[1][1], reverse=True)
        )

        return regions_with_coords[0]

    else:
        print("No ultrasound regions found in DICOM file.")
        return None, None

def get_doppler_region(ds):
    doppler_region = get_coordinates_from_dicom(ds)[0]
    if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region:
        conversion_factor = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)
    if REGION_Y0_SUBTAG in doppler_region: 
        y0 = doppler_region[REGION_Y0_SUBTAG].value
    if REGION_Y1_SUBTAG in doppler_region: 
        y1 = doppler_region[REGION_Y1_SUBTAG].value
    if REGION_X0_SUBTAG in doppler_region: 
        x0 = doppler_region[REGION_X0_SUBTAG].value
    if REGION_X1_SUBTAG in doppler_region: 
        x1 = doppler_region[REGION_X1_SUBTAG].value
    if y0 <340 or y0 > 350:
        print("Error: Doppler Region is not located in the correct position. Please check the DICOM file. Our developed model is trained with y0 Doppler Region located in 342-348.")
        return -1,-1,-1,-1,-1
    return x0,x1,y0,y1,conversion_factor

def find_horizontal_line(
    image: np.ndarray,
    angle_threshold: float = np.pi / 180,
    line_threshold: float = 100,
):
    """
    Horizontal line detection for Doppler images.
    
    Uses Canny edge detection and the Hough Transform to find the most prominent horizontal line in the image. 
    Returns the y-coordinate of this line.
    """

    if len(image.shape) == 2: #Already gray image
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, line_threshold)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            if (
                abs(theta - np.pi / 2) < angle_threshold
                or abs(theta - 3 * np.pi / 2) < angle_threshold
            ):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                y = int(y0)
                return y
    return None

def calculate_weighted_centroids_with_meshgrid(logits):
    """
    #Write Explanation
    From Logit input, calculate the weighted centroids of the contours.
    If the number of objects is 3, the function returns the weighted centroids of the 3 objects.
    
    Args:
        logits: np.array of shape (H, W) with values in [0, 1]
    Returns:
        pair_centroids: list of tuples [(x1, y1), (x2, y2)]
        binary_image: np.array of shape (H, W) with values in {0, 255}
    
    """
    logits = (logits / logits.max()) * 255
    logits = logits.astype(np.uint8)
    _, binary_image = cv2.threshold(logits, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    centroids = []
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        h, w = mask.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        mask_indices = mask == 255
        filtered_logits = logits[mask_indices]
        x_coords_filtered = x_coords[mask_indices]
        y_coords_filtered = y_coords[mask_indices]
        weight_sum = filtered_logits.sum()
        if weight_sum != 0:
            cx = (x_coords_filtered * filtered_logits).sum() / weight_sum
            cy = (y_coords_filtered * filtered_logits).sum() / weight_sum
            centroids.append((int(cx), int(cy)))
    centroids = [(int(x), int(y)) for x, y in centroids]
    return centroids, binary_image

'''
    Plot and save results of EchoNet-Measurements for Doppler parameters
'''

def crop_image(img,y0=342):
    # (0,0) in top left corner
    cropped_img = img[y0:,:] # height 342 to end, full x-axis
    return cropped_img

def get_first_black_pixel(img):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for z in range(img.shape[2]):
                if img[x,y,z] == 0:
                    return x,y,z
                
def plot_results(m_name,filepath,peak_velocity,pred_x0,pred_y0,save_dir,pred_x1=None,pred_y1=None):
    pixels = change_dicom_color(filepath)
    first_black_pixel = get_first_black_pixel(pixels)
    pixels = crop_image(pixels,y0=first_black_pixel[0])
    plt.imshow(pixels)
    plt.scatter(pred_x0,pred_y0,color='royalblue')
    if (pred_x1) and (pred_y1):
        plt.scatter(pred_x1,pred_y1,color='tomato')
    plt.title(f'{m_name}: {peak_velocity}')
    save_path = Path(save_dir/f'{filepath.name}.png')
    plt.savefig(save_path,bbox_inches='tight',format='png')
    plt.close()
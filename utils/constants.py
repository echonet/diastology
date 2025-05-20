import torch
import numpy as np

### ------------ Constants ------------ ###
SEGMENTATION_THRESHOLD = 0.0
DO_SIGMOID = True
N_POINTS = 1
ULTRASOUND_REGIONS_TAG = (0x0018, 0x6011)
REGION_X0_SUBTAG = (0x0018, 0x6018)  # left
REGION_Y0_SUBTAG = (0x0018, 0x601A)  # top
REGION_X1_SUBTAG = (0x0018, 0x601C)  # right
REGION_Y1_SUBTAG = (0x0018, 0x601E)  # bottom
STUDY_DESCRIPTION_TAG = (0x0008, 0x1030)
SERIES_DESCRIPTION_TAG = (0x0008, 0x103E)
PHOTOMETRIC_INTERPRETATION_TAG = (0x0028, 0x0004)
REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)
### Normalization for LA segmentation model 
GLOBAL_LA_MEAN = np.array([33.492306,33.319996,33.252243])
GLOBAL_LA_STD = np.array([51.186474,51.035355,50.928997])
### Normalization for view classifier model 
VIEW_MEAN = torch.tensor([29.110628, 28.076836, 29.096405], dtype=torch.float32)
VIEW_STD = torch.tensor([47.989223, 46.456997, 47.20083], dtype=torch.float32)
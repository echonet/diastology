import argparse
import pandas as pd
import numpy as np 
import pydicom
from pathlib import Path
import tqdm
import sys
import os
import torch 
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

# from diastology import utils 
from utils import ase_guidelines,dicom_utils,model_utils,lav_mask


'''
    TO DO: 
        3. Comment code
        4. Consider more thorough error handling
'''

diastology_views = [
    'A4C','A4C_LV','A2C', # EF, LAVi 
    'Doppler_A4C_MV_PW', # MV E and MV E/A 
    'TDI_MV_Medial e', # MV medial e' velocity 
    'TDI_MV_Lateral e', # MV lateral e' velocity 
    'Doppler_A4C_TV_CW','Doppler_PSAX_Great_vessel_level_TV_CW' # TR Vmax
]

doppler_names = {
    'MEDEVEL':['TDI_MV_Medial e'],
    'LATEVEL':['TDI_MV_Lateral e'],
    'TRVMAX':['Doppler_A4C_TV_CW','Doppler_PSAX_Great_vessel_level_TV_CW']
}

parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,type=str,help='Path to directory of study with DICOM files')
parser.add_argument('--quality_threshold',required=False,type=float,help='Minimum echo quality')
args = parser.parse_args()
path = Path(args.path)
if args.quality_threshold:
    quality_threshold = args.quality_threshold
else:
    quality_threshold = 0.3 

if not path.exists():
    print(f'Path for study folder {path} does not exist. \nPlease provide a path to an existing directory')
else: 
    '''
        Pre-process DICOM into tensors
    '''
    files = [f for f in path.iterdir()]
    dataset = {}
    image_dataset = {}
    bsa = 0.
    for f in files:
        filename = f.stem
        dcm_path = path/f
        if bsa ==0:
            bsa = dicom_utils.get_bsa(dcm_path)
        pixels = dicom_utils.change_dicom_color(dcm_path)
        if len(pixels.shape)==4 and pixels.shape[0]>1: 
            x = dicom_utils.convert_video_dicom(pixels)
            x_first_frame = dicom_utils.pull_first_frame(x)
            image_dataset[filename] = x_first_frame
        else:
            x = dicom_utils.convert_image_dicom(pixels)
            image_dataset[filename] = x
        dataset[filename] = x
    '''
        View Classification
    '''
    print(f'Classifying views for {len(dataset)} files')
    view_input = torch.stack(list(image_dataset.values()))
    view_model = model_utils.load_view_classifier()
    filename = list(dataset.keys()) # Names of files
    predicted_view = model_utils.view_inference(view_input,view_model,filename)
    view_df = pd.DataFrame({'filename':list(predicted_view.keys()),'predicted_view':list(predicted_view.values())})
    ### Save predicted views
    view_df.to_csv('predicted_views.csv',index=None)
    print('Extracting views for diastology')
    to_remove = view_df[~view_df.predicted_view.isin(diastology_views)].filename
    for f in to_remove:
        dataset.pop(f)
        image_dataset.pop(f)
    view_df = view_df[view_df.predicted_view.isin(diastology_views)]

    '''
        Quality Control 
    '''
    print(f'Conducting quality control for {len(view_df)} files')
    quality_model = model_utils.load_quality_classifier()
    quality_input = torch.stack(list(image_dataset.values()))
    predicted_quality = model_utils.quality_inference(quality_input,quality_model)
    quality_df = pd.DataFrame({'filename':list(predicted_quality.keys()),
                                    'pred_quality':list(predicted_quality.values())})
    diastology = pd.merge(view_df,quality_df,on='filename')
    ### Save quality predictions
    diastology.to_csv('predicted_quality.csv',index=None)
    ### Remove low-quality echoes from tensor dataset and diastology dataframe
    diastology = diastology[diastology.pred_quality>=quality_threshold]
    low_qual = quality_df[quality_df.pred_quality<quality_threshold].filename
    for file in low_qual:
        dataset.pop(file)   
    print(f'Identified {len(diastology)} files with sufficient quality for diastology')

    '''
        LVEF Calculation
    '''
    print('Calculating LVEF')
    a4c = diastology[diastology.predicted_view.isin(['A4C','A4C_LV'])]
    lvef = []
    for filename in a4c.filename:
        a4c_tensor = dataset[filename]
        ef_model,ef_checkpoint = model_utils.ef_regressor()
        ef = model_utils.predict_lvef(a4c_tensor,ef_model,ef_checkpoint)
        lvef.append((ef,filename))
    lvef = pd.DataFrame(lvef,columns=['lvef','filename'])

    '''
        Left Atrial Volume Calculation
    '''
    la_model = model_utils.load_la_model()
    a2c = diastology[diastology.predicted_view=='A2C']
    if len(a2c)==0: 
        left_atrial_volume = []
        for filename in a4c.filename:
            try:
                a4c_tensor = dataset[filename]
                mask,area = model_utils.la_seg_inf(la_model,a4c_tensor)
                lav = model_utils.calc_lav_from_a4c(mask,area)
                left_atrial_volume.append(lav)
            except:
                left_atrial_volume.append(0)
        lav = np.max(left_atrial_volume)
    else: 
        a4c_areas = {} # key:value :: area : (np.array,float)
        a2c_areas = {} # key:value :: area : (np.array,float)
        for filename in a4c.filename:
            try:
                a4c_tensor = dataset[filename]
                a4c_mask,a4c_area = model_utils.la_seg_inf(la_model,a4c_tensor)
                a4c_areas[a4c_area] = (a4c_mask,a4c_area)
            except:
                continue
        for filename in a2c.filename:
            try:
                a2c_tensor = dataset[filename]
                a2c_mask,a2c_area = model_utils.la_seg_inf(la_model,a2c_tensor)
                a2c_areas[a2c_area] = (a2c_mask,a2c_area)
            except:
                continue
        a4c_key = max(list(a4c_areas.keys())) # Find A4C with maximum LA area
        a4c_mask_area = a4c_areas[a4c_key] # Find corresponding mask and area
        a2c_key = max(list(a2c_areas.keys()))
        a2c_mask_area = a2c_areas[a2c_key]
        try:
            lav = model_utils.calc_lav_biplane(la_model,a4c_mask_area[0],a4c_mask_area[1],a2c_mask_area[0],a2c_mask_area[1])
        except:
            lav = 0.
            print('Left atrial volume was not calculated')
    lavi = lav/bsa 
    
    '''
        Doppler Measurements
    '''
    print('Measuring Doppler parameters')
    doppler = [] # List to hold results  
    for i in range(len(doppler_names)): 
        doppler_view = list(doppler_names.values())[i]
        m_name = list(doppler_names.keys())[i]
        velocities = []
        m_parameter = m_name.lower()
        m_name_view = diastology[diastology.predicted_view.isin(doppler_view)]
        m_name_view['m_name'] = m_name
        if len(m_name_view)==0:
            ### Account for missing studies 
            doppler_filler = pd.DataFrame([0],columns=[m_name])
            doppler.append(doppler_filler)
            print(f'No views to measure {m_name[i].lower()} were found')
            continue
        #### Perform inference
        for filename in m_name_view.filename:
            dcm_path = Path(path/{filename})
            doppler_img,peak_velocity,pred_x,pred_y = model_utils.doppler_inference(dcm_path,m_parameter)
            velocities.append((filename,peak_velocity,pred_x,pred_y))
            save_dir = Path(path/f'{m_name}_results')
            if not save_dir.exists():
                os.mkdir(save_dir)
            dicom_utils.plot_results(m_name,dcm_path,peak_velocity,pred_x,pred_y,save_dir)
        velocities = pd.DataFrame(velocities,columns=['filename',m_name,'pred_x','pred_y'])
        doppler.append(velocities[m_name])
    doppler_measurements = pd.concat(doppler,axis=1)
    eovera = diastology[diastology.predicted_view=='Doppler_A4C_MV_PW']
    if len(eovera)==0: # Account for missing views
        print('No A4C PW Doppler of mitral valve to measure mitral E and A velocities were found')
        ea_vel = pd.DataFrame([(0,0)],columns=['evel','avel'])
    else:
        eovera['m_name'] = 'MVPEAKBOTH'
        ea_vel = []
        for filename in eovera.filename:
            path = path/{filename}
            eovera_image,y0,point_x1,point_x2,point_y1,point_y2,Inference_A_Vel,Inference_E_Vel,Inference_EperA = model_utils.eovera_inference(path)
            ea_vel.append((filename,Inference_E_Vel,Inference_A_Vel,y0,point_x1,point_x2,point_y1,point_y2))
            save_dir = Path(path/'mvEoverA_results')
            if not save_dir.exists():
                os.mkdir(save_dir)
            dicom_utils.plot_results(m_name,dcm_path,peak_velocity,point_x1,point_y1,save_dir,point_x2,point_y2)
        ea_vel = pd.DataFrame(ea_vel,columns=['filename','evel','avel','y0','x1','x2','y1','y2'])
    '''
        Grading diastology using 2016 ASE Guidelines
    '''    
    parameters = pd.concat([lvef['lvef'],doppler_measurements[list(m_name.keys())],ea_vel[['evel','avel']]],axis=1)
    parameters['lav'] = lav
    parameters['lavi'] = lavi
    try:
        lvef = np.mean(parameters.lvef)
    except:
        print('LVEF was not calculated. Diastology cannot be graded')
        sys.exit()
    try:
        medevel = np.mean(parameters['MEDEVEL'])
    except:
        print("Medial e' velocity was not measured")
    try:
        latevel = np.mean(parameters['LATEVEL'])
    except:
        print("Lateral e' velocity was not measured")
    try:
        trvmax = max(parameters['TRVMAX'])
    except:
        print("TR Vmax was not calculated. Assuming normal value")
    try:
        mvE = np.mean(parameters('evel'))
    except:
        mvE = 0.
        print('MV E velocity was not calculated')
    if mvE != 0:
        try:
            mvEoverA = mvE/np.mean(parameters['avel'])
            parameters['EoverA'] = mvEoverA
        except:
            parameters['EoverA'] = 0
            print("MV E/A was not calculated")
    else: 
        print("MV E/A was not calculated")
    mvE_eprime = ase_guidelines.calc_eeprime(mvE,latevel,medevel)
    parameters['E_eprime'] = mvE_eprime

    if lvef>=50:
        grade = ase_guidelines.preserved_ef_dd(medevel,latevel,trvmax,mvE_eprime,lavi) 
        if grade == 1: 
            grade = ase_guidelines.reduced_ef_dd(trvmax,mvE_eprime,mvEoverA,mvE,lavi) 
    else:
        grade = ase_guidelines.reduced_ef_dd(trvmax,mvE_eprime,mvEoverA,mvE,lavi) 

    print(f'Found {ase_guidelines.map_grade_to_text[grade]}.')
    parameters['diastology_grade'] = grade
    parameters.to_csv('diastology_parameters.csv',index=None)
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
import warnings
warnings.filterwarnings('ignore')

# from diastology import utils 
from utils import ase_guidelines,dicom_utils,model_utils,lav_mask


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
parser.add_argument('--guideline_year',required=True,type=int,help='Please indicate which year of ASE guideline to follow - 2016 vs 2025')
parser.add_argument('--quality_threshold',required=False,type=float,help='Minimum echo quality')
parser.add_argument('--to_save',required=False,type=bool,help='Option to save results')
parser.add_argument('--save_path',required=False,type=str,help='Path for saving diastology results')

args = parser.parse_args()
ase_year = args.guideline_year
path = Path(args.path)
if args.quality_threshold:
    quality_threshold = args.quality_threshold
else:
    quality_threshold = 0.
if args.to_save:
    save_flag = args.to_save
else:
    save_flag = True
if args.save_path: 
    save_path = Path(args.save_path)
else: 
    save_path = path/'diastology_results'
if not save_path.exists():
    os.mkdir(save_path)

if not path.exists():
    print(f'Path for study folder {path} does not exist. \nPlease provide a path to an existing directory')
else: 
    '''
        Pre-process DICOM into tensors
    '''
    files = [f for f in path.iterdir() if f.is_file()]
    dataset = {}
    image_dataset = {}
    video_dataset = {}
    bsa = 0.
    for f in files:
        # filename = f.stem
        dcm_path = Path(path/f)
        if bsa ==0:
            bsa = dicom_utils.get_bsa(dcm_path)
        pixels = dicom_utils.change_dicom_color(dcm_path)
        if len(pixels.shape)==4 and pixels.shape[0]>1: 
            x = dicom_utils.convert_video_dicom(pixels)
            x_first_frame = dicom_utils.pull_first_frame(x)
            video_dataset[f] = x 
            image_dataset[f] = x_first_frame
        else:
            x = dicom_utils.convert_image_dicom(pixels)
            image_dataset[f] = x
        dataset[f] = x

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
    if save_flag:
        view_df.to_csv(save_path/'predicted_views.csv',index=None)

    ### Extract views for deep learning calculation of diastology parameters
    print('Extracting views for diastology')
    to_remove = view_df[~view_df.predicted_view.isin(diastology_views)].filename
    for f in to_remove:
        dataset.pop(f)
        image_dataset.pop(f)
        try:
            video_dataset.pop(f)
        except: 
            continue 
    view_df = view_df[view_df.predicted_view.isin(diastology_views)]
    print('views:\n',view_df.predicted_view.unique())

    '''
        Quality Control for Images & Videos
    '''
    print(f'Conducting quality control for {len(view_df)} files')
    image_quality_model = model_utils.load_quality_classifier(input_type='image',
                                                              weights_path='/workspace/vic/hfpef/model_weights/quality/image_quality_classifier.pt'
                                                              )
    video_quality_model = model_utils.load_quality_classifier(input_type='video',
                                                              weights_path='/workspace/vic/hfpef/model_weights/quality/video_quality_classifier.pt'
                                                              )
    image_quality_input = torch.stack(list(image_dataset.values()))
    predicted_image_quality = model_utils.quality_inference(image_quality_input,image_quality_model,list(image_dataset.keys()))
    image_quality_df = pd.DataFrame({'filename':list(predicted_image_quality.keys()),
                                    'pred_quality':list(predicted_image_quality.values())
                                    })
    if len(list(video_dataset.values())) > 0:
        video_quality_input = torch.stack(list(video_dataset.values()))
        predicted_video_quality = model_utils.quality_inference(video_quality_input,video_quality_model,list(video_dataset.keys()))
        video_quality_df = pd.DataFrame({'filename':list(predicted_video_quality.keys()),
                                     'pred_quality':list(predicted_video_quality.values())
                                     })
        quality_df = pd.concat([image_quality_df,video_quality_df])
    else:
        quality_df = image_quality_df
    # predicted_quality = model_utils.quality_inference(quality_input,quality_model,list(view_df.filename))
    diastology = pd.merge(view_df,quality_df,on='filename')
    ### Save quality predictions
    if save_flag:
        diastology.to_csv(save_path/'predicted_quality.csv',index=None)

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
    if len(a4c)>0:
        for filename in a4c.filename:
            a4c_tensor = dataset[filename]
            ef_model,ef_checkpoint = model_utils.ef_regressor()
            ef = model_utils.predict_lvef(a4c_tensor,ef_model,ef_checkpoint)
            lvef.append((ef,filename))
        lvef = pd.DataFrame(lvef,columns=['LVEF','filename'])
    else:
        print('\tNo A4C found. Cannot calculate LVEF')
        lvef = pd.DataFrame({'LVEF':0,'filename':''})

    '''
        Left Atrial Volume Calculation
    '''
    la_model = model_utils.load_la_model()
    print('Calucating LAVi')
    a2c = diastology[diastology.predicted_view=='A2C']
    if len(a2c)==0 and len(a4c)==0: 
        print('\tNo A4C or A2C videos found. Cannot calculate LAVi')
        lav = 0
        lavi = 0
        df_lavi = pd.DataFrame({'filename':[''],'LAVi':[0],'LAV':[0],'BSA':[bsa]})
    elif len(a2c)==0 and len(a4c)>0: 
        print('\tNo A2C found. Using A4C only')
        left_atrial_volume = {}
        for filename in a4c.filename:
            try:
                a4c_tensor = dataset[filename]
                mask,area = model_utils.la_seg_inf(la_model,a4c_tensor)
                lav = model_utils.calc_lav_from_a4c(mask,area)
                left_atrial_volume[filename] = lav
            except:
                left_atrial_volume[filename] = 0.
                continue
        # lav = np.max(left_atrial_volume)
        a4c_key = max(list(left_atrial_volume.keys()))
        lav = left_atrial_volume[a4c_key]
        lavi = lav/bsa 
        df_lavi = pd.DataFrame({'filename':[a4c_key],'LAVi':[lavi],'LAV':[lav],'BSA':[bsa]})
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
            lav = model_utils.calc_lav_biplane(a4c_mask_area[0],a4c_mask_area[1],a2c_mask_area[0],a2c_mask_area[1])
            lavi = lav/bsa 
            df_lavi = pd.DataFrame({'filename':[a4c_key,a2c_key],'LAVi':[lavi],'LAV':[lav],'BSA':[bsa]})
        except:
            lav = 0.
            lavi = 0
            print('Left atrial volume was not calculated')
            df_lavi = pd.DataFrame({'filename':[''],'LAVi':[0],'LAV':[0],'BSA':[bsa]})
    
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
        for f in m_name_view.filename:
            dcm_path = Path(path/f)
            doppler_img,peak_velocity,pred_x,pred_y = model_utils.doppler_inference(dcm_path,m_parameter)
            velocities.append((f,peak_velocity,pred_x,pred_y))
            save_dir = Path(save_path/f'{m_name}_results')
            ### Save Doppler echo with predicted annotation
            if save_flag:
                if not save_dir.exists():
                    os.mkdir(save_dir)
                dicom_utils.plot_results(m_name,dcm_path,peak_velocity,pred_x,pred_y,save_dir)
        velocities = pd.DataFrame(velocities,columns=['filename',m_name,'pred_x','pred_y'])
        doppler.append(velocities[['filename',m_name]])
    doppler_measurements = pd.concat(doppler)
    eovera = diastology[diastology.predicted_view=='Doppler_A4C_MV_PW']
    if len(eovera)==0: # Account for missing views
        print('No A4C PW Doppler of mitral valve to measure mitral E and A velocities were found')
        ea_vel = pd.DataFrame({'filename':'','eovera':0,'evel':0,'avel':0})
    else:
        eovera['m_name'] = 'MVPEAKBOTH'
        ea_vel = []
        for f in eovera.filename:
            dcm_path = Path(path/f)
            eovera_image,y0,point_x1,point_x2,point_y1,point_y2,Inference_A_Vel,Inference_E_Vel,Inference_EperA = model_utils.eovera_inference(dcm_path)
            ea_vel.append((f,Inference_EperA,Inference_E_Vel,Inference_A_Vel,y0,point_x1,point_x2,point_y1,point_y2))
            save_dir = Path(save_path/'mvEoverA_results')
            ### Save Doppler echo with predicted annotation
            if save_flag:
                if not save_dir.exists():
                    os.mkdir(save_dir)
                dicom_utils.plot_results('MV E/A',dcm_path,Inference_EperA,point_x1,point_y1+y0,save_dir,point_x2,point_y2+y0)
        ea_vel = pd.DataFrame(ea_vel,columns=['filename','MV_E_over_A','MV_E','MV_A','y0','x1','x2','y1','y2'])

    '''
        Grading diastology using 2016 ASE Guidelines
    '''    
    print('Grading diastology by 2016 ASE Guidelines')
    parameters = pd.merge(diastology,lvef[['filename','LVEF']],on='filename',how='outer')
    parameters = pd.merge(parameters,doppler_measurements[['filename','MEDEVEL','LATEVEL','TRVMAX']],on='filename',how='outer')
    parameters = pd.merge(parameters,ea_vel[['filename','MV_E_over_A','MV_E','MV_A']],on='filename',how='outer')
    parameters = pd.merge(parameters,df_lavi,on='filename',how='outer')
    try:
        lvef = np.mean(parameters.LVEF)
        print('LVEF:\t\t\t%0.2f' % lvef)
    except:
        print('LVEF was not calculated. Diastology cannot be graded')
        sys.exit()
    print('LAVi:\t\t\t%0.2f' % lavi)
    try:
        medevel = np.mean(parameters['MEDEVEL'])
        print("Medial e' velocity:\t%0.2f" % medevel)
    except:
        print("Medial e' velocity was not measured")
    try:
        latevel = np.mean(parameters['LATEVEL'])
        print("Lateral e' velocity:\t%0.2f" % latevel)
    except:
        print("Lateral e' velocity was not measured")
    try:
        trvmax = np.max(parameters['TRVMAX'])
        print("TR Vmax:\t\t%0.2f" % trvmax)
    except:
        trvmax = 0.
        print("TR Vmax was not calculated. Assuming normal value")
    try:
        mvE = np.mean(parameters['MV_E'])
        print("Mitral E velocity:\t%0.2f" % mvE)
        mvE_eprime = ase_guidelines.calc_eeprime(mvE,latevel,medevel)
        print("Mitral E/e':\t\t%0.2f" % mvE_eprime)
    except:
        mvE = 0.
        mvE_eprime = 0.
        print('MV E velocity was not calculated')
    parameters['E_eprime'] = mvE_eprime

    try:
        mvEoverA = np.mean(parameters['MV_E_over_A'])
        print("MV E/A:\t\t\t%0.2f" % mvEoverA)
    except:
        mvEoverA = 0.
        print('MV E/A was not calculated')

    if ase_year==2016:
        if lvef>=50:
            grade = ase_guidelines.preserved_ef_dd(medevel,latevel,trvmax,mvE_eprime,lavi) 
            if grade == 1: 
                grade = ase_guidelines.reduced_ef_dd(trvmax,mvE_eprime,mvEoverA,mvE,lavi) 
        else:
            grade = ase_guidelines.reduced_ef_dd(trvmax,mvE_eprime,mvEoverA,mvE,lavi)
    elif ase_year==2025:
        grade = ase_guidelines.ase2025(medevel,latevel,trvmax,lavi,mvEoverA,mvE) 

    diastolic_grade = ase_guidelines.map_grade_to_text[grade]
    print(f'Found {diastolic_grade}.')
    parameters['diastology_grade'] = diastolic_grade
    save_diastology_path = save_path/'diastology.csv'
    if save_flag:
        parameters.to_csv(save_diastology_path,index=None)
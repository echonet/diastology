# Deep Learning Pipeline to Automate Diastology 
Left ventricular diastology dysfunction (LVDD) is an early precursor of cardiac pathology. Assessing LVDD is integral to the diagnosis and prognostics of both cardiac and extra-cardiac diseases, such as heart failure with preserved ejection fraction (HFpEF) and diabetes complicated by cardiomyopathy. LVDD is most commonly evaluated with echocardiography according to the 2016 guidelines outlined by the American Society of Echocardiography (ASE). The diagnostic algorithm integrates multiple modalities of echo and involves over 8 parameters. Because the algorithm is so extensive, characterizing LVDD can be laborious and time-intensive. 

This repository presents a workflow of 8 different EchoNet deep learning models to automate each step of the clinical workflow for diastology. Specifically, the pipeline: 

0. Processes DICOMs as input for deep learning 
1. View classifies each echo and extracts the necessary files for diastology
2. Removes low-quality echoes
3. Calculates parameters for diastology:
   - LVEF 
   - LAVi
   - TR Vmax
   - Medial e' velocity
   - Lateral e' velocity
   - Mitral E/A
4. Evaluates left ventricular diastolic function according to 2016 ASE guidelines
5. Saves results to file:
   - .csv with filenames, their corresponding deep learning echo parameter, and the final diastology grade
   - Doppler echoes with predicted annotation point and deep learning value

<p align='center'>
  <img width="900" alt="schematic_of_diastology_pipeline" src="https://github.com/user-attachments/assets/bea1ae52-a034-427e-a902-4635118d0295"/>
</p>

## How to Use
1. Download this repo

2. Download weights for:
   - View classification - https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip
   - Quality control
   - LVEF - https://github.com/echonet/dynamic/releases/tag/v1.0.0
   - Medial e' velocity - https://github.com/echonet/measurements/blob/main/weights/Doppler_models/medevel_weights.ckpt
   - Lateral e' velocity - https://github.com/echonet/measurements/blob/main/weights/Doppler_models/latevel_weights.ckpt
   - TR Vmax - https://github.com/echonet/measurements/blob/main/weights/Doppler_models/trvmax_weights.ckpt
   - Mitral E/A - https://github.com/echonet/measurements/blob/main/weights/Doppler_models/mvpeak_2c_weights.ckpt

3. Add the paths to the weights in utils/model_utils.py

4. Run inference to analyze diastology
   ```
   python main.py --path <path to echo study> --quality_threshold <minimum echo quality> --to_save <flag to save results> --save_path <directory where results will be saved>
   ```
   The script takes in the following arguments:
   1. --path - path to directory with the echo files; required parameter
   2. --quality_threshold - minimum threshold for echo quality, where the default is 0; optional parameter
   3. --to_save - True or False to indicate whether results should be saved, where the default is True; optional parameter
   4. --save_path - path to directory for saving pipeline results, where the default saves results to the input directory; optional parameter
  








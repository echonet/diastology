import numpy as np

map_grade_to_text = {
    0:'normal diastolic function',
    1:'grade I diastolic dysfunction',
    2:'grade II diastolic dysfunction',
    3:'grade III diastolic dysfunction',
    4:'indeterminate diastolic function',
    0.5:'indeterminate diastolic function',
    5:'indeterminate diastolic function',
    -1:'insufficient parameters for diastology'
}

def calc_eeprime(E=0,latevel=100,medevel=100):
    if E!=0:
        if (latevel!=100) & (medevel!=100):
            return E/(np.mean([latevel,medevel]))
        elif (latevel==100) & (medevel!=100):
            return E/medevel
        elif (latevel!=100) & (medevel==100):
            return E/latevel
    else:
        return 0.

def preserved_ef_dd(medevel=0,latevel=0,trvmax=0,E_evel=0,lavi=0):
    params = [medevel,latevel,trvmax,E_evel,lavi]
    n_criteria = float(len([p for p in params if p != 0]))
    abnormal = {}
    if params[0] != 0 and params[1] != 0: 
        n_criteria -= 1.
    if n_criteria < 3:
        return -1 
    if E_evel >14: 
        abnormal['Mitral E/e\''] = E_evel
    if medevel<7:
        abnormal['Mitral medial e\''] = medevel 
    elif latevel<10: # cm/s
        abnormal['Mitral lateral e\''] = latevel
    if trvmax >= 2.8: # Convert from cm/s to m/s
        abnormal['TR Vmax'] = trvmax
    if lavi>34:
        abnormal['LAVi'] = lavi
    if len(abnormal)/n_criteria == 0.5: 
        return 0.5 
    elif len(abnormal)/n_criteria > 0.5: 
        return 1    
    return 0

def reduced_ef_dd_subcriteria(E_evel=0,trvmax=0,lavi=0):
    positive_criteria = 0
    criteria = [E_evel,trvmax,lavi]
    if len([c for c in criteria if c>0])<=1:
        return 4
    if E_evel>14: 
        positive_criteria += 1
    if trvmax > 2.8: # Convert from cm/s to m/s
        positive_criteria += 1
    if lavi > 34: 
        positive_criteria += 1 
    return positive_criteria


def reduced_ef_dd(trvmax=0,E_evel=0,E_A=0,E=0,lavi=0):
    if E_A==0:
        return -1
    if E_A >= 2: 
        # Grade III LVDD
        return 3 
    if E_A <= 0.8: 
        if E <= 50: 
            return 1.1 
    if (E_A<=0.8 and E>50) or (E_A>0.8):  
        # 0.8 < E/A < 2. 
        subcriteria = [E_evel,trvmax,lavi]
        n_subcriteria = len([s for s in subcriteria if s>0]) #subcriteria.count(0)
        positive = reduced_ef_dd_subcriteria(E_evel,trvmax,lavi)
        if n_subcriteria == 3: # 3 criteria available
            if positive >= 2: 
                return 2
            elif positive <= 1: 
                return 1
        elif n_subcriteria == 2: # 2 criteria available
            if positive >= 2: 
                return 2 
            elif positive == 1: 
                return 4 # 4 for abnormal, but can't grade
            elif positive == 0: 
                return 5
        elif n_subcriteria <= 1: # Insufficient parameters to grade 
            return -1
    return 0

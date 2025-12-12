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

def ase2025(medevel,latevel,trvmax,lavi,eovera,e,
            pulmsd=1.0,lars=0.2,
            t_septal_e=6,t_lateral_e=7,t_avg_e=6.5,
            t_septalEe=15,t_lateralEe=13,t_avg_Ee=14,
            t_trvmax=280.,
            t_lavi=34,
            t_pulmsd=0.67,
            t_lars=0.18
            ):
    abnormal = []
    if medevel!=100 and latevel!=100:
        avg_evel = np.mean([medevel,latevel])
    else:
        avg_evel = 100
    if (medevel <= t_septal_e) or (latevel <= t_lateral_e) or (avg_evel <= t_avg_e):
        abnormal.append('reduced_e')
        # print('FOUND REDUCED E', medevel,latevel,avg_evel)
    if e!=0: 
        if medevel != 100 and latevel != 100:
            avg_Ee = e/np.mean([medevel,latevel])
            septal_Ee = e/medevel
            lateral_Ee = e/latevel
            if avg_Ee >= t_avg_Ee or septal_Ee >= t_septalEe or lateral_Ee >= t_lateralEe:
                abnormal.append('increased_Ee')
        if medevel != 100 and latevel == 100:
            septal_Ee = e/medevel 
            if septal_Ee >= t_septalEe: 
                abnormal.append('increased_Ee')
        if latevel != 100 and medevel == 100:
            lateral_Ee = e/latevel 
            if lateral_Ee >= t_lateralEe: 
                abnormal.append('increased_Ee')
    if trvmax >= t_trvmax: 
        abnormal.append('increased_TR')
    abnormal = set(abnormal)
    # print(abnormal)
    if len(abnormal) == 0:
        return 0 
    if len(abnormal) == 3:
        return 3 
    if len(abnormal) == 2 or (len(abnormal)==1 and 'increased_Ee' in abnormal) or (len(abnormal)==1 and 'increased_TR' in abnormal): 
        # print(len(abnormal),abnormal)
        ### If LAVi abnormal 
        if lavi > t_lavi or pulmsd <= t_pulmsd or lars <= t_lars: 
            ### Increased LAP, can't grade bc NO E/A
            if eovera == 0:
                return 5
            ### If E/A <2 --> Grade 2 
            elif eovera < 2: 
                return 2 
            ### If E/A >=2 --> Grade 3 
            elif eovera >= 2:
                return 3 
        ### If LAVi normal --> Grade 1
        elif lavi <= t_lavi: 
            return 1
    if len(abnormal) == 1: 
        if 'reduced_e' in abnormal: 
            ### Missing E/A so can't grade 
            if eovera==0.:
                return 4
            if eovera <= 0.8: 
                return 1 
            elif eovera > 0.8: 
                if lavi > 34: 
                    if eovera <2: 
                        return 2 
                    elif eovera >= 2:
                        return 3
                else:
                    return 1
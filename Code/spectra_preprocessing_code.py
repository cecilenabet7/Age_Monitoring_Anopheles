# Import librairies
import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from scipy import sparse
import os

### STEP 1 : Importing spectra from Maldi TOF Bruker folder

def tof2mass(tof, ML1, ML2, ML3):
    A = ML3
    B = np.sqrt(1E12/ML1)
    C = ML2 - tof
    if (A == 0): return ((C * C)/(B * B))
    else: return (((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A))**2)

def read_Bruker_spectrum(folder):
    '''
    Code to import Bruker spectrum and return raw mz and raw intensity data in Python
    Based on https://cran.r-project.org/web/packages/readBrukerFlexData/readBrukerFlexData.pdf
    Example of directory for input : MySpectrumFile_to_Analyse/.../.../0_H6/2/1SLin
    WARNING ! : need to end with '/1SLin' (1SLin folder need to contain 'acqu' and 'fid' files)
    Output return raw mz data and raw intensity data for one spectrum
    '''
    #folder=list_dir_to1SLin('C:\\Users\\...\\MySpectrumFile_to_Analyse') 
    #folder=list_dir_to1SLin('my_path_panel')
    files=os.listdir(folder)
    spectrumread=[]
    if 'acqu' not in files or 'fid' not in files:
        data=[[0]*2000,[0]*2000]
        spectrumread.append(data)
    else:
        try :
            parameters = open(folder  + "/acqu").read()
        except UnicodeDecodeError :
            parameters = open(folder  + "/acqu", encoding='latin-1').read()
        if parameters =='' or np.fromfile(folder  +'/fid', dtype = np.int32).size==0:
            data=[[0]*2000,[0]*2000]
            spectrumread.append(data)
        else:
            parse_var = parameters.find('$ML1= ')
            ML1 = float(parameters[parse_var + 6:parse_var + 20].split(' ')[0])
            parse_var = parameters.find('$ML2= ')
            ML2 = float(parameters[parse_var + 6:parse_var + 20].split(' ')[0])
            parse_var = parameters.find('$ML3= ')
            ML3 = float(parameters[parse_var + 6:parse_var + 20].split(' ')[0])
            parse_var = parameters.find('$DELAY= ')
            DELAY = int(parameters[parse_var + 8:parse_var + 22].split(' ')[0])
            parse_var = parameters.find('$DW= ')
            DW = float(parameters[parse_var + 5:parse_var + 19].split(' ')[0])
            parse_var = parameters.find('$TD= ')
            TD = int(parameters[parse_var + 5:parse_var + 19].split(' ')[0])
            parse_var = parameters.find('$AQ_DATE= ')
        #        DATE = parameters[parse_var + 11:parse_var + 21].split(' ')[0]
            raw_mz_scale = tof2mass(DELAY + np.arange(TD) * DW, ML1, ML2, ML3) #DELAY + np.arange(TD) * DW => to take only time of flight values
            raw_mz_scale2= raw_mz_scale.tolist()
            raw_intensite = np.zeros((len(files), TD), dtype = np.int32)
            raw_intensite = np.fromfile(folder  +'/fid', dtype = np.int32)
            raw_intensite2=raw_intensite.tolist()

    return raw_mz_scale2, raw_intensite2



### STEP 2 : Preprocessing phase

def baseline_als(y, lam = 10000, p = 0.01, niter=10):
    '''
    Baseline fitting
    Based on https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    See also https://pyspeckit.readthedocs.io/en/latest/baseline.html
    '''
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y) #z => noise 
        w = p * (y > z) + (1-p) * (y < z)
    noise = z
    return np.maximum(0, y-z), noise

def traitement_spectre(spectrum_read):
    '''
    Spectrum processing for retention of only the most representative positive peaks.
    '''
    
    #smoothing : moving average method
    N = 9
    spectrum_smooth = np.convolve(spectrum_read, np.ones((N,))/N, mode='valid')

    #baseline correction : asymmetric least squares method
    spectrum, noise_filtrated  = baseline_als(spectrum_smooth, lam = 10000, p = 0.01, niter=10)

    #soft peak picking : derivative method
    derivative = np.diff(spectrum)
    sign_change = np.where(np.sign(derivative[:-1]) != np.sign(derivative[1:]))[0] + 1 #We have a record of the positions where there is a change of sign.

    peak_detect = np.zeros(len(spectrum))
    peak_detect[sign_change] = spectrum[sign_change] #calibation_peak

    return peak_detect


### STEP 3 : Alignment step before injecting preprocessed spectra in models

''' See https://github.com/horvatovichlab/MSIWarp to follow the steps for aligning spectra'''

''' WARNING : If you are using machine learning (or Deep learning) models, 
be sure to separate the training and test data before alignment. 
Ideally, the spectra from the test set should be aligned with a spectrum from the training set 
(the spectrum from the training set with the highest correlation rate with the other spectra in the training set). 
Be careful with anatomical parts too! 
Make sure that you align the anatomical parts independently!'''
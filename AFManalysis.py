import numpy as np
import pandas as pd
from scipy import fftpack

class AFM(object):
    def __init__(self, path, scanSizeX = 256, scanSizeY = 256, size_um = 5):
    	'''
        path is file directory
        scanSizeX and scanSizeX are number of scans lines
        size_um is of square AFM image
        NOTE: AFM data must be in 1-D array as given by Nanoscope Analysis. 
        1-D array converted to a matrix; with analysis returning:
        depth histogram, cross section, RMS roughness, Fourier transform
        
        '''
        self.data = pd.read_csv(path, encoding = 'ANSI')
        self.matrix = np.array([self.data[name].values.tolist()[i:i + scanSizeX] 
        				for i in range(0,len(self.data),scanSizeY)])
        self.matrixc = self.correction()
        self.hist = np.histogram(self.matrixc,bins = np.linspace(-10,10,100),
        				density=True)[0]
        self.roughRMS = np.sqrt(np.mean(np.square(self.data.values))) 
        self.cross = self.matrixc[scanSizeY/2]
        self.fourier, self.fourierProfile = self.f_four(self.matrixc)
        self.thresh, self.threshRatio = self.threshold(self.matrixc)
        
    def correction(self):
    	'''
    	adjusts AFM height to be 0 nm at the (first) maximum in histogram
        '''
        maxHeight = np.linspace(-10,10,100)[np.where(self.hist == max(self.hist))] 
        return self.matrix - maxHeight[0]


    def f_four(self, matrix):
        '''
        returns 2-D Fourier transform and calculates the radial profile
        Creats fft, shifts to centre and abs values, 
        finds centre and distances from centre to sum fft
        '''
        FT = fftpack.fft2(matrix) 
        FTshift = abs(fftpack.fftshift(FT)) 
        centre = [FTshift.shape[0] / 2, FTshift.shape[1] / 2] 
        x, y = np.indices((FTshift.shape))
        r = np.sqrt((x - centre[0])**2 + (y - centre[1])**2).astype(np.int) 
        rprofile = np.bincount(r.ravel(),FTshift.ravel())/np.bincount(r.ravel())    
        return FTshift, rprofile

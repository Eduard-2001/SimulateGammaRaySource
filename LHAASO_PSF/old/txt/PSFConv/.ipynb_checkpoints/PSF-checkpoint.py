import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import interpolate
import tqdm


def csinterp(index):
    '''
    Args:
    1. index, an integer, 
        0 : 1.4.txt
        1 : 1.6.txt
        2 : 2.8.txt
        3 : 2.0.txt
        4 : 2.2.txt
        5 : 2.4.txt
        6 : 2.6.txt
        7 : 2.8.txt
        8 : 3.0.txt
        
    Returns:
    1. func, a function for convolution of psf
    '''
    Data = np.load('psf.npy')
    theta_2 = Data[index,:,0]
    dtheta = np.zeros_like(theta_2)
    dtheta[1:] = theta_2[1:]-theta_2[:-1]
    dtheta[0] = theta_2[0]
    theta=np.sqrt(theta_2)
    
    mc = Data[index,:,1]
    mc = mc/ (np.pi * dtheta)

    cubicsplinefunc = interpolate.CubicSpline(theta,mc)
    
    def func(src,xx,yy):
        '''
        Args: 
        1. src, sky map containing the source, without PSF
        2. xx,yy:  xx,yy = np.meshgrid(x,y)
        
        Returns: 
        1. the sky map after convolution with PSF
        '''
        summap = np.zeros(src.shape)
        r0 = np.sqrt(xx**2+yy**2)
        tmpmap = cubicsplinefunc(r0)
        tmpmap = np.where(r0 > theta[-1] , 0, tmpmap)
        norm = 1/tmpmap.sum()
        
        for i in tqdm.trange(src.shape[0]):
            for j in range(src.shape[1]):
                if src[i,j] == 0:
                    continue
                r = np.sqrt((xx-xx[0,j])**2 +(yy-yy[i,0])**2)
                tmpmap = cubicsplinefunc(r) *norm * src[i,j]
                tmpmap = np.where(r > theta[-1] , 0, tmpmap)
                summap += tmpmap
                
        return summap
    return func
from scipy.optimize import curve_fit
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
import tqdm


def gauss1d(x,sig,mu,amp):
        return amp * 1/(np.sqrt(2 * np.pi)*sig) * np.exp(-(x-mu)**2/2/sig**2)
    
    
def doublegauss(x,sig1,mu1,amp1,sig2,mu2,amp2):
    return gauss1d(x,sig1,mu1,amp1)+gauss1d(x,sig2,mu2,amp2)


def func(theta,sig1,mu1,amp1,sig2,mu2,amp2):
    result = np.zeros_like(theta)
    x = np.linspace(0,2,201)
    dx = x[1]-x[0]
    y = doublegauss(x,sig1,mu1,amp1,sig2,mu2,amp2)
    #plt.figure()
    #plt.plot(x,y)
    for i in range(theta.shape[0]):
        if i == 0:
            annulus_theta_idx = np.argwhere(x<theta[i])
            ph_sum = (y[annulus_theta_idx]*np.pi * x[annulus_theta_idx]*2).sum()*dx
        else:
            annulus_theta_idx = np.argwhere((x<theta[i])&(x > theta[i-1]))
            ph_sum = (y[annulus_theta_idx]*np.pi * x[annulus_theta_idx]*2).sum()*dx
        result[i] = ph_sum
    return result


def fitpsf(index,plot=False):
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
    density = mc/ (np.pi * dtheta)

    
    bestfit,pcov = curve_fit(func,theta,mc,bounds=([0.02,0,0,0.02,0,0],[0.2,0.0001,1e2,0.2,1,1e2]))
    
    if plot:
        x = np.linspace(0,2,201)
        print(bestfit)
        plt.figure()
        plt.ylim(0,1e3)
        plt.plot(x,doublegauss(x,*bestfit))
        plt.plot(x,gauss1d(x,*bestfit[0:3]))
        plt.plot(x,gauss1d(x,*bestfit[3:]))
        plt.scatter(theta,mc)
        
    
    
    def psfconv(src,xx,yy):
        '''
        Args: 
        1. src, sky map containing the source, without PSF
        2. xx,yy:  xx,yy = np.meshgrid(x,y)
        
        Returns: 
        1. the sky map after convolution with PSF
        '''
        summap = np.zeros(src.shape)
        r0 = np.sqrt(xx**2+yy**2)
        tmpmap = doublegauss(r0,*bestfit)
        norm = 1/tmpmap.sum()
        #print('norm %.2f'%norm)
        
        for i in tqdm.trange(src.shape[0]):
            for j in range(src.shape[1]):
                if src[i,j] == 0:
                    continue
                r = np.sqrt((xx-xx[0,j])**2 +(yy-yy[i,0])**2)
                tmpmap = doublegauss(r,*bestfit) * norm * src[i,j]
                #tmpmap = np.where(r > theta[-1] , 0, tmpmap)
                summap += tmpmap
                
        return summap
    return psfconv,bestfit

def bindata(sky,xx,yy,pw):
    '''
    Args: 
    1. sky, sky map being binned
    2. xx, yy
    3. pw, pixel width, in degree
    
    return :
    1. skybin: sky rebinned
    2. xxb,yyb : rebinned coordinate
    '''
    xmin = xx.min()
    xmax = xx.max()
    ymin = yy.min()
    ymax = yy.max()
    xbin = np.linspace(xmin,int((xmax-xmin)/pw)*pw-pw+xmin,int((xmax-xmin)/pw))
    ybin = np.linspace(ymin,int((ymax-ymin)/pw)*pw-pw+ymin,int((ymax-ymin)/pw))
    #print(xbin.shape)
    xxb,yyb = np.meshgrid(xbin,ybin)
    skybin = np.zeros_like(xxb)
    for i in tqdm.trange(skybin.shape[0]):
        for j in range(skybin.shape[1]):
            binpixel_idx = np.argwhere((xx <= xxb[i,j]+pw) & (xx  > xxb[i,j])&(yy <= yyb[i,j]+pw) & (yy > yyb[i,j]))
            
            #print(binpixel_idx.shape)
            skybin[i,j] = sky[binpixel_idx[:,0],binpixel_idx[:,1]].sum()
    return skybin,xxb,yyb,binpixel_idx

if __name__ == '__main__':
    psfconv,bestfit = fitpsf(1,plot=True)
    x = y = np.linspace(-1,1,2001)
    xx,yy = np.meshgrid(x,y)
    z = np.zeros_like(xx)
    z[500,700] = 1
    z[500,500] = 1
    blurz = psfconv(z,xx,yy)

    zbin,xxb,yyb,bp1 = bindata(z,xx,yy,0.2) 
    blurzbin,xxb,yyb,bp2 = bindata(blurz,xx,yy,0.2) 

    fig,ax = plt.subplots(2,2)
    ax[0][0].imshow(z)
    ax[0][1].imshow(blurz)
    ax[1][0].imshow(zbin)
    ax[1][1].imshow(blurzbin)
    plt.show()
    print(zbin.sum(),blurzbin.sum())

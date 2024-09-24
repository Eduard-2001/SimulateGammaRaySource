from scipy.optimize import curve_fit
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import interpolate
import tqdm
import emcee

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

    def gauss1d(x,sig,mu,amp):
        return amp * 1/(np.sqrt(2 * np.pi)*sig) * np.exp(-(x-mu)**2/2/sig**2)
    def doublegauss(x,params):
        sig1,sig2,mu2,amp1,amp2 = params
        return gauss1d(x,sig1,0,amp1)+gauss1d(x,sig2,mu2,amp2)

    def func(theta,params):
        sig1,sig2,mu2,amp1,amp2 = params
        result = np.zeros_like(theta)
        x = np.linspace(0,2,201)
        dx = x[1]-x[0]
        y = doublegauss(x,params)
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
    params = (0.06,0.1,0.26,600,300)
    #params = (0.1,0.1,0,0.2,10000,1000)
    res = func(theta,params)

    def log_prior(params):
        sig1,sig2,mu2,amp1,amp2 = params
        if sig1<0 or sig2<0 or amp1<0 or amp2<0 or mu2 < 0:
            return -np.inf
        return 0.0
    def log_likelihood(params,x,ytrue):
        model = func(x,params)
        log_p = (-(ytrue - model)**2/2).sum()
        return log_p
    def log_post(params,x,ytrue):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return log_likelihood(params,x,ytrue) + lp

    nwalkers,ndim = (10,5)
    pos = np.random.uniform(-0.1,0.1,(nwalkers,ndim))
    for i in range(pos.shape[0]):
        pos[i,:]+=params
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_post,args=(theta,mc))
    result = sampler.run_mcmc(pos,500,progress=True)
    sampler.reset()
    posnew = result[0]
    result = sampler.run_mcmc(posnew,2000,progress=True)


    '''fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()'''
    labels = ["sig1", "sig2","mu2","amp1","amp2"]
    '''
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");

    import corner
    '''
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    '''
    print(flat_samples.shape)
    fig = corner.corner(
        flat_samples, labels=labels
    );

    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        plt.plot(theta, func(theta,sample), "C1", alpha=0.1)
    plt.xlim(0,1)
    plt.scatter(theta, mc, label="truth")
    plt.legend(fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y");
    '''
    from IPython.display import display, Math
    bestfit = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        bestfit.append(mcmc[1])
        display(Math(txt))

    '''plt.scatter(theta,mc)
    plt.plot(theta,func(theta,bestfit))'''
    if plot:
        plt.figure()
        x=  np.linspace(0,2,1001)
        plt.xlim(0,2)
        sig1,sig2,mu2,amp1,amp2 = bestfit
        plt.plot(x,doublegauss(x,bestfit),lw = 1,c = 'black')
        plt.plot(x,gauss1d(x,sig1,0,amp1),lw = 0.5,c = 'red')
        plt.plot(x,gauss1d(x,sig2,mu2,amp2),lw = 0.5,c = 'blue')
        plt.scatter(theta,density,marker = 'x',c = 'black',s = 5)
        plt.title('photon density')
        plt.figure()
        plt.plot(theta,func(theta,bestfit))
        plt.scatter(theta,mc)
        plt.title('distribution-radius')
    
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
        tmpmap = doublegauss(r0,bestfit)
        norm = 1/tmpmap.sum()
        #print('norm %.2f'%norm)
        
        for i in tqdm.trange(src.shape[0]):
            for j in range(src.shape[1]):
                if src[i,j] == 0:
                    continue
                r = np.sqrt((xx-xx[0,j])**2 +(yy-yy[i,0])**2)
                tmpmap = doublegauss(r,bestfit) * norm * src[i,j]
                #tmpmap = np.where(r > theta[-1] , 0, tmpmap)
                summap += tmpmap
                
        return summap
    return psfconv

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
    psfconv = fitpsf(1,plot=True)
    x = y = np.linspace(-1,1,2001)
    xx,yy = np.meshgrid(x,y)
    z = np.zeros_like(xx)
    z[500,700] = 1
    z[500,500] = 1
    blurz = psfconv(z,xx,yy)

    zbin,xxb,yyb,bp1 = bindata(z,xx,yy,0.1) 
    blurzbin,xxb,yyb,bp2 = bindata(blurz,xx,yy,0.1) 
    fig,ax = plt.subplots(2,2)
    ax[0][0].imshow(z)
    ax[0][1].imshow(blurz)
    ax[1][0].imshow(zbin)
    ax[1][1].imshow(blurzbin)
    print(zbin.sum(),blurzbin.sum())

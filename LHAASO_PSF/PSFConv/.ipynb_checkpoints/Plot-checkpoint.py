import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

def hms2ra(h,m,s):
    '''
    convert hms(hour,minute,second) to degrees. 
    should not exceed 24h
    
    input: 
        hour, minute, second
    output:
        degree
    '''
    return h*15+m*15/60 + s*15/3600


def dms2dec(d,m,s):
    '''
    convert degree, arcmin, arcsec to decimal angle(deg)
    
    input:
        degree,arcmin,arcsec
    output:
        degree
    '''
    if d > 0:
        return d + m/60+s/3600
    elif d < 0:
        if m > 0:
            return d-m/60+s/3600
        else:
            return d-m/60-s/3600
    else:
        if m > 0:
            return m/60+s/3600
        elif m < 0:
            return m/60-s/3600
        else:
            return s/3600

        
def plotGala(coord='C'):
    '''
    plot galactic plane
    
    input:
        coord = 'C', coordinate of target plot
    output:
        none
    
    '''
    rot = hp.Rotator(coord=['G',coord])
    theta_g = np.ones(101)*np.pi/2
    phi_g = np.linspace(0,2*np.pi,101)
    theta_g_e,phi_g_e = rot(theta_g,phi_g)
    hp.projplot(theta_g_e,phi_g_e,'--',color='black')
    return


def selectreg(ma,ra,dec,radius,NSIDE,Equat=True,dpar=30,dmer = 30,cmap='jet',xsize=800,plot=True):
    '''
    select a region with one center and one radius, and convert it to cartesian coordinate.
    
    input:
        ma, map of all sky
        ra, dec: float or arraylike, in degree. 
        radius, the radius of ROI in degrees
        NSIDE, resolution of healpy grafic.
        Equat=True, coord of target plot
        dpar,dmer, density of reference line
        cmap='jet'
        xsize = 800, the final size of img
    output:
        selected image, 2darray
        '''

    vec = hp.ang2vec(ra,dec,lonlat=True)
    pix_select = hp.query_disc(NSIDE,vec,radius)
    region = ma[pix_select]
    lonra = [-radius,+radius]
    latra = [-radius,+radius]
    if Equat:
        coord = 'C'
    else:
        coord = 'G'
    img = hp.cartview(region,
                      coord=coord,
                      rot=(ra,dec),
                      lonra=lonra,
                      latra=latra,
                      return_projected_map=True,
                      cmap=cmap,
                      min = region.min(),
                      max = region.max(),
                      xsize=xsize,
                     )
    hp.graticule(dpar = dpar,dmer = dmer)
    plotGala(coord)
    if plot == False:
        plt.close('all')

    return img


def plotneutrinos(RA,DEC,ERR,E,NSIDE,cmap='jet',Equat = True):
    '''
    tmp func.
    '''
    if Equat:
        coord = 'C'
    else:
        coord = 'G'
    NPIX = hp.nside2npix(NSIDE)
    m = hp.ma(np.zeros(NPIX))
    for ra,dec,err,e in zip(RA,DEC,ERR,E):
        vec = hp.ang2vec(ra,dec,lonlat=True)
        selected_pix = hp.query_disc(NSIDE,vec,radius=np.radians(err/60))
        m[selected_pix] += np.log(e)
    hp.mollview(m,
               min = m.min(),
               max = m.max(),
               cmap=cmap,
               coord=coord,
               )
    hp.graticule()
    plotGala(coord)
    return m

def allskymap(RA,DEC,FLUX,NSIDE,cmap='jet',Equat = True,Plot=False):
    '''
    tmp func.
    '''
    if Equat:
        coord = 'C'
    else:
        coord = 'G'
    NPIX = hp.nside2npix(NSIDE)
    m = hp.ma(np.zeros(NPIX))
    for ra,dec,flux in zip(RA,DEC,FLUX):
        vec = hp.ang2vec(ra,dec,lonlat=True)
        pix = hp.vec2pix(NSIDE,*vec)
        m[pix] += flux
    if Plot:
        hp.mollview(m,
                   min = m.min(),
                   max = m.max(),
                   cmap=cmap,
                   coord=coord,
                   )
        hp.graticule()
        plotGala(coord)
    return m
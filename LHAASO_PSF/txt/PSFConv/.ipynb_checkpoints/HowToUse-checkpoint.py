#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import PSF

x = np.linspace(-1,1,101)
y = np.linspace(-1,1,101)
xx,yy = np.meshgrid(x,y)
z = np.zeros_like(xx)
z[10,20] += 100
z[10,32] += 120
noise = np.random.poisson(1,z.shape)
z += noise
psfconv = PSF.csinterp(1)
plt.imshow(z)
plt.colorbar()
plt.figure()
srcblur = psfconv(z,xx,yy)
plt.imshow(srcblur)
plt.colorbar()
plt.grid()
plt.show()
print(z.sum())
print(srcblur.sum())
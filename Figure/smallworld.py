#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:14:43 2018

@author: ksxu
"""
from __future__ import division

import matplotlib
matplotlib.use('Agg')
#import csv
import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal
import time as TM
import sys
import os

nsim = 20




# smal-world topology
deg = 3 # the degree of the nodes for neural networks
pij=0.5
CM = np.zeros((nsim,nsim))
CM_chem = np.zeros((nsim,nsim))
CM0 = np.zeros((nsim,nsim))
CM[0,-1]=1
for i in range(nsim):
    for d in range(1,deg+1):
        CM[i,i-d]=1
        CM_chem[i,i-d]=1
    if np.random.uniform()<pij:
        r1=np.random.randint(i-nsim+deg,i-deg)
        r2=np.random.randint(1,deg+1)
        CM[i,r1]=1
        if deg >= 2:
            CM[i,i-r2]=0

CM = CM+CM.T
CM_chem = CM_chem+CM_chem.T

DiffCM = CM-CM_chem
DiffCM[DiffCM == -1] = 0
CM0 = CM-DiffCM



CM = np.minimum(CM,np.ones_like(CM))
CM_chem = np.minimum(CM_chem,np.ones_like(CM_chem))
DiffCM = np.minimum(DiffCM,np.ones_like(DiffCM))
CM0 = np.minimum(CM0,np.ones_like(CM0))


CMl = np.tril(CM)
CMl_chem = np.tril(CM_chem)
DiffCM = np.tril(DiffCM)
CMl0 = np.tril(CM0)

cc0=np.array(np.where(CMl0==1)).T
cc=np.array(np.where(CMl==1)).T
cc_chem=np.array(np.where(CMl_chem==1)).T
cc_DiffCM=np.array(np.where(DiffCM==1)).T


N=20
p=0.1
R=0.9
np.random.seed(10)

angles=np.linspace(0,2*np.pi,num=N,endpoint=False)
xcoords=np.cos(angles)*R
ycoords=np.sin(angles)*R

colors=np.random.uniform(0.0,0.3,size=N)

fig=plt.figure(1,figsize=(8,8))
#ax1=plt.subplot(221)
#
#plt.scatter(xcoords,ycoords,s=80,marker='o',cmap='jet',c=colors,
#            linewidths=0,vmin=0,vmax=1,alpha=1,zorder=3)
#for c1,c2 in cc:
#    plt.plot((xcoords[c1],xcoords[c2]),(ycoords[c1],ycoords[c2]),'-',
#             color='k')
#
#ax2=plt.subplot(222)
#
#plt.scatter(xcoords,ycoords,s=80,marker='o',cmap='jet',c=colors,
#            linewidths=0,vmin=0,vmax=1,alpha=1,zorder=3)
#for c1,c2 in cc_chem:
#    plt.plot((xcoords[c1],xcoords[c2]),(ycoords[c1],ycoords[c2]),'-',
#             color='k')
    
    
#ax3=plt.subplot(223)
ax3=plt.subplot(111)

plt.scatter(xcoords,ycoords,s=160,marker='o',cmap='jet',c='blue',
            linewidths=0,vmin=0,vmax=1,alpha=1,zorder=3)
#for c1,c2 in cc_DiffCM:
for c1,c2 in cc:
    plt.plot((xcoords[c1],xcoords[c2]),(ycoords[c1],ycoords[c2]),'-',linewidth=6,
             color='r')
for c1,c2 in cc0:
    plt.plot((xcoords[c1],xcoords[c2]),(ycoords[c1],ycoords[c2]),'--',linewidth=3,
             color='K')

#ax4=plt.subplot(224)
#plt.title('gap junction')
#plt.scatter(xcoords,ycoords,s=80,marker='o',cmap='jet',c=colors,
#            linewidths=0,vmin=0,vmax=1,alpha=1,zorder=3)
#for c1,c2 in cc0:
#    plt.plot((xcoords[c1],xcoords[c2]),(ycoords[c1],ycoords[c2]),'-',
#             color='k')
plt.axis('off')

fig.tight_layout() 
plt.savefig('Smallworld.png',dpi = 300)
plt.savefig('Smallworld.eps',dpi = 1200)

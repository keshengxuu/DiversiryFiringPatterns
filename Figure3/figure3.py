#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:49:16 2019

@author: ksxu
"""

import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from numpy.ma import masked_array
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib as mpl        #Used for controlling color
from matplotlib import rc, cm
import matplotlib.colors as colors

#Fonts!
plt.rcParams['mathtext.sf'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

#changing the xticks and  yticks fontsize for all sunplots
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font',size=12)

#cm1 = mpl.colors.ListedColormap(['brg'])

Jee05 = 5*[0.5]
Jee25 = 5*[2.5]
JEI05 = [0,0.2,0.4,0.6,1.0]
JEI25 = [0,0.2,0.4,0.6,1.0]

#cm2 =mpl.colors.ListedColormap(['palegreen','green','brown','darkred'])


Gap00 = np.loadtxt('Gap00EISW_JeeVsJei_PSW001_data.txt',delimiter='\t')
Gap01 = np.loadtxt('Gap005EISW_JeeVsJei_PSW001_data.txt',delimiter='\t')


Gap10 = np.loadtxt('Gap10EISW_JeeVsJei_PSW001_data.txt',delimiter='\t')
Gap05 = np.loadtxt('Gap01EISW_JeeVsJei_PSW001_data.txt',delimiter='\t')


yN01 = np.shape(Gap01)[0]

def orderclassfy(gapdata):
    for i in range(yN01):
        if gapdata[i,1]>=0.50 and gapdata[i,3] >=0.90:
            if gapdata[i,2] >=0.90:
                gapdata[i,3] = -1  
        if gapdata[i,1]>=0.8:
            gapdata[i,3] = -1 
            
    return  gapdata
    


Gap01cla =  orderclassfy(Gap01)  
Gap05cla =  orderclassfy(Gap05)  
Gap10cla =  orderclassfy(Gap10)  
       

def shapedata2(trandata):
    yN = len(np.unique(trandata[:,0]))
    xN = len(np.unique(trandata[:,1]))
    XNmax = np.max(trandata[:,0])
    xNmin = np.min(trandata[:,0])
    yNmax = np.max(trandata[:,1])
    yNmin = np.min(trandata[:,1])
    
    SynIndex = np.reshape(trandata[:,-4],(yN,xN))
    SynIndex = np.transpose(SynIndex)
    
    OrderPara = np.reshape(trandata[:,-3],(yN,xN))
    OrderPara = np.transpose(OrderPara)
    
    
    Meanfre = np.reshape(trandata[:,-1],(yN,xN))
    Meanfre = np.transpose(Meanfre)
    
    Met = np.reshape(trandata[:,-2],(yN,xN))
    Met = np.transpose(Met)
    
    synch = [SynIndex,OrderPara,Met,Meanfre]
    extent = (xNmin,XNmax,yNmin,yNmax)
    return synch,extent


synchgap00,extentgap00 = shapedata2(Gap00)
synchgap01,extentgap01 = shapedata2(Gap01cla)

synchgap10,extentgap10 = shapedata2(Gap10cla)
synchgap05,extentgap05 = shapedata2(Gap05cla)




fig=plt.figure(1,figsize=(7,9))
plt.clf()
cmap =  plt.get_cmap('Greens')
#cmap =  plt.get_cmap('jet')
#cmap = plt.get_cmap('jet', 5)    # 5 discrete colors
#cmap1 = plt.get_cmap('jet', 5)    # 5 discrete colors

cmap0=plt.get_cmap('viridis_r')
cmap0.set_under('w',1)
norm=colors.Normalize(vmin=0,vmax=1)

cmap00=plt.get_cmap('viridis_r')
cmap30=plt.get_cmap('BuGn')

cmap1 =  plt.get_cmap('Greens')

ax1=plt.subplot(431)
im1= plt.imshow(synchgap00[0], origin = 'lower',extent=extentgap00, interpolation='nearest',cmap=cmap00,norm=norm,aspect='auto')
#plt.xlabel('$J_{EE}$',fontsize = 'x-large')
plt.ylabel(r'$\mathsf{J_{EI}}$',fontsize = 'x-large')


ax2=plt.subplot(434)
im2= plt.imshow(synchgap01[0], origin = 'lower',extent=extentgap00, interpolation='nearest',cmap=cmap00,norm=norm,aspect='auto')

ax3=plt.subplot(437)
im3= plt.imshow(synchgap05[0], origin = 'lower',extent=extentgap00, interpolation='nearest',cmap=cmap00,norm=norm,aspect='auto')


ax4=plt.subplot(4,3,10)
im4= plt.imshow(synchgap10[0], origin = 'lower',extent=extentgap00, interpolation='nearest',cmap=cmap00,norm=norm,aspect='auto')

# the colorbar setting 
#cbar = fig.colorbar(im4,ticks=[ 0.0, 0.2, 0.4,0.6,0.8, 1.0])
#cbar.set_ticks(np.linspace(0,1,3))



ax5=plt.subplot(432)
im5= plt.imshow(synchgap00[1],  origin = 'lower', extent=extentgap00,norm=norm, interpolation='nearest',
                cmap=cmap0, aspect='auto')
plt.plot(Jee05, JEI05,'r',lw = '2')
plt.plot(Jee25, JEI25,'tab:brown',lw = '2')
ax5.text(2.38,-0.09,'2.5')
ax5.text(0.38,-0.09,'0.5')

#plt.xlabel('$J_{EE}$',fontsize = 'x-large')
#plt.ylabel(r'$\mathsf{J_{EI}}$',fontsize = 'x-large')

ax6=plt.subplot(435)
#im6a= plt.imshow(synchgap01[1], origin = 'lower', extent=extentgap01,vmin =0, vmax = 1, interpolation='nearest',
#               cmap=plt.get_cmap('brg'), aspect='auto')
im6a = ax6.imshow(synchgap01[1], interpolation='nearest',cmap=cmap0,norm=norm,aspect='auto',origin='lower',extent=extentgap01)
                


ax7=plt.subplot(438)
im7= plt.imshow(synchgap05[1], origin = 'lower', extent=extentgap05,norm=norm, interpolation='nearest',
                cmap=cmap0, aspect='auto')

plt.plot(Jee05, JEI05,'r',lw = '2')
plt.plot(Jee25, JEI25,'tab:brown',lw = '2')
ax7.text(0.38,-0.09,'0.5')
ax7.text(2.38,-0.09,'2.5')

ax8=plt.subplot(4,3,11)
im8= plt.imshow(synchgap10[1], origin = 'lower', extent=extentgap10,cmap=cmap0,norm=norm, interpolation='nearest', aspect='auto')

# the colorbar setting 
#cbar = fig.colorbar(im1,ticks=[ 0.0, 0.2, 0.4,0.6,0.8, 1.0])
#cbar.set_ticks(np.linspace(0,1,3))

ax9=plt.subplot(433)
im9= plt.imshow(synchgap00[2],  origin = 'lower', extent=extentgap00,vmin =0, vmax = 0.06, interpolation='nearest',
                cmap=cmap30, aspect='auto')


#plt.xlabel('$J_{EE}$',fontsize = 'x-large')
#plt.ylabel(r'$\mathsf{J_{EI}}$',fontsize = 'x-large')


ax10=plt.subplot(4,3,6)
im10= plt.imshow(synchgap01[2],  origin = 'lower', extent=extentgap01,vmin =0, vmax = 0.06, interpolation='nearest',
                cmap=cmap30, aspect='auto')

ax11=plt.subplot(4,3,9)
im11= plt.imshow(synchgap05[2], origin = 'lower', extent=extentgap05,vmin =0, vmax = 0.06, interpolation='nearest',
                cmap=cmap30, aspect='auto')


ax12=plt.subplot(4,3,12)
im11= plt.imshow(synchgap10[2], origin = 'lower', extent=extentgap10,vmin =0, vmax = 0.06, interpolation='nearest',
                cmap=cmap30, aspect='auto')




#plt.xlabel('$J_{EE}$',fontsize = 'x-large')
#plt.ylabel(r'$\mathsf{J_{EI}}$',fontsize = 'x-large')


ax =[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]
for ax0 in ax:
#    ax0.set_xlim([0.15,extentgap1[1]])
#    ax0.set_ylim([0.035,extentgap1[3]])
    ax0.set_yticks(np.linspace(0,1.0,3))
#    ax0.set_yticks([0.01,0.02,0.03],['0.01','0.02','0.03'])


    
ax110 = [ax4,ax8,ax12]
for ax0 in ax110:
    ax0.set_xlabel(r'$\mathsf{J_{EE}}$',fontsize = 'x-large')
    
ax120 = [ax1,ax2,ax3,ax4]

for  ax0 in ax120:
    ax0.set_ylabel(r'$\mathsf{J_{EI}}$',fontsize = 'x-large')
    
ax_set = [ax1,ax2,ax3,ax4,ax9,ax10,ax11,ax12]
colorss = 'red'
for ax0 in ax_set:
    ax0.text(0.8,0.82,'d',fontsize='x-large',color = colorss)
    ax0.text(0.5,0.35,'c',fontsize='x-large',color = colorss)
    ax0.text(0.48,0.16,'b',fontsize='x-large',color = colorss)
    ax0.text(1.1,0.045,'a',fontsize='x-large',color = colorss)
    ax0.text(2.4,0.40,'e',fontsize='x-large',color = colorss)

#setting the colorbar for the Firing rate and MLE
cax1 = fig.add_axes([0.12, 0.952, 0.25, 0.01])
cbar1=fig.colorbar(im1, cax=cax1, orientation="horizontal")
#cbar1.set_label(' Firing Rate ',fontsize='medium')
cbar1.set_label(r'$\mathsf{\chi}$',fontsize='x-large',labelpad=-13, x=-0.07, rotation=0,color='red')
cbar1.set_ticks(np.linspace(0,1,3))
##change the appearance of ticks anf tick labbel
cbar1.ax.tick_params(labelsize='medium')
cax1.xaxis.set_ticks_position("top")



cax2 = fig.add_axes([0.43, 0.952, 0.22, 0.01])
cbar2=fig.colorbar(im6a,cax=cax2, orientation="horizontal")
#cbar1.set_label(' Firing Rate ',fontsize='medium')
cbar2.set_label(r'$\mathsf{R}$',fontsize='x-large',labelpad=-13, x=-0.07, rotation=0,color='red')
cbar2.set_ticks(np.linspace(0,1.0,3))
##change the appearance of ticks anf tick labbel
cbar2.ax.tick_params(labelsize='medium')
cax2.xaxis.set_ticks_position("top")


cax3 = fig.add_axes([0.735, 0.952, 0.21, 0.01])
cbar3=fig.colorbar(im10, extend='max', cax=cax3, orientation="horizontal")
#cbar1.set_label(' Firing Rate ',fontsize='medium')
cbar3.set_label(r'$\mathsf{Met}$',fontsize= 'x-large',labelpad=-13, x=-0.18, rotation=0,color='red')
cbar3.set_ticks(np.linspace(0,0.06,3))
##change the appearance of ticks anf tick labbel
cbar3.ax.tick_params(labelsize='medium')
cax3.xaxis.set_ticks_position("top")


axopt =[ax5,ax6,ax7,ax9,ax10,ax11]
axopt1 =[ax1,ax2,ax3]
axopt2 =[ax8,ax12]


for ax in axopt:
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    
for ax in axopt1:
    ax.set_xticks([])
#    ax.set_yticks([])
    
for ax in axopt2:
    ax.set_yticks([])
    

    
#plt.figtext(0.12,0.966,r'$\mathsf{J_{gap}}= 0.0$',fontsize = 'large')
#plt.figtext(0.36,0.966,r'$\mathsf{J_{gap}}= 0.1$',fontsize = 'large')
#plt.figtext(0.58,0.966,r'$\mathsf{J_{gap}}= 0.5$',fontsize = 'large')
#plt.figtext(0.8,0.966,r'$\mathsf{J_{gap}}= 1.0$',fontsize = 'large')
#ax1.set_yticks([0.01,0.5,0.99],['0.0','0.5','1.0'])



plt.figtext(0.01,0.93,r'$\mathsf{A}$',fontsize = 'x-large')
plt.figtext(0.01,0.7,r'$\mathsf{B}}$',fontsize = 'x-large')
plt.figtext(0.01,0.47,r'$\mathsf{C}$',fontsize = 'x-large')
plt.figtext(0.01,0.24,r'$\mathsf{D}$',fontsize = 'x-large')


S = [0.0,0.05,0.1,1.0]
axxx=[ax9, ax10, ax11, ax12]
    
for ax0, s0 in zip(axxx,S):
    ax0.text(3.05,0.5,'%s'%s0,fontsize='large')

ax9.text(3.03,0.9,r'$\mathsf{J_{gap}}}$',fontsize = 'x-large')
plt.subplots_adjust(bottom=0.08,left=0.10,wspace = 0.06,hspace = 0.1,right=0.94, top=0.943)

plt.savefig('Figure2.png',dpi = 150)
#plt.savefig('Figure2.eps',dpi = 600)

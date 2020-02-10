# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:20:45 2015

@author: ksxuu
"""
from __future__ import print_function
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
tEnd=50000
nsim=50


dt = 0.02
Total = 2400
equil_dt=int(400/dt)
Time = np.arange(0,Total,dt)
Time2=Time[equil_dt:]
Time3=Time[equil_dt::2]

#Fonts!
plt.rcParams['mathtext.sf'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

#changing the xticks and  yticks fontsize for all sunplots
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('font',size=12)

def read_csv_file(filename):
    """Reads a CSV file and return it as a list of rows."""
    data = []
    for row in csv.reader(open(filename)):
# transfer the string type to the float type
        row=list((float(x) for x in row))
        data.append(row)
    return data

# spikes
spikes_G00 = [read_csv_file('Gap00_Spikes/Spikes_%d.txt'%s ) for s in range(5) ]
spikes_G01 = [read_csv_file('Gap005_Spikes/Spikes_%d.txt'%s ) for s in range(5) ]
spikes_G05 = [read_csv_file('Gap01_Spikes/Spikes_%d.txt'%s ) for s in range(5) ]
spikes_G10 = [read_csv_file('Gap10_Spikes/Spikes_%d.txt'%s ) for s in range(5) ]

# amplitude of time series
AMP_G00 = [np.loadtxt('Gap00_AMP/AMP_%d.txt'%s ) for s in range(5) ]
AMP_G005 = [np.loadtxt('Gap005_AMP/AMP_%d.txt'%s ) for s in range(5) ]
#AMP_G01 = [np.loadtxt('Gap01_AMP/AMP_%d.txt'%s ) for s in range(5) ]
#AMP_G05 = [np.loadtxt('Gap05_AMP/AMP_%d.txt'%s ) for s in range(5) ]
#AMP_G10 = [np.loadtxt('Gap10_AMP/AMP_%d.txt'%s ) for s in range(5) ]

Jgap00 = np.loadtxt('met00.txt',delimiter='\t')
Jgap005 = np.loadtxt('met005.txt',delimiter='\t')
Jgap05 = np.loadtxt('met01.txt',delimiter='\t')
Jgap10 = np.loadtxt('met10.txt',delimiter='\t')

Xindex = np.linspace(0,4,5)

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0))  # outward by 10 points
#            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


xlims = [500,550]
xlims1 = [900,950]
ylims = [0,1250] 
ylims1 = [-75,40] 
dt = 0.02
N= 1250
fig=plt.figure(1,figsize=(11,18))
plt.clf()

# the setting of  the  figure layouts
ax1 = []
ax2 = []
ax3 = []
ax4 = []

gs1 = gridspec.GridSpec(nrows=2, ncols=5, left=0.06, right=0.955,top=0.96,bottom=0.69,
                        wspace=0.08,hspace=0.06)

gs2 = gridspec.GridSpec(nrows=2, ncols=5, left=0.06, right=0.955,top=0.64,bottom=0.38,
                        wspace=0.08,hspace=0.06)

gs3 = gridspec.GridSpec(nrows=2, ncols=5, left=0.06, right=0.955,top=0.33,bottom=0.04,
                        wspace=0.08,hspace=0.3)

#gs4 = gridspec.GridSpec(nrows=1, ncols=4, left=0.06, right=0.93,top=0.22,bottom=0.06,
#                        wspace=0.08,hspace=0.06)


for i in range(2):
    for j in range(5):
        ax1.append(fig.add_subplot(gs1[i, j]))
        
for i in range(2):
    for j in range(5):
        ax2.append(fig.add_subplot(gs2[i, j]))
        

for i in range(2):
    for j in range(5):
        ax3.append(fig.add_subplot(gs3[i, j]))
        
#for i in range(1):
#    for j in range(4):
#        ax4.append(fig.add_subplot(gs4[i, j]))
        
        
#plot the raster plot without gap junction
for ii, idx in zip(range(1,6), range(0,5)):
    
    for i in range(N):
        if i<1000:
            ax1[idx].plot(dt*np.array(spikes_G00[ii-1][i]),i*np.ones_like(spikes_G00[ii-1][i]),'r.',markersize=1)
        else:
            ax1[idx].plot(dt*np.array(spikes_G00[ii-1][i]),i*np.ones_like(spikes_G00[ii-1][i]),'g.',markersize=1)
            
        ax1[idx].set_xlim(xlims) 
        ax1[idx].set_ylim(ylims)
        
# the subplot for action potional
for ii, idx in zip(range(0,5), range(5,10)):
        ax1[idx].plot(Time2[25000:35000],AMP_G00[ii][:,:1000:10],lw=0.5)
       # ax1[idx].plot(Time2,AMP_G00[ii][:,:],lw=0.5)
        ax1[idx].set_xlim(xlims1) 
        ax1[idx].set_ylim(ylims1)
        


for i in range(5,6):
    ax3[i].plot((525,535),(-100,-100),'k',lw=2, clip_on = False)
    ax3[i].text(522,-300, '10 ms',fontsize = 'medium')
    

ax1[5].set_yticks(np.linspace(-60,30,4)) 



#plot the raster plot without gap junction equail 005
for ii, idx in zip(range(1,6), range(0,5)):
    
    for i in range(N):
        if i<1000:
            ax2[idx].plot(dt*np.array(spikes_G01[ii-1][i]),i*np.ones_like(spikes_G01[ii-1][i]),'r.',markersize=1)
        else:
            ax2[idx].plot(dt*np.array(spikes_G01[ii-1][i]),i*np.ones_like(spikes_G01[ii-1][i]),'g.',markersize=1)
            
        ax2[idx].set_xlim(xlims) 
        ax2[idx].set_ylim(ylims)  
        
# the subplot for action potional
for ii, idx in zip(range(0,5), range(5,10)):
        ax2[idx].plot(Time3[:],AMP_G005[ii][:,:200],lw=0.5)
       # ax1[idx].plot(Time2,AMP_G00[ii][:,:],lw=0.5)
        ax2[idx].set_xlim([520,570]) 
        ax2[idx].set_ylim(ylims1)        
ax2[6].set_xlim([540,590])       
# the subplots for gap=0.1
for ii, idx in zip(range(1,6), range(0,5)):
    
    for i in range(N):
        if i<1000:
            ax3[idx].plot(dt*np.array(spikes_G05[ii-1][i]),i*np.ones_like(spikes_G05[ii-1][i]),'r.',markersize=1)
        else:
            ax3[idx].plot(dt*np.array(spikes_G05[ii-1][i]),i*np.ones_like(spikes_G05[ii-1][i]),'g.',markersize=1)
            
        ax3[idx].set_xlim(xlims) 
        ax3[idx].set_ylim(ylims)  


# the subplots for gap = 1
for ii, idx in zip(range(1,6), range(5,10)):
    
    for i in range(N):
        if i<1000:
            ax3[idx].plot(dt*np.array(spikes_G10[ii-1][i]),i*np.ones_like(spikes_G10[ii-1][i]),'r.',markersize=1)
        else:
            ax3[idx].plot(dt*np.array(spikes_G10[ii-1][i]),i*np.ones_like(spikes_G10[ii-1][i]),'g.',markersize=1)
            
        ax3[idx].set_xlim(xlims) 
        ax3[idx].set_ylim(ylims) 
        
        
        
        
axset = [ax1,ax2,ax3]     
for ax in axset:
    if ax == ax1:
        for i in range(10):
            ax[i].tick_params(
            axis=u'both',          # changes apply to the x-axis
            which=u'both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,         # ticks along the left edge are off
            labelleft=False,    # labels along the lef edge are off
            labelbottom=False) # labels along the bottom edge are off
    elif ax == ax2:
        for i in range(10):
            ax[i].tick_params(
            axis=u'both',          # changes apply to the x-axis
            which=u'both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,         # ticks along the left edge are off
            labelleft=False,    # labels along the lef edge are off
            labelbottom=False) # labels along the bottom edge are off
    else:
        for i in range(10):
            ax[i].tick_params(
            axis=u'both',          # changes apply to the x-axis
            which=u'both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,         # ticks along the left edge are off
            labelleft=False,    # labels along the lef edge are off
            labelbottom=False) # labels along the bottom edge are off





#ax11.set_yticks([])        

#for i in  range(10,15):
#    ax2[i].set_xlabel('Times (ms)',fontsize ='large')
    
    
#for i in range(10,11):
#    ax2[i].plot((525,535),(-30,-30),'k',lw=2, clip_on = False)

#plt.figtext(0.14,0.013, '10 ms',fontsize = 'medium')   
    
    
axx = [ax1[0],ax2[0],ax3[0],ax3[5]]

S = [0.0,0.1,0.5,1.0]

for ax0 in axx:
    ax0.text(494,480,' Ex',color='red',rotation=90,fontsize='large')
    ax0.text(494,1100,' In',color='green',rotation=90,fontsize='large')     
    
stitle = ['a','b','c','d','e']
for i ,s in zip (range(5),stitle):
    ax1[i].text(505,1300,'%s'%s,fontsize ='xx-large')

for i, chi, met in zip(range(5), Jgap00[:,2], Jgap00[:,-2]):
    ax1[i].text(515,1300,'(%0.2f, %0.2f)'%(chi,met),fontsize ='large')

for i, chi, met in zip(range(5), Jgap005[:,2], Jgap005[:,-2]):
    ax2[i].text(515,1300,'(%0.2f, %0.2f)'%(chi,met),fontsize ='large')

for i, chi, met in zip(range(5), Jgap05[:,2], Jgap05[:,-2]):
    ax3[i].text(515,1300,'(%0.2f, %0.2f)'%(chi,met),fontsize ='large')

for i, chi, met in zip(range(5,10), Jgap10[:,2], Jgap10[:,-2]):
    ax3[i].text(515,1300,'(%0.2f, %0.2f)'%(chi,met),fontsize ='large')
     
# move the xlabel for action potential 
for i in range(10):
    adjust_spines(ax1[i], ['left'])
    ax1[i].axes.tick_params(direction="out")
    
for i in range(10):
    adjust_spines(ax2[i], ['left'])
    ax2[i].axes.tick_params(direction="out")
    
for i in range(10):
    adjust_spines(ax3[i], ['left'])
    ax3[i].axes.tick_params(direction="out")

idd0 = [6,7,8,9]
#idd1= [0,5,10,15]

for i in range(10):
    ax1[i].tick_params(
    axis='y',          # changes apply to the y-axis
    labelleft=False) 

for i in range(10):
    ax2[i].tick_params(
    axis='y',          # changes apply to the y-axis
    labelleft=False) 

for i in range(10):
    ax3[i].tick_params(
    axis='y',          # changes apply to the y-axis
    labelleft=False) 

plt.figtext(0.02,0.82,'V (mV)',rotation=90,fontsize = 'large')
plt.figtext(0.005,0.94,'A',fontsize = 'xx-large') 
plt.figtext(0.005,0.5,'B',fontsize = 'xx-large')
#plt.figtext(0.005,0.42,'C',fontsize = 'xx-large')
plt.figtext(0.005,0.18,'C',fontsize = 'xx-large')

#ax2[4].text(522,1025,r'$\mathsf{t_{1} }$',fontsize = 'x-large')
#ax2[4].text(535,1025,r'$\mathsf{t_{2} }$',fontsize = 'x-large')


plt.figtext(0.965,0.97,r'$\mathsf{J_{gap} }$',fontsize='large')

ax11 = [ax1[4],ax2[4],ax3[4],ax3[9]]

S = [0.0,0.05,0.1,1.0]
    
for ax0, s0 in zip(ax11,S):
    ax0.text(552,600,'%s'%s0,fontsize='large')


plt.subplots_adjust(bottom=0.1,left=0.10,wspace = 0.06,hspace = 0.1,right=0.94, top=0.943)

plt.savefig('Figure3.png',dpi = 600)
#plt.savefig('Figure3_rasterplot.eps')
#plt.savefig('Figure3_rasterplot.tif',dpi = 300)


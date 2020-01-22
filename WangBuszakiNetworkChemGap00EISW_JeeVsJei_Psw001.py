# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:34:11 2017

@author: Patricio 
@author:keshengXu
"""
import matplotlib
matplotlib.use('Agg')
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import Wavelets
from NetworkarchitectureB import Networkmodel
#from mpi4py import MPI

#rank=MPI.COMM_WORLD.rank
#threads=MPI.COMM_WORLD.size


rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1

def alphan(v):
    return -0.01*(v+34)/(np.exp(-0.1*(v+34))-1) # ok RH
def betan(v):
    return 0.125*np.exp(-(v+44)/80) # ok RH
def alpham(v):
    return -0.1*(v+35)/(np.exp(-0.1*(v+35))-1) # ok RH
def betam(v):
    return 4*np.exp(-(v+60)/18) # ok RH
def alphah(v):
    return 0.07*np.exp(-(v+58)/20) # ok RH
def betah(v):
    return 1/(np.exp(-0.1*(v+28))+1) # ok RH

def expnorm(tau1,tau2):
    if tau1>tau2:
        t2=tau2; t1=tau1
    else:
        t2=tau1; t1=tau2
    tpeak = t1*t2/(t1-t2)*np.log(t1/t2)
    return (np.exp(-tpeak/t1) - np.exp(-tpeak/t2))/(1/t2-1/t1)

def ISI_Phase(Num_Neurs,TIME,spikes):
     #==============================================================================
        #
        # calculationt the phase of neuron i based on the t_i(m), which is the mth spike of neuron i and t 
        #belong to [t_i(m),t_i(m+1)]
        #see Physical Review E 95,012308(2017)
    #==============================================================================
    Phase = np.zeros((TIME.size, Num_Neurs), dtype=np.float32)
    for ci in range (Num_Neurs): # Num_Neurs is total number of neurons
        mth = 0 #  the flag of mth spikes of neuron i at time t
        nt = 0
        for t in TIME:
            #if np.array(spikes[ci]).size ==0:
            if np.sum(spikes[ci]) < 2:
                Phase[nt, ci] = 0
            elif (mth+1) < np.array(spikes[ci]).size:
                Phase[nt, ci]= 2.0*np.pi*(t-spikes[ci][mth])/(spikes[ci][mth+1]-spikes[ci][mth])
                nt = nt+ 1
                if t > spikes[ci][mth+1]:
                    mth = mth +1
    return Phase



def WB_network(X,i):
    global firing
    v,h,n,sex,sey,six,siy,sexe,seye=X # agregué variable 's'
    minf=alpham(v)/(betam(v)+alpham(v)) # aqui estaba el error. corregido RH
    INa=gNa*minf**3*h*(v-ENa) # ok RH
    IK=gK*n**4*(v-EK) # ok RH
    IL=gL*(v-EL) # ok RH
    
    ISyn= (sey + seye) * (v - VsynE) + siy * (v - VsynI)
    Igap = np.sum(CMgapMat*(v[:,None] - v),axis =1) # the gap jucntion currents
    
    firingExt = np.random.binomial(1,iRate*dt,size=N)

    if any(i>delay_dt):
        firing=(V_t[i-delay_dt,range(N)]>theta)*(V_t[i-delay_dt-1,range(N)]<theta)

    return np.array([-INa-IK-IL-ISyn-Igap+Iapp,
                     phi*(alphah(v)*(1-h) - betah(v)*h),
                     phi*(alphan(v)*(1-n) - betan(v)*n),
                     -sex*(1/tau1E + 1/tau2E) - sey/(tau1E*tau2E) + np.dot(CMeMatrix,firing[0:Ne])+ np.dot(CMieMatrix,firing[0:Ne]),
                     sex,
                     -six*(1/tau1I + 1/tau2I) - siy/(tau1I*tau2I) + np.dot(CMiMatrix,firing[Ne:]) + np.dot(CMeiMatrix,firing[Ne:]),
                     six,
                     -sexe*(1/tau1E + 1/tau2E) - seye/(tau1E*tau2E) + firingExt*GsynExt,
                     sexe])  

equil= 400 #400
Trun=2000#2000    
Total=Trun + equil #ms

dt = 0.02 #ms
    # Neurons Parameters
gNa = 35.0; gK = 9.0;  gL=0.1  #mS/cm^2
ENa = 55.0; EK = -90.0; EL = -65.0 #mV
phi = 5.0

#Synaptic parameters
mGsynE = 5; mGsynI = 200; mGsynExt = 3  #mean
sGsynE = 1; sGsynI = 10; sGsynExt = 1
VsynE = 0; VsynI = -80  #reversal potential
tau1E = 3; tau2E = 1
tau1I = 4; tau2I = 1
Pe=0.3; Pi=0.3
iRate = 6#10*Pi
P_SW = 0.01

factE = 1000*dt*expnorm(tau1E,tau2E)
factI = 1000*dt*expnorm(tau1I,tau2I)
W_gap = 0 #0.001
W_gapi =0.001 #0.05

mdelay=1.5; sdelay = 0.1 #ms

Iapp = 0; # uA/cm^2, injected current

theta=0

Ne=1000 #Numero de neuronas excitatorias
Ni=250 #Numero de neuronas inhibitorias
N=Ne+Ni
#CM=np.zeros((N,N))
GsynE = np.random.normal(mGsynE,sGsynE,size=(N,Ne)) / factE
GsynExt = np.random.normal(mGsynExt,sGsynExt,size=N)
GsynExt = GsynExt*(GsynExt>0) / factE
GsynI = np.random.normal(mGsynI,sGsynI,size=(N,Ni)) / factI
delay = np.random.normal(mdelay,sdelay,size=N)


np.random.seed(10)
#Igap = np.sum(ConMatrix*(Vx[:,None] - Vx),axis =1) # the gap jucntion currents


firing=np.zeros(N) 

delay_dt=(delay/dt).astype(int)
equil_dt=int(equil/dt)
Time = np.arange(0,Total,dt)
nsteps=len(Time)
Time2=Time[equil_dt:]

decimate=50;cutfreq=100
b,a=signal.bessel(4,cutfreq*2*dt/1000,btype='low')

freqs=np.arange(10,100,0.5)  #Desired frequencies
Periods=1/(freqs*(decimate*dt)/1000)    #Desired periods in sample untis
dScales=Periods/Wavelets.Morlet.fourierwl  #desired Scales

binsize = 0.5 # bin size for population activity in ms
tbase = np.arange(equil, Total/dt, binsize/dt) # raster time base in time points

kernel=signal.gaussian(10*2/binsize+1,2/binsize)
kernel/=np.sum(kernel)
    
Jee = np.arange(0.0,3.01,0.1)
Jei=np.arange(0.0,1.01,0.025)
Vals=[(v1,v2) for v1 in Jee for v2 in Jei]

isim=0

for val in Vals:
    if isim%threads==rank:
        
        jee,jei = val
        
        net = Networkmodel(Ne1 =Ne,   # The numbers of excitatroy ensemble
              Ni1 = Ni,  # The numbers of inhibitory ensemble
              cp =P_SW,  # the probabilty to add new links
              W_gap =0.0, # the weight of gap junction for excitatoty population.
              W_Chem1 = jee, # the weight of  itself chemical synapse in excitatoty population.
              W_ei1 = jei, # the weight of inhibitory synapse currents from inhibitory to excitatory population.
              W_ii1 = 0.04, # the weight of  itself inhibitory currents in inhibitory population
              W_ie1 = 0.01, # the weight of excitatory currents in inhibitory population from excitatory population
              )
 
        wEE_gap, wEE_Chem, wEI_chem,wII_chem, wIE_chem= net.SmallWorld(deg=10,opt = 2)
        
        
        
         # the self-excitatory connections matrix  of the excitatory populations
        CMeMatrix = np.concatenate((wEE_Chem, np.zeros((Ni,Ne))),axis = 0)*GsynE
        # the inhiboptry connections from inhibitory populations to excitatory populatuions
        CMeiMatrix = np.concatenate((wEI_chem, np.zeros((Ni,Ni))),axis = 0) 
        #the itself-inhibitory connections matrix of the inhibitory populations
        CMiMatrix = np.concatenate((np.zeros((Ne,Ni)), wII_chem),axis = 0)*GsynI
        #the excitatory connection matrix from excitatory populations to inhibitory populations
        CMieMatrix = np.concatenate((np.zeros((Ne,Ne)), wIE_chem),axis = 0) 
        # the gap junction matrix 
        CM0= np.concatenate((wEE_gap, np.zeros((Ni,Ne))),axis = 0) 
        CMgapMat = np.concatenate((CM0,np.zeros((N,Ni))),axis = 1)
        
        #CMi=np.random.binomial(1,Pi,size=(N,Ni)) * GsynI
    
        V_t = np.zeros((nsteps,N))
        
        v_init=np.random.uniform(-80,-60,size=N) #-70.0 * np.ones(N) # -70 is the one used in brian simulation
        h=1/(1+betah(v_init)/alphah(v_init))
        n=1/(1+betan(v_init)/alphan(v_init))
        sex=np.zeros_like(v_init)
        sey=np.zeros_like(v_init)
        six=np.zeros_like(v_init)
        siy=np.zeros_like(v_init)
        sexe=np.zeros_like(v_init)
        seye=np.zeros_like(v_init)
        
        X=(v_init,h,n,sex,sey,six,siy,sexe,seye)
        
        print("starting simulation", isim," rank #", rank) 
        for i in range(nsteps):
            V_t[i]=X[0]    
            X+=dt*WB_network(X,i)

        print("ready simulation", isim," rank #", rank) 

        V_t=V_t[equil_dt:,:]
        
        #%%
        spikes=[(np.diff(1*(V_t[:,i]>theta))==1).nonzero()[0] for i in range(N)]
        pop_spikes = np.asarray([item for sublist in spikes for item in sublist]) # todas las spikes de la red
        popact,binedge = np.histogram(pop_spikes, tbase)
        
        conv_popact=np.convolve(popact,kernel,mode='same')
        
        LFP=np.mean(V_t[:,0:Ne],-1)
        sdLFP = np.std(LFP)
        sdV_t = np.std(V_t[:,0:Ne],0)
        chi_sq = Ne*sdLFP**2/(np.sum(sdV_t**2))
        
        # the phase of whole networks
        Phase = ISI_Phase(N,Time,spikes) 
        # the order parameter of the whole networks
        Rt = np.sqrt(np.mean(np.cos(Phase[:,0:Ne]),1)**2 + np.mean(np.sin(Phase[:,0:Ne]),1)**2) 
        RPamater = np.mean(Rt)
        Met = np.var(Rt)
        
        #%%
        xlims = [400,450]
        
        fig=plt.figure(1,figsize=(12,8), dpi= 80, facecolor='w', edgecolor='k') # tamaño, resolucion, color de fondo y borde de la figura
        plt.clf()
        
        plt.subplot(341)
        plt.plot(Time2,V_t[:,::5],lw=0.5)
        plt.ylim(-80,50 ) 
        
        plt.subplot(342)
        plt.plot(Time2,LFP)
        
        plt.subplot(343)
        for i in range(N):
            plt.plot(dt*np.array(spikes[i]),i*np.ones_like(spikes[i]),'.')
        
        plt.subplot(344)
        plt.plot(popact)
        plt.plot(conv_popact)
        
        plt.subplot(345)
        plt.plot(Time2,V_t[:,::5],lw=0.5)
        plt.xlim(xlims) 
        plt.ylim(-80,50 ) 
        
        plt.subplot(346)
        plt.plot(Time2,LFP)
        plt.xlim(xlims) 
        
        plt.subplot(347)
        for i in range(N):
            plt.plot(dt*np.array(spikes[i]),i*np.ones_like(spikes[i]),'.')
        plt.xlim(xlims) 
        
        plt.subplot(348)
        plt.plot(popact)
        plt.plot(conv_popact)
        plt.xlim(xlims) 
        
        #plt.subplot(3,4,9)
        #spect,freqs,t,_=plt.specgram(LFP,Fs=1000/dt,interpolation=None,
        #                             detrend='mean')
        #plt.ylim(20,80)
        
        plt.subplot(3,4,12)
        
        lags,c,_,_=plt.acorr(conv_popact,maxlags=100,usevlines=False,linestyle='-',ms=0)
        peaks=lags[np.where(np.diff(1*(np.diff(c)>0))==-1)[0]+1]
        
        if len(peaks)>1:
            netwFreq=1000/(binsize*peaks[peaks>0][0])
        else:
            netwFreq=np.nan
         
        Vfilt=signal.filtfilt(b,a,LFP)[::decimate]
        
       
        #wavel=Wavelets.Morlet(EEG,largestscale=10,notes=20,scaling='log')
        wavelT=Wavelets.Morlet(Vfilt,scales=dScales)
        #cwt=np.array([wavel.getdata() for wavel in wavelT])
        pwr=wavelT.getnormpower()
        
        plt.subplot(3,4,9)
        plt.specgram(Vfilt,Fs=1000/(dt*decimate),interpolation=None,
                                     detrend='mean')
        plt.ylim(0,100)
        
        plt.subplot(3,4,11)
        plt.plot(Vfilt)        


        plt.subplot(3,4,10)
        plt.imshow(pwr,aspect='auto',extent=(equil,Total,min(freqs),max(freqs)),origin='lower')

        plt.tight_layout()
        
        plt.savefig("Gap00EISW_JeeVsJei_PSW001_plots/Jee_%02d_Jei_%02d.png"%(jee*100,jei*1000),dpi=300)
        print("ready sim %g rank%g (%g,%g)"%(isim,rank,jee,jei))        

        with open("Gap00EISW_JeeVsJei_PSW001_data/results.txt_%04d"%isim,'a') as outfile:
            outfile.write("%g\t%g\t%g\t%g\t%g\t%g\n"%(jee,jei,chi_sq,RPamater,Met,netwFreq))
        
    isim+=1
        





#%% Characterization of network activity

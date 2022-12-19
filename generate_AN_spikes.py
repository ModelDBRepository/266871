#This script determines the release times from inner hair cells based on the release probabilities 
#and generates the spike times of each corresponding auditory nerves.
#The output is a (size(trials_tot)-by-6300) array of spike times (spikes_*dB_Lumax*_Lumin*_Lhmax*_Lhmin*.npy).

from __future__ import division
import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
from neuron import h
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
import os

spl=sys.argv[1]   #dB SPL
trials_tot=int(sys.argv[2])   #number of trials to average

#length ranges for Lu and Lh (um)
length_Lu_max=float(sys.argv[3])
length_Lu_min=float(sys.argv[4])
length_Lh_max=float(sys.argv[5])
length_Lh_min=float(sys.argv[6])

path = os.getcwd()

freq=10000        #Hz
tstim=5           #ms (length of sound stimulation)
num_ch=21         #channel number

#load timestep and release probabilities
dt=loadmat('timestep_'+str(freq)+'Hz.mat')['dt'][0][0]*1000
probs=loadmat('probs_'+str(spl)+'dB.mat')['ANprob']

#load auditory nerve fiber model
h.load_file(path+"/AuditoryFiberProperties.hoc")
h.node[1].insert('k')
h.node[1].insert('na')

fiber_per_ch=100 #number of fibers per channel

#start and end times of simulation (ms)
start = 10
stop = start+dt*probs.shape[1]
trange = np.arange(start,stop,dt)

h.tstop=stop

spikes_all=[]
for trial in range(trials_tot): 
	trial_no=trial
	np.random.seed(trial_no)
	
	#randomly assign Lu and Lh for all fibers
	lengths_Lu=np.random.uniform(length_Lu_min,length_Lu_max,num_ch*3*fiber_per_ch)
	lengths_Lh=np.random.uniform(length_Lh_min,length_Lh_max,num_ch*3*fiber_per_ch)

	#determine release times
	randoms_for_release=np.random.uniform(0,1,(3*num_ch*fiber_per_ch,np.array(probs).shape[1]))
	probs_all=np.repeat(probs,len(probs)*[fiber_per_ch],axis=0)
	release=randoms_for_release<probs_all

	#insert action potential counter at the heminode with a threshold of 10mV
	apc_hemi = h.APCount(0.5, sec=h.node[1])
	apc_hemi.thresh = 10.0
	apc_times_hemi = h.Vector()
	apc_hemi.record(apc_times_hemi)

	#initialize voltage and time vectors
	voltage = h.Vector()
	voltage.record(h.node[1](0.5)._ref_v)

	time = h.Vector()
	time.record(h._ref_t)

	#synaptic current from inner hair cells with rise and decay time constants 0.1 and 0.3 and reversal potential at 0 mV
	stim = h.Exp2Syn(0,sec = h.node[0])
	stim.tau1 = 0.1
	stim.tau2=0.3
	stim.e = 0

	firing_times = []
	for i in range(len(release)):

		#assign previously determined Lu and Lh to the auditory fiber
		h.node[0].L=lengths_Lu[i]
		h.node[1].L=lengths_Lh[i]
		
		#modify the channel conductance at the heminode according to Lh (assuming total number of channels is constant at the heminode)
		h.node[1].gmax_k=0.225/lengths_Lh[i]
		h.node[1].gmax_na=0.1812/lengths_Lh[i]

		#synaptic current at the release times
		release_times=trange[release[i]]
		vec = h.Vector(release_times)
		vs = h.VecStim()
		vs.play(vec)
		nc = h.NetCon(vs,stim)
		nc.delay = 0                #no delay at the synapse
		nc.weight[0]= 0.00012       #synaptic weight
		h.run()
		times=[]
		for k in apc_times_hemi:
			times.append(k)    #times at which auditory nerve spikes
		firing_times.append(times)

	np.asarray(firing_times)
	spikes_all.append(firing_times)

np.save(path+'/spikes_'+str(spl)+'dB_Lumax'+str(length_Lu_max)+'_Lumin'+str(length_Lu_min)+'_Lhmax'+str(length_Lh_max)+'_Lhmin'+str(length_Lh_min),spikes_all)

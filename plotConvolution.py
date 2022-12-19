# encoding=utf8

#This script takes the spike times of an auditory nerve population as an input and convolves the spikes with the unitary response of CAP.
#The output is a figure with simulated CAPs (plot.png)

from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
import os
from os import path
from matplotlib.patches import Rectangle
import sys
import pickle as pl
import matplotlib.font_manager

#Unitary response of CAP from Bourien et al., J Neurophysiol 112: 1025–1039, 2014. (K=1.14815 to simulate a 100uV CAP at 80dB SPL)
def func_conv(t):
    return 1.14815*0.14*np.exp(-1.440*t)*np.sin(2*np.pi*0.994*t)  

currentPath = os.getcwd()

#input parameters
spl=sys.argv[1]
trial_no=int(sys.argv[2])
length_Lu_max=float(sys.argv[3])
length_Lu_min=float(sys.argv[4])
length_Lh_max=float(sys.argv[5])
length_Lh_min=float(sys.argv[6])
color=sys.argv[7]

convs_all=np.empty((trial_no,22000))

#start and end times of the simulation
start = -0.503 #to shift the peak of the unitary response of CAP to 0 ms
stop = start + 20
tstep = loadmat('timestep_10000Hz.mat')['dt'][0][0]*1000
num_timestep = int((stop-start)/tstep)
t=np.linspace(start,stop,num_timestep)

#generate convolution function
func = np.array([])
conv_t=3 #damped oscillations are negligible after 3ms
t1=np.linspace(start,start+conv_t,int(conv_t/tstep))
for i in t1:
	func = np.append(func,[func_conv(i)])
x= round(optimize.fmin(lambda x: func_conv(x),0)[0],3)

for j in range(0,trial_no,1):
	spikes = np.load(currentPath+'/spikes_'+str(spl)+'dB_Lumax'+str(length_Lu_max)+'_Lumin'+str(length_Lu_min)+'_Lhmax'+str(length_Lh_max)+'_Lhmin'+str(length_Lh_min)+'.npy',allow_pickle=True)[j]
	spike_times = np.concatenate(spikes)
	spike_times = np.around(np.sort(spike_times,axis = None),decimals=3) #sorted spike times of a population
	unique_elements, counts_elements = np.unique(spike_times, return_counts=True) #unique_elements: each unique spike times ; count_elements: how many spikes in the population at each unique spike time
	spike_times=np.asarray((unique_elements, counts_elements))

	spike_array = np.array([])

	#generate an array (spike_array) of length "num_timestep" that shows number of spikes at each time step
	for i in range(len(spike_times[0])):
		if i == 0:
			spike_array = np.append(spike_array,np.zeros([1,int(round((spike_times[0][i]-x)/tstep,0))]))
			spike_array = np.append(spike_array,[spike_times[1][i]])

		else:
			spike_array = np.append(spike_array,np.zeros([1,int(round((spike_times[0][i]-spike_times[0][i-1])/tstep,0))-1]))
			spike_array = np.append(spike_array,[spike_times[1][i]])


	if len(spike_array)< num_timestep:
		spike_array = np.append(spike_array,np.zeros([1,num_timestep-len(spike_array)]))

	#convolve with the unitary response
	conv=np.convolve(spike_array,func)
	convs_all[j]=conv[:22000]

#calculate avg and standard error of CAPs
conv_avg=np.average(convs_all,axis=0)
conv_sterr=np.std(convs_all,axis=0)/np.sqrt(trial_no)

#Plot simulated CAPs
if path.exists("plot.pickle"):
	fig=pl.load(open('plot.pickle','rb'))
else:
	fig=plt.figure(figsize=(3.5,2))
	ax=fig.add_axes([0.13,0.16,0.83,0.8])
	ax.tick_params(right="off",top="off")
	ax.set_xticks([15,16,17])    
	ax.set_xticklabels([15,16,17])
	ax.set_yticks([0,-35,-70])    
	ax.set_yticklabels([0,-35,-70])

	plt.rcParams['mathtext.default'] = "regular"
	plt.xlabel('Time (ms)',fontsize=12,labelpad=0)
	plt.ylabel('Voltage ('+r'$\mu$'+'V)',fontsize=12,labelpad=2)
	plt.xlim(14.9,17.9)
	p=Rectangle((15,-75),5,5,facecolor='gray',edgecolor='None')
	ax.add_patch(p)
	plt.text(16.1,-74.5,'Sound stimulus',fontsize=9,color='w')
	plt.ylim(-75,1)
	plt.tick_params(labelsize=7)

t=np.arange(-0.503,22-0.503,0.001)
plt.plot(t,conv_avg,linewidth=2,color=color)
plt.fill_between(t,conv_avg-conv_sterr,conv_avg+conv_sterr,linewidth=0,alpha=0.3,facecolor=color)

pl.dump(fig,open('plot.pickle','wb'))
plt.savefig(currentPath+'/plot.png',dpi=300)

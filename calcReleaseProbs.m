% This script calculates release probabilities at each time step from each synapse in response to 5ms sound whose pressure level is defined by dB (in decibels) as an input. 
% The simulation starts and ends with 5ms long 0dB sound. 
% To calculate the probabilities, the model from "Steadman MA and Sumner CJ (2018) Changes in Neuronal Representations of Consonants in the Ascending Auditory System and Their Role in Speech Recognition. Front. Neurosci. 12:671. doi: 10.3389/fnins.2018.00671" is used. See https://zenodo.org/record/1345757#.X8aHLdNKhTY for the original version of the model implementation. 
% The output is a 63 (21 channels x 3 types)-by-15000 (number of timesteps) array (probs_*dB.mat)

function calcReleaseProbs(dB)

addpath (['.' filesep 'neural-representations-of-speech-master/Scripts/Auditory Nerve Model']);

paramnames={'GP_LSR','GP_MSR','GP_HSR'};  %low, medium and high spontaneous rate synapses (aka HT, MT and LT) 
BF=round(greenwood(21,5600,32000)); %define the frequencies of channels based on greenwood function

freq=10000;  %sound frequency in Hz

%define timestep based on the sound frequency
fs=freq*100;
dt=1/fs; 
dtname=strcat('timestep_',num2str(freq),'Hz.mat');
save(dtname,'dt')

%start the simulation with 5ms 0dB sound
initialdB=0;   %decibel
time1=dt:dt:0.005;
signal1=sum(sin(2*pi*freq'*time1), 1);
sig1=setleveldb(signal1,initialdB);

%5ms sound stimulus 
dur=0.005;    %duration of sound stimulus in seconds
time2=time1(end)+dt:dt:time1(end)+dur;
signal2=sum(sin(2*pi*freq'*time2), 1);
sig2=setleveldb(signal2,dB); 

%end the simulation with 5ms 0dB sound
time3=time2(end)+dt:dt:time2(end)+0.005;
signal3=sum(sin(2*pi*freq'*time3), 1);
sig3=setleveldb(signal3,initialdB);

%combine time and signal vectors
time=[time1,time2,time3];
sig=[sig1,sig2,sig3];

ANprob=zeros(length(BF)*length(paramnames),length(time));

%calculate release probabilities based on the sound stimulus and type of synapse (HSR, MSR, LSR)
for i=1:numel(paramnames)
  modeldata=runmodel_prob(sig,fs,BF,paramnames{i});
  j=length(BF)*i;
  ANprob(j-(length(BF)-1):j,:)=modeldata;
end

probname=strcat('probs_',num2str(dB),'dB.mat');
save(probname,'ANprob')

end

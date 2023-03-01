%This code will read edf files and will generate and save spectrograms of
%desired window length
clc
clear
%% Read the edf file : Specify the path in which you have stored data
pathtoData='/home/ambika/epilepsy/data/';
edfFilename=dir([pathtoData '*.edf']);
%% get the data from edf file
%This tells from which channel to read data
targetSignals=[1,2];
[hdr,chanData]=edfread(edfFilename(2).name,'targetSignals',targetSignals);
%% Set parameters to remove line noise Remove line noise
opts.NH=1; % No of harmonics of the line noise
opts.LF=50;% Line noise 
%opts.TOL = error tolerence. (default: 1% tol)
opts.HW = 2; %half-width of peak in samples.
%       M   = Size of window. (For fs = 1 kHz, m = 1024). If M is set, TOL
%                              has no effect.
%       WIN = {hanning,hamming}. Default: hanning
chanData_LNRemoved = removeLineNoise_SpectrumEstimation(chanData, hdr.frequency(1),'opts');
%% bandpass filter the signal in the cutoff 0.1Hz-100Hz
sig_filter_seizure=[];       
fn = 4;
Wn_low = 0.1/(hdr.frequency(1)/2);
Wn_high = 100/(hdr.frequency(1)/2);
[b,a] = butter(fn,[Wn_low Wn_high]); % band pass filter

for channo=1:size(chanData_LNRemoved,1)
    sig_filter_seizure(channo,:) = filtfilt(b,a,chanData_LNRemoved(channo,:));
end
%% Plot after line noise removal (Only 10 sec)

taxis = 0:1/hdr.frequency(1):(length(sig_filter_seizure(1,1:2500))-1)/hdr.frequency(1);
%% Test figure
figure(1)
ax1=subplot(2,1,1);
plot(taxis,sig_filter_baseline(1,5001:7500),'r')
title('baseline')
ylabel('Frequency (Hz)')
ax2=subplot(2,1,2);
plot(taxis,sig_filter_seizure(1,7001:9500))
ylabel('Frequency (Hz)')
title('seizure');
xlabel('Time(s)')
ylabel('Frequency (Hz)')
%% Divide the signal into one second chunks
IDX=chanData;
chunkSize=1; % Now I am taking chunkSize=1sec
vectorSizes=repmat(hdr.frequency(1),1,length(chanData)/(hdr.frequency(1)*chunkSize));
for channo=1:size(chanData,1)
    slicedData{channo}=mat2cell(chanData(channo,:),1,vectorSizes);
end

%% let's plot the spectrograms and save the images 
pltdir='/home/ambika/epilepsy/figures/';
for channo=2
    for jj=1:100
%        
        %j_specgram2(slicedData{channo}{jj},250,1) 
        spectrogram(slicedData{channo}{jj})
        colormap(plasma)
        pause
        savefig(gcf,[pltdir,'EPi_',num2str(jj),'.fig'])

        close all
    end
end
%%
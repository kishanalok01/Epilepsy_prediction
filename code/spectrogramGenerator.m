%This code will read edf files and will generate and save spectrograms of
%desired window length
clc
clear
chunkSize=1; % Select the size 
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
Wn_high = 60/(hdr.frequency(1)/2);
[b,a] = butter(fn,[Wn_low Wn_high]); % band pass filter

for channo=1:size(chanData_LNRemoved,1)
    sig_filter_seizure(channo,:) = filter(b,a,chanData_LNRemoved(channo,:));
end

%% Divide the signal into  chunks of desired seconds

vectorSizes=repmat(hdr.frequency(1)*chunkSize,1,floor(length(chanData)/(hdr.frequency(1)*chunkSize)));
%%
for channo=1:size(chanData,1)
    slicedData{channo}=mat2cell(sig_filter_seizure(channo,1:vectorSizes(1)*length(vectorSizes)),1,vectorSizes);
end

%% let's plot the spectrograms and save the images 
clc
pltdir='/home/ambika/epilepsy/figures/';
taxis = 0:1/hdr.frequency(1):(length(sig_filter_seizure(1,1:2500))-1)/hdr.frequency(1);
for channo=2
    for jj=333:335% length(slicedData{channo}{1})
%        
        j_specgram2(slicedData{channo}{jj},hdr.frequency(1),1) 
        % If you want to plot power spectrum
         plotpowerspectrum(slicedData{channo}{jj},hdr.frequency(1))
        %spectrogram(slicedData{channo}{jj})
        %colormap(plasma)
        pause % comment this line if you want to save all the figures without looking at them and annotate later
%         savefig(gcf,[pltdir,'EPi_',num2str(jj),'.fig'])

         close all
    end
end



%%
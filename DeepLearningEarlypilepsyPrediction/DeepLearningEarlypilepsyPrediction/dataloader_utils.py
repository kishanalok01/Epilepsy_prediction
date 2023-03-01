import os
import pandas as pd
import numpy as np
import h5py
import torch
import random
import eeghdf
from tqdm import tqdm

# Configuring the standard channels, frequency, and epoch length of the signals, as well as the label for seizure
SEIZURE_STRINGS = ['sz','seizure','absence','spasm']
FILTER_SZ_STRINGS = ['@sz','@seizure']

FREQUENCY = 200
INCLUDED_CHANNELS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4',
'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG Fz', 'EEG Cz', 'EEG Pz']

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

#from random import shuffle # used to shuffle initial file names
#from random import randint # used to randomly pull nonSeizures
from numpy.random import seed,choice,randint,shuffle


# Convert the global variable INCLUDED_CHANNELS,
# which is currently a list of strings, into a list of indices
# Returns: list of indices
def getChannelIndices(labelsObject):
#     labels = [l.decode("utf") for l in list(labelsObject)] # needed because might come as b strings
    labels = labelsObject
    channelIndices = [labels.index(ch) for ch in INCLUDED_CHANNELS]
    return channelIndices

# Check if a list of indices is sorted in ascending order.
# If not, we will have to convert it to a numpy array before slicing,
# which is a rather expensive operation
# Returns: bool
def is_increasing(channelIndices):
    last = channelIndices[0]
    for i in range(1, len(channelIndices)):
        if(channelIndices[i]<last):
            return False
        last = channelIndices[i]
    return True

# Swap select adjacenet channels
# Returns: list of tuples, each a pair of channel indices being swapped
def get_swap_pairs(channels):
    f12 = (channels.index('EEG Fp1'), channels.index('EEG Fp2'))
    f34 = (channels.index('EEG F3'), channels.index('EEG F4'))
    f78 = (channels.index('EEG F7'), channels.index('EEG F8'))
    c34 = (channels.index('EEG C3'), channels.index('EEG C4'))
    t34 = (channels.index('EEG T3'), channels.index('EEG T4'))
    t56 = (channels.index('EEG T5'), channels.index('EEG T6'))
    o12 = (channels.index('EEG O1'), channels.index('EEG O2'))
    return [f12, f34, f78, c34, t34, t56, o12]
   
# Augment the data by randomly deciding whether to swap some channel pairs,
# and independently, whether to slightly shrink the amplitude of the signals
# Returns: the processed (augmented or not) signals
def random_augmentation(signals):
#     print(signals.shape)
    for pair in get_swap_pairs(INCLUDED_CHANNELS):
        if(random.choice([True, False])):
            signals[:,[pair[0],pair[1]]] = signals[:,[pair[1], pair[0]]]
    if random.choice([True, False]):
        signals = signals * np.random.uniform(0.8,1.2)
    return signals

# Slice an epoch of EPOCH_LENGTH_SEC starting from sliceTime - offset from signals
# and only include channels represented in channelIndices
# Returns: a numpy array of shape (FREQUENCY * EPOCH_LENGTH_SEC, len(channelIndices))
def sliceEpoch(orderedChannels, signals, sliceTime, augmentation = True, clip_len = 10, normalization = False, offset=0):
    if (sliceTime == -1):
        maxStart = max(signals.shape[1] - FREQUENCY * clip_len, 0)
        #sliceTime = randint(0, maxStart)
        sliceTime = int(maxStart/2)
        sliceTime /= FREQUENCY
                
    startTime = int(FREQUENCY * max(0, sliceTime - offset))
    endTime = startTime + int(FREQUENCY * clip_len)


#     if(not is_increasing(orderedChannels)):  
    if(not is_increasing(orderedChannels)):  
        sliceMatrix = signals[:,startTime:endTime] 
        sliceMatrix = sliceMatrix[orderedChannels, :]
    else:
        sliceMatrix = (signals.s2u[orderedChannels] \
                       * signals.data[orderedChannels, startTime:endTime].T).T
        
#         sliceMatrix = signals[orderedChannels, startTime:endTime]
    # augment if requested. Default is True.
    if (augmentation):
        sliceMatrix = random_augmentation(sliceMatrix)
    
    # normalize by row if requested. Default is False.
    if (normalization):
        row_max = np.max(np.abs(np.int32(sliceMatrix)), axis = 1, keepdims = True)
        row_max = np.maximum(row_max, 1e-8) #for numerical stability
        sliceMatrix = sliceMatrix / row_max
        
    diff = FREQUENCY * clip_len - sliceMatrix.shape[1]
    ## padding zeros
    if diff > 0:
        zeros = np.zeros((sliceMatrix.shape[0], diff))
        sliceMatrix = np.concatenate((sliceMatrix, zeros), axis=1)
    sliceMatrix = sliceMatrix.T
    ## doing augmentation
    if(augmentation):
        sliceMatrix = random_augmentation(sliceMatrix)        
        
            
    return sliceMatrix

def getSeizureTimes(eegf):
    df = eegf.edf_annotations_df
#     antext = [s.decode('utf-8') for s in annot['texts'][:]]
#     starts100ns = [xx for xx in annot['starts_100ns'][:]]
#     df = pd.DataFrame(data=antext, columns=['text'])
#     df['starts100ns'] = starts100ns
#     df['starts_sec'] = df['starts100ns']/10**7
    seizureDF = df[df.text.str.contains('|'.join(SEIZURE_STRINGS),case=False)]
    seizureDF = seizureDF[seizureDF.text.str.contains('|'.join(FILTER_SZ_STRINGS),case=False)==False]

    seizureTimes = seizureDF['starts_sec'].tolist()
    return seizureTimes

def get_gold_ndxs(num_sz,cv_seed):
    seed(cv_seed)
    # want to return a random split of the Sz indeces for train/test split on gold experiment
    indices = list(range(num_sz))
    shuffle(indices)
    
    val_ndxs = indices[:int(num_sz/2)]    
    test_ndxs = indices[int(num_sz/2):]

    return (val_ndxs,test_ndxs)
import os,sys
import pandas as pd
import numpy as np
import h5py
import torch
import random

#sys.path.append('/home/ksaab/eeg_fully_supervised/model/')
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


from dataloader_utils import getChannelIndices, is_increasing, get_swap_pairs, random_augmentation, sliceEpoch, getSeizureTimes



# Create a balanced (50% seizure and 50% non-seizure) list of eeg files
# Returns: a list of two-element lists, with the first being the name of the file,
#          and the second being a binary indicator (1 for seizure and 0 otherwise)
def parseTxtFiles(split_type, seizure_file, nonseizure_file, cv_seed=0, scale_ratio=1,gold = False):

    np.random.seed(cv_seed)
    
    seizure_str = []
    nonseizure_str = []
    
    seizure_contents = open(seizure_file, "r")
    seizure_str.extend(seizure_contents.readlines())


    nonseizure_contents = open(nonseizure_file, "r")
    nonseizure_str.extend(nonseizure_contents.readlines())
    
    if gold:
        
        if split_type == 'train':
            
            # to make sure splitting is the same when function called multiple times
            np.random.seed(cv_seed) 
            # for train, use 80% of clips
            num_dataPoints = int(0.8*len(seizure_str))
            sz_ndxs_all = list(range(len(seizure_str)))
            np.random.shuffle(sz_ndxs_all)
            sz_ndxs_train = sz_ndxs_all[:num_dataPoints]
            np.random.shuffle(nonseizure_str)
            
            
            seizure_str = [seizure_str[i] for i in sz_ndxs_train]
            # doing an 80-20 split
            nonseizure_str = nonseizure_str[:4*num_dataPoints] 
            
        elif split_type == 'val':
            
            # to make sure splitting is the same when function called multiple times
            np.random.seed(cv_seed) 
            # for train, use 80% of clips
            num_dataPoints = int(0.8*len(seizure_str))
            sz_ndxs_all = list(range(len(seizure_str)))
            np.random.shuffle(sz_ndxs_all)
            sz_ndxs_val = sz_ndxs_all[num_dataPoints:]
            np.random.shuffle(nonseizure_str)
            
            seizure_str = [seizure_str[i] for i in sz_ndxs_val]
            # doing an 80-20 split, while making sure different than train
            nonseizure_str = nonseizure_str[4*num_dataPoints:4*num_dataPoints+4*len(seizure_str)]  
        
        
        
    else:

        # balanced dataset if train
        if split_type == 'train':

            #smaller_half = min(len(seizure_str), len(nonseizure_str))
            #combined_str = seizure_str[ : smaller_half] + nonseizure_str[ : smaller_half]
            num_dataPoints = int(scale_ratio*len(seizure_str))
            sz_ndxs_all = list(range(len(seizure_str)))
            np.random.shuffle(sz_ndxs_all)
            sz_ndxs = sz_ndxs_all[:num_dataPoints]
            seizure_str = [seizure_str[i] for i in sz_ndxs]
            np.random.shuffle(nonseizure_str)
            nonseizure_str = nonseizure_str[:num_dataPoints] 
            
            
        
    
    combined_str = seizure_str + nonseizure_str
    
    np.random.shuffle(combined_str)

    combined_tuples = []
    for i in range(len(combined_str)):
        tup = combined_str[i].strip("\n").split(",")
        tup[1] = float(tup[1])
        combined_tuples.append(tup)

    print_str = 'Number of clips in ' + split_type + ': ' + str(len(combined_tuples))
    print(print_str)
    
    return combined_tuples

class SeizureDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    
    def __init__(self, data_dir, file_dir, split_type, is_training = True, cv_seed=0, scale_ratio=1, clip_len = 12, gold=False):
        """
        Store the filenames of the seizures to use. 

        Args:
            data_dir: (string) directory containing the dataset
            file_dir: (string) directory containing the list of file names to pull in
            split_type: (string) whether train, val, or test set
        """
        
        
        
        if split_type == 'train':
            seizure_file = os.path.join(file_dir, 'sz_train.txt')
            nonSeizure_file = os.path.join(file_dir, 'non_sz_train.txt')
            if gold:
                seizure_file = os.path.join(file_dir, 'sz_dev.txt')
                nonSeizure_file = os.path.join(file_dir, 'non_sz_dev.txt')
        elif split_type == 'val':
            seizure_file = os.path.join(file_dir, 'sz_dev.txt')
            nonSeizure_file = os.path.join(file_dir, 'non_sz_dev.txt')
        elif split_type == 'test':
            seizure_file = os.path.join(file_dir, 'sz_test.txt')
            nonSeizure_file = os.path.join(file_dir, 'non_sz_test.txt')
        
        if clip_len == 60:
            self.offset = 15
        else:
            self.offset = 0
            
        self.file_tuples = parseTxtFiles(split_type,seizure_file, nonSeizure_file, cv_seed, scale_ratio,gold)
        self.is_training = is_training
        self.data_dir = data_dir
        self.file_dir = file_dir
        self.num_seiz_shaped = 0
        self.num_nonseiz_shaped = 0
        self.clip_len = clip_len
        self.split_type = split_type
        self.gold=gold

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, idx):
        """
        Fetch index idx seizure and label from dataset. 

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
#         file_name, seizure_idx = self.file_tuples[idx]
#         currentFileName = os.path.join(self.data_dir, file_name)
#         hdf = h5py.File(currentFileName)
#         orderedChannels = getOrderedChannels(hdf['record-0']['signal_labels'])


        file_name, seizure_idx = self.file_tuples[idx]
        currentFileName = os.path.join(self.data_dir, file_name)
        eegf = eeghdf.Eeghdf(currentFileName)
#         hdf = h5py.File(currentFileName)
        
#         orderedChannels = getOrderedChannels(hdf['record-0']['signal_labels'])
        
        orderedChannels = getChannelIndices(eegf.electrode_labels)
        phys_signals = eegf.phys_signals
        

        if seizure_idx == -1:
            
            nonSeizure = sliceEpoch(orderedChannels, phys_signals, -1, self.is_training, self.clip_len,offset=self.offset)
#             nonSeizure = sliceEpoch(orderedChannels, hdf['record-0']['signals'], -1, self.is_training)
            nonSeizure = torch.FloatTensor(nonSeizure)
            # for metal
            #return (nonSeizure, 2)
            return (nonSeizure, 0)
        else:
            if self.split_type == 'train' and not(self.gold):
                seizure_idx = int(seizure_idx)
                seizureTimes = getSeizureTimes(eegf)
                seizure = sliceEpoch(orderedChannels, phys_signals, seizureTimes[seizure_idx], self.is_training, self.clip_len,offset=self.offset)
            
            else:
                seizure_time = seizure_idx #seizure idx for dev/test represents starting time
                seizure = sliceEpoch(orderedChannels, phys_signals, seizure_time, self.is_training, self.clip_len,offset=self.offset)


            seizure = torch.FloatTensor(seizure)
            # for metal
            #return (seizure,1)
            return (seizure, 1)

def fetch_dataloader(types, data_dir, file_dir, use_gpu = True, batch_num = 10, cv_seed=0, scale_ratio=1, num_workers = 8, is_training = True, clip_length = 12,gold=False):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        file_dir: (string) directory containing the list of file names to pull in
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            if split in ['val','test']:
                batch_size = 1
            else:
                batch_size = batch_num
                
            dl = DataLoader(SeizureDataset(data_dir, file_dir, split, is_training, cv_seed, scale_ratio, clip_length,gold), \
                            batch_size=batch_size, shuffle=True,\
                                        num_workers=num_workers,\
                                        pin_memory=use_gpu)

            dataloaders[split] = dl

    return dataloaders

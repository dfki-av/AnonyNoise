import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import numpy as np 
import os 
from PIL import Image
from expelliarmus import Wizard
from numpy.lib.recfunctions import structured_to_unstructured
import torchvision.transforms.functional as fn
import  torchvision 
import tonic
import json
from collections import Counter
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DVSGesture(Dataset):
    classes = [
        "Hand clapping",
        "Right hand wave",
        "Left hand wave",
        "Right arm cw",  # clockwise
        "Right arm ccw",  # counter-clockwise
        "Left arm cw",
        "Left arm ccw",
        "Arm roll",
        "Air drums",
        "Air guitar",
        "Other gestures",
    ]

    sensor_size = (128, 128, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    def __init__(self, main_dir = './data/DVSGesture', mode = 'train', transform = None, target_transform = None, transforms= None, ):
        self.mode = mode
        self.main_dir = main_dir
        
        self.transform = transform
        if self.mode == 'train':
            self.folder_name = "ibmGestureTrain"
        else:
            self.folder_name = "ibmGestureTest"
            
        if self.mode in ['query', 'gallery']:
            self.query_ids = []
            query_ids_file = np.loadtxt("./reid_utils/dvsg_query_subset_ids.txt", delimiter=' ', dtype= str)
            for query in query_ids_file:
                self.query_ids.append(query[2])
                
        ev_repr_delta_ts_ms= 300
        
        self.target_ids = []
        self.user_ids = []
        self.samples = []
        self.file_ids = []
        file_path = os.path.join(self.main_dir, self.folder_name)
        
        time_diff = []
        for dir in os.scandir(file_path):
            for file in os.scandir(dir):
                if file.name.endswith(".npy"):
                    if self.mode == 'query' and file.path.replace(self.main_dir, '') not in self.query_ids:
                        continue
                    if self.mode == 'gallery' and file.path.replace(self.main_dir, '') in self.query_ids:
                        continue
                    events = np.load(file.path)
                    events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
                    to_frame = tonic.transforms.ToFrame(sensor_size = self.sensor_size, n_time_bins=5)
                    sample = torch.from_numpy(to_frame(events)).float()
                    self.samples.append(sample)
                    self.target_ids.append(int(file.name[:-4]))
                    self.user_ids.append(dir.name.split('_')[0])
                    self.file_ids.append(file.path) 
        self.label_map_id = {label: idx for idx, label in enumerate(sorted(set(self.user_ids)))}
        self.label_map_target = {label: idx for idx, label in enumerate(set(self.target_ids))}
        self.unique_labels_target = [ self.classes[int(v)] for v in self.label_map_target.values()]
        
    def __getitem__(self, idx):
        
        events = self.samples[idx] 
        
        user_id = self.label_map_id[self.user_ids[idx]]
        target = self.label_map_target[self.target_ids[idx]]
        file_id = self.file_ids[idx] 
        
        if self.transform:
            events = self.transform(events)
        events = events.reshape(events.size(0)*2, events.size(2), events.size(3))
        return events, target, user_id, file_id
    
    
    def __len__(self):
        return len(self.samples)
    
class EventReID_Dataset(Dataset):
    def __init__(self, main_dir="./data/EventReID_dataset", mode = 'train', transform=None):
        self.mode=mode
        self.transform = transform
        
        self.image_dir = []
        self.events = []
        self.events_anno = []
        self.sensor_size = (640,480, 2)
        self.dtype = np.dtype([("t", np.int64),("x", np.int16), ("y", np.int16), ("p", bool)])
        if self.mode in ['query', 'gallery']:
            self.query_ids = []
            query_ids_file = np.loadtxt(f"./reid_utils/eventreid_query_subset_ids.txt", delimiter=' ', dtype= str)
            for query in query_ids_file:
                self.query_ids.append(query)
                
        self.config = {
                "height": 720,
                "width": 1280,
                "T":5,
                "nbins": 1
                        }
      
        if self.mode == 'train':
            cur_dir = f'{main_dir}/train'
        else:
            cur_dir = f'{main_dir}/test'
            
        self.target_ids = []
        self.user_ids = []
        self.samples = []
        self.file_ids = []
        
        for file in os.listdir(cur_dir):
            if self.mode == 'query' and file not in self.query_ids:
                continue
            if self.mode == 'gallery' and file in self.query_ids:
                continue
            
            events = np.loadtxt(f'{cur_dir}/{file}', delimiter=' ', dtype=np.float64,  usecols=(0, 1, 2, 3))
            if len(events) == 0:
                continue
            events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
            self.samples.append(events)
            
            f = file.split('_')
            user_id = f[0]
            img_filename = f'{f[0]}_{f[1]}'
         
            self.target_ids.append(f[1])
            self.user_ids.append(user_id)
            self.file_ids.append(img_filename) 
            
        self.label_map_id = {label: idx for idx, label in enumerate(sorted(set(self.user_ids)))}
        self.label_map_target = {label: idx for idx, label in enumerate(sorted(set(self.target_ids)))}
        self.unique_labels_target = list(set(self.label_map_target.values()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        events = self.samples[idx] 
        
        user_id = self.label_map_id[self.user_ids[idx]]
        target = self.label_map_target[self.target_ids[idx]]
        file_id = self.file_ids[idx] 
        events['x'] = events['x'] - min(events['x']) 
        events['y'] = events['y'] - min(events['y']) 
        
        to_frame = tonic.transforms.ToFrame(sensor_size = (max(events['x'])+1, max(events['y'])+1, 2), n_time_bins=5)
        events = torch.from_numpy(to_frame(events)).float()

        events = self.pad_to_square(events)
        
        if self.transform:
            events = self.transform(events)
        events = events.reshape(events.size(0)*2, events.size(2), events.size(3))
       
        return events, target, user_id, file_id
    
    def normalize_voxel(self, events):
        non_zero_elements = events[events != 0]

        if non_zero_elements.numel() > 0:
            mean_value = non_zero_elements.mean()
            std_value = non_zero_elements.std()
            events[events != 0] = (events[events != 0] - mean_value) / std_value

        return events

    def pad_to_square(self, image_tensor):
        n, p, height, width = image_tensor.shape
        max_size = max(height, width)
        pad_height = max_size - height
        pad_width = max_size - width

        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        padded_image = torch.nn.functional.pad(image_tensor, (left_pad, right_pad, top_pad, bottom_pad))

        return padded_image
    

class SEEDataset(Dataset):

    def __init__(self, main_dir = '../SEE2', mode = 'train', transform = None, target_transform = None, transforms= None, ):
        self.mode = mode
        self.main_dir = main_dir
        self.transform = transform
        tensor_transform = torchvision.transforms.ToTensor()
        
        metadata_path = './reid_utils/see_metadata.json'
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        
        self.target_ids = []
        self.user_ids = []
        self.samples = []
        self.file_ids = []
        
        n_bins = 5
        
        for data in metadata[self.mode]:
            self.file_ids.append(data['filename']) 
            self.target_ids.append(data['target_id'])
            self.user_ids.append(data['user_id'])
            
            folderpath = f"{main_dir}/event/{data['target_id']}/{data['filename']}"
            filepaths = os.listdir(folderpath)
            events = []
            for file in filepaths:
                events.append(tensor_transform(Image.open(f'{folderpath}/{file}')))
            events = torch.stack(events, dim=0)
            mask_1 = (events == 1).float()
            mask_0 = (events == 0).float()
            events = torch.cat((mask_1, mask_0), dim=1)
            events_len = (events.shape[0]//n_bins)*n_bins
            events = events[:events_len].view(n_bins, -1, 2, events.shape[-2], events.shape[-1]).sum(dim=1).view(-1, 2, events.shape[-2], events.shape[-1]).float()
            
            self.samples.append(events)
            
        self.label_map_id = {label: idx for idx, label in enumerate(sorted(set(self.user_ids)))}
        self.label_map_target = {label: idx for idx, label in enumerate(sorted(set(self.target_ids)))}
        
        self.unique_labels_target = list(set(self.label_map_target.keys()))
        
    def __getitem__(self, idx):
        events = self.samples[idx] 
        
        user_id = self.label_map_id[self.user_ids[idx]]
        target = self.label_map_target[self.target_ids[idx]]
        file_id = self.file_ids[idx] 
        
        if self.transform:
            events = self.transform(events)
        events = events.reshape(events.size(0)*2, events.size(2), events.size(3))
        
        return events, target, user_id, file_id
    
    
    def __len__(self):
        return len(self.samples)

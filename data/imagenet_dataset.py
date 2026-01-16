# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import os
import torchvision.transforms as transforms

SENSOR_H = 480
SENSOR_W = 640
IMAGE_H = 224
IMAGE_W = 224
TIME_SCALE = 1000000


class ImageNetDataset(Dataset):
    def __init__(self, cfg):
        super(ImageNetDataset, self).__init__()
        self.mode = cfg["phase"]
        root = cfg["root"]        
        self.file = [os.path.join(root, i.strip()) for i in open(cfg["file"], 'r').readlines()]
        self.label = sorted(os.listdir(cfg["label_map"]))

        assert len(self.label) == 1000
        self.cfg = cfg
        self.augment_type = cfg.get('augment_type', None)
   
        
    
    def load_event(self, event_path, cfg):
        # Returns time-shifted numpy array event from event_path
        event = np.load(event_path)
        if cfg.get('compressed', True):
            event = event['event_data']
            event = np.vstack([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T
        else:
            event = np.vstack([event['x_pos'], event['y_pos'], event['timestamp'], event['polarity'].astype(np.uint8)]).T

        event = event.astype(np.float32)

        # Account for zero polarity
        if event[:, 3].min() >= -0.5:
            event[:, 3][event[:, 3] <= 0.5] = -1

        return event
    
    def slice_event(self, event, cfg):
        slice_method = cfg.get('slice_method', 'idx')
        if slice_method == 'idx':
            start = cfg.get('slice_start', None)
            end = cfg.get('slice_end', None)
            event = event[start:end]
        elif slice_method == 'time':
            start = cfg.get('slice_start', None)
            end = cfg.get('slice_end', None)
            event = event[(event[:, 2] > start) & (event[:, 2] < end)]
        elif slice_method == 'random':
            length = cfg.get('slice_length', None)
            slice_augment = cfg.get('slice_augment', False)

            if slice_augment and cfg["phase"] == 'train':
                slice_augment_width = cfg.get('slice_augment_width', 0)
                length = random.randint(length - slice_augment_width, length + slice_augment_width)

            if len(event) > length:
                start = random.choice(range(len(event) - length + 1))
                event = event[start: start + length]

        return event
    
    def reshape_event_no_sample(self, event, orig_h, orig_w, new_h, new_w):
        event[:, 0] *= (new_w / orig_w)
        event[:, 1] *= (new_h / orig_h)

        return event
    
    def parse_event(self, event_path, cfg):
        event = self.load_event(event_path, cfg)

        event = torch.from_numpy(event)

        # Account for slicing
        slice_events = cfg.get('slice_events', False)

        if slice_events:
            event = self.slice_event(event, cfg)

        reshape = cfg.get('reshape', False)
        if reshape:
            reshape_method = cfg.get('reshape_method', 'no_sample')

            if reshape_method == 'no_sample':
                event = self.reshape_event_no_sample(event, SENSOR_H, SENSOR_W, IMAGE_H, IMAGE_W)
            else:
                raise SystemExit

        return event
            
    
    def add_correlated_events(self, event, xy_std = 1.5, ts_std = 0.001, add_noise=0):
        if event.size(0) < 1000:
            return event
        to_add = np.random.randint(min(100, event.size(0)-1),min(5000,event.size(0)))
        event_new = torch.cat((
            event[:,[0]] + torch.normal(0, xy_std,size = (event.size(0),1)),
            event[:,[1]] + torch.normal(0, xy_std,size = (event.size(0),1)),
            event[:,[2]] + torch.normal(0, ts_std,size = (event.size(0),1)),
            event[:,[3]]
            ),-1)

        idx = np.random.choice(np.arange(event_new.size(0)), size=to_add, replace=False)
        event_new = event_new[idx]
        event_new[:,[0]] = torch.clip(event_new[:,[0]],0,event[:,[0]].max()) 
        event_new[:,[1]] = torch.clip(event_new[:,[1]],0,event[:,[1]].max())

        event = torch.cat((event,event_new))
        return event[event[:,2].argsort(descending = False)]  
    
    def random_time_flip(self, event_tensor, resolution=(224, 224), p=0.5):
        if np.random.random() < p:
            event_tensor[:, 3] = - event_tensor[:, 3]  # Inversion in time means inversion in polarity
        return event_tensor
    
    def random_shift_events(self, event_tensor, max_shift=20, resolution=(224, 224)):
        H, W = resolution
        x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
        event_tensor[:, 0] += x_shift
        event_tensor[:, 1] += y_shift

        valid_events = (event_tensor[:, 0] >= 0) & (event_tensor[:, 0] < W) & (event_tensor[:, 1] >= 0) & (event_tensor[:, 1] < H)
        event_tensor = event_tensor[valid_events]

        return event_tensor

    
    def reshape_then_acc_count_pol(self, event_tensor, augment_mode):
        # Accumulate events to create a 2 * H * W image

        if augment_mode == 'train':
            event_tensor = self.random_time_flip(event_tensor)
            event_tensor = self.random_shift_events(event_tensor)
            event_tensor = self.add_correlated_events(event_tensor)

        H = IMAGE_H
        W = IMAGE_W

        pos = event_tensor[event_tensor[:, 3] > 0]
        neg = event_tensor[event_tensor[:, 3] < 0]
        
        last_stamp = event_tensor[-1, 2]
        first_stamp = event_tensor[0, 2]
        deltaT = last_stamp - first_stamp
        
        pos_ts = 5 * (pos[:, 2] - first_stamp) / (deltaT+1)
        pos_tis = torch.floor(pos_ts)
        pos_tis_long = pos_tis.long()
        
        neg_ts = 5 * (neg[:, 2] - first_stamp) / (deltaT+1)
        neg_tis = torch.floor(neg_ts)
        neg_tis_long = neg_tis.long()
        
        # Get pos, neg counts
        pos = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W + pos_tis_long*H*W, minlength=5 * H * W).reshape(5, H, W)
        neg = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W + neg_tis_long*H*W, minlength=5 * H * W).reshape(5, H, W)

        return torch.stack([pos, neg], dim=3).permute(0, 3, 1, 2).float()

    def get_label(self, name):
        name = name.split(os.sep)[-2]
        label = self.label.index(name)
        return torch.LongTensor([label])

    def __len__(self):
        return len(self.file)

class PretrainImageNetDataset(ImageNetDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.point_level_aug = cfg.get("point_level_aug", False)
        self.img_augmentation_view1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.), interpolation = 0),
                transforms.RandomHorizontalFlip(),
        ])

        if self.point_level_aug:
            raise SystemExit
    
    
    def get_events(self, event):
        if not self.point_level_aug:
            raw_event = self.reshape_then_acc_count_pol(event, self.mode)
        else:
            raise SystemExit
        return raw_event
            
    def __getitem__(self, idx):
        
        event_path = self.file[idx]
        label = self.get_label(event_path)
    
        
        event = self.parse_event(event_path, self.cfg)
        event = self.get_events(event)
            
        data = {
            "event": event,
            "label": label, 
            }
        return data
    


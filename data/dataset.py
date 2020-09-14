# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform, num_views):
        
        self.num_views = num_views
        
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []
            
        if self.num_views:
#             for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
#                 self.sub_meta[y].append(x)
            stride = int(12/self.num_views)
            for x,y in zip(self.meta['image_names'][::stride],self.meta['image_labels'][::stride]):
                self.sub_meta[y].append(x) 
            shuffle = False
        else:
            for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
                self.sub_meta[y].append(x)
            shuffle = True

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                              shuffle = shuffle,
                              num_workers = 0, #use main thread only or may receive multiple batches
                              pin_memory = False)   
            
        for cl in self.cl_list:
            if self.num_views:
                #shuffle
                rand_idx = np.random.permutation(int(len(self.sub_meta[cl])/self.num_views))
                sub_meta_new = []
                for i in range(len(rand_idx)):
                    sub_meta_new.extend(self.sub_meta[cl][rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
                self.sub_meta[cl] = sub_meta_new
            sub_dataset = SubDataset(self.sub_meta[cl], cl, self.num_views, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, num_views, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.num_views = num_views

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        target = self.target_transform(self.cl)
        if self.num_views:
            imgs = []
            for j in range(self.num_views):
                image_path = os.path.join( self.sub_meta[i*self.num_views+j])
                img = Image.open(image_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
            return torch.stack(imgs), target
        else:
            image_path = os.path.join( self.sub_meta[i])
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.num_views:
            return int(len(self.sub_meta)/self.num_views)
        else:
            return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

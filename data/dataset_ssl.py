# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x
import math

def retrive_permutations(classes):
    all_perm = np.load('permutations_%d.npy' % (classes))
    # from range [1,9] to [0,8]
    if all_perm.min() == 1:
        all_perm = all_perm - 1

    return all_perm

def get_patches(img, transform_jigsaw, transform_patch_jigsaw, permutations):
#     #Do we need the commented line below?
#     if np.random.rand() < 0.30:
#         img = img.convert('LA').convert('RGB')## this should be L instead....... need to change that!!
    
    img = transform_jigsaw(img)
    s = float(img.size[0]) / 3
    a = s / 2
    tiles = [None] * 9
    for n in range(9):
        i = int(n / 3)
        j = n % 3
        c = [a * i * 2 + a, a * j * 2 + a]
        c = np.array([math.ceil(c[1] - a), math.ceil(c[0] - a), int(c[1] + a ), int(c[0] + a )]).astype(int)
        tile = img.crop(c.tolist())
        tile = transform_patch_jigsaw(tile)
        # Normalize the patches independently to avoid low level features shortcut
        m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
        s[s == 0] = 1
        norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
        tile = norm(tile)
        tiles[n] = tile
    order = np.random.randint(len(permutations))
    data = [tiles[permutations[order][t]] for t in range(9)]
    data = torch.stack(data, 0)

    return data, int(order)

def get_patches_mv(img, transform_jigsaw, transform_patch_jigsaw, permutations, order):
#     #Do we need the commented line below?
#     if np.random.rand() < 0.30:
#         img = img.convert('LA').convert('RGB')## this should be L instead....... need to change that!!
    
    img = transform_jigsaw(img)
    s = float(img.size[0]) / 3
    a = s / 2
    tiles = [None] * 9
    for n in range(9):
        i = int(n / 3)
        j = n % 3
        c = [a * i * 2 + a, a * j * 2 + a]
        c = np.array([math.ceil(c[1] - a), math.ceil(c[0] - a), int(c[1] + a ), int(c[0] + a )]).astype(int)
        tile = img.crop(c.tolist())
        tile = transform_patch_jigsaw(tile)
        # Normalize the patches independently to avoid low level features shortcut
        m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
        s[s == 0] = 1
        norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
        tile = norm(tile)
        tiles[n] = tile
    data = [tiles[permutations[order][t]] for t in range(9)]
    data = torch.stack(data, 0)

    return data


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
    def __init__(self, data_file, batch_size, transform, num_views, rotation=False, jigsaw=False, transform_jigsaw=None, transform_patch_jigsaw=None):
        self.num_views = num_views
        self.rotation = rotation
        self.jigsaw = jigsaw
        self.transform_jigsaw = transform_jigsaw
        self.transform_patch_jigsaw = transform_patch_jigsaw
        
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
                
            sub_dataset = SubDataset(self.sub_meta[cl], cl, self.num_views, transform = transform, rotation=self.rotation, jigsaw=self.jigsaw, transform_jigsaw=self.transform_jigsaw, transform_patch_jigsaw=self.transform_patch_jigsaw)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, num_views, transform=transforms.ToTensor(), target_transform=identity, rotation=False, jigsaw=False, transform_jigsaw=None, transform_patch_jigsaw=None):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.num_views = num_views
        
        self.rotation = rotation
        self.jigsaw = jigsaw
        if self.jigsaw:
            self.permutations = retrive_permutations(35)
            self.transform_jigsaw = transform_jigsaw
            self.transform_patch_jigsaw = transform_patch_jigsaw

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        target = self.target_transform(self.cl)
        if self.num_views:
            imgs = []
            all_patches = []
            if self.jigsaw:
                order = np.random.randint(len(self.permutations))
            for j in range(self.num_views):
                image_path = os.path.join( self.sub_meta[i*self.num_views+j])
                img = Image.open(image_path).convert('RGB')
                if self.jigsaw:
                    patches = get_patches_mv(img, self.transform_jigsaw, self.transform_patch_jigsaw, self.permutations, order)
                img = self.transform(img)
                imgs.append(img)
                all_patches.append(patches)
            return torch.stack(imgs), target, torch.stack(all_patches), order
        else:
            image_path = os.path.join( self.sub_meta[i])
            img = Image.open(image_path).convert('RGB')
            if self.jigsaw:
                patches, order = get_patches(img, self.transform_jigsaw, self.transform_patch_jigsaw, self.permutations)
            if self.rotation:
                rotated_imgs = [
                        self.transform(img),
                        self.transform(img.rotate(90,expand=True)),
                        self.transform(img.rotate(180,expand=True)),
                        self.transform(img.rotate(270,expand=True))
                    ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
            img = self.transform(img)
            if self.jigsaw:
                return img, target, patches, order
            elif self.rotation:
                return img, target, torch.stack(rotated_imgs, dim=0), rotation_labels
            else:
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

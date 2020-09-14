#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import argparse
import numpy as np

import configs
from data.datamgr import SetDataManager
import backbone
from methods.protonet import ProtoNet


# In[2]:


model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101) 


# In[3]:


script = 'train'
parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char')
parser.add_argument('--model'       , default='Conv4',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
parser.add_argument('--method'      , default='baseline',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
parser.add_argument('--n_query'      , default=16, type=int,  help='number of query images')
parser.add_argument('--num_views'   , default=None, type=int,  help='number of rendered views for each model')
parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly

if script == 'train':
    parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
    parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
    parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
    parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
    parser.add_argument('--lr'  , default=0.001, type=float, help ='Learning Rate')
    parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
    parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true')

# params = parser.parse_args('--dataset ModelNet --model ResNet18 --method protonet --n_shot 1 --num_views=1'.split())


# In[4]:


if __name__ == '__main__':
    
    #np.random.seed(10) #Why do we need to set this random seed?
    params = parser.parse_args()

    base_file = configs.data_dir[params.dataset] + 'base.json' 
    val_file   = configs.data_dir[params.dataset] + 'val.json' 
    #what about test file?

    if 'Conv' in params.model:
        image_size = 84
    else:
        image_size = 224 #for modelnet decide based on model architecture

    optimization = 'Adam'
    # start_epoch = 0
    start_epoch = params.start_epoch
    stop_epoch = 400
    #os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
    # CUDA_VISIBLE_DEVICES=6,7


    # In[5]:


    #code from mvcnn, add logging later
    #parse_args

    # num_models = 1000 #max number of models to use per class, add this functionality later
    # n_models_train = num_models*num_views

#     if params.num_views and params.num_views >=5:
#         n_query = max(1, int(8* params.test_n_way/params.train_n_way)) #why is this required?
#     else:
#         n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #why is this required?

    train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
    base_datamgr            = SetDataManager(image_size, n_query = params.n_query,  **train_few_shot_params, num_views = params.num_views)
    base_loader             = base_datamgr.get_data_loader(base_file , aug = params.train_aug)

    test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot)
    val_datamgr             = SetDataManager(image_size, n_query = params.n_query, **test_few_shot_params, num_views = params.num_views)
    val_loader              = val_datamgr.get_data_loader(val_file, aug = False)


    backbone = model_dict[params.model]
    model = ProtoNet(backbone, params.num_views, **train_few_shot_params)
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_%dway_%dshot_%dviews_lr%f' %(params.train_n_way, params.n_shot, params.num_views, params.lr)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])


    def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
        if optimization == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=params.lr)
        else:
           raise ValueError('Unknown optimization, please define by yourself')

        max_acc = 0       

        for epoch in range(start_epoch,stop_epoch):
            model.train()
            model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return 
            model.eval()

            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            acc = model.test_loop( val_loader)
            if acc > max_acc :
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

            if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
                outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        return model


    # In[ ]:


    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)


    # In[ ]:





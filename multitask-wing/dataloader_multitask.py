from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from Places205 import Places205
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv

from pdb import set_trace as breakpoint

# Set the paths of the datasets here.
FINGER_DATA = './data_trial'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

class MyFingerData(data.Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform
        #self.target_transform = target_transform

    def __getitem__(self, index):
       
        image_path = self.image_paths[index]
        filename = image_path.split('/')[-1]
        img = Image.open(image_path).convert('RGB')
        image_name = filename.split('.')[0]
        target = image_name.split('_')[0]

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.image_paths)

class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.num_imgs_per_cat = num_imgs_per_cat
        
        if self.dataset_name =='finger':
            assert(self.split=='train' or self.split=='test')
            self.transform=None
            split_data_dir = FINGER_DATA + '/' + self.split
            self.data = MyFingerData(split_data_dir, self.transform)
        
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
    
    
###################################################################################################
def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2))).copy()
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img)).copy()
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2)).copy()
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

def horizon_img(img,no=2):
    imarray = np.array(img)
    #print("---------->",imarray.shape)
    im1 = imarray[:85,:,:]
    im2 = imarray[85:170,:,:]
    im3 = imarray[170:,:,:]
    #print(im1.shape,im2.shape,im3.shape)
    if no==0:
        return Image.fromarray(imarray)   
    if no ==1:
        imm1=np.concatenate((im1,im3),axis=0)
        img_new = np.concatenate((imm1,im2),axis=0)
        #print(img_new.shape)
    elif no ==2:
        imm1=np.concatenate((im2,im3),axis=0)
        img_new = np.concatenate((imm1,im1),axis=0)
        #print(img_new.shape)
    elif no ==3:
        imm1=np.concatenate((im2,im1),axis=0)
        img_new = np.concatenate((imm1,im3),axis=0)
    elif no ==4:
        imm1=np.concatenate((im3,im2),axis=0)
        img_new = np.concatenate((imm1,im1),axis=0)
    elif no ==5:
        imm1=np.concatenate((im3,im1),axis=0)
        img_new = np.concatenate((imm1,im2),axis=0)
        
    aerr2img = Image.fromarray(img_new)
    return aerr2img
    
def vertic_img(img,no=2):
    imarray = np.array(img)
    #print("---------->",imarray.shape)
    im1 = imarray[:,:85,:]
    im2 = imarray[:,85:170,:]
    im3 = imarray[:,170:,:]
    #print(im1.shape,im2.shape,im3.shape)
    if no==0:
        return Image.fromarray(imarray)   
    if no ==1:
        imm1=np.concatenate((im1,im3),axis=1)
        img_new = np.concatenate((imm1,im2),axis=1)
        #print("*************",img_new.shape)
    elif no ==2:
        imm1=np.concatenate((im2,im3),axis=1)
        img_new = np.concatenate((imm1,im1),axis=1)
        #print("**************",img_new.shape)
    elif no ==3:
        imm1=np.concatenate((im2,im1),axis=1)
        img_new = np.concatenate((imm1,im3),axis=1)
    elif no ==4:
        imm1=np.concatenate((im3,im2),axis=1)
        img_new = np.concatenate((imm1,im1),axis=1)
    elif no ==5:
        imm1=np.concatenate((im3,im1),axis=1)
        img_new = np.concatenate((imm1,im2),axis=1)
        
    aerr2img = Image.fromarray(img_new)
    return aerr2img
###################################################################################################



class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True,
                 tk='rotation'):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers
        self.tk=tk
        
        self.r1=0
        self.h1=1
        self.v1=2
        print("****************", self.tk)
        #mean_pix  = self.dataset.mean_pix
        #std_pix   = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor()])
            #transforms.Normalize(mean=mean_pix, std=std_pix)
        
        self.inv_transform = transforms.Compose([
            
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])
        #Denormalize(mean_pix, std_pix),

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            #print("############@@@@@@@@@@ -----> using self supervision")
            
            def _load_function(idx):
                idx = idx % len(self.dataset)
                #print("########____________",idx)
                img0, _ = self.dataset[idx]
                img0=img0.resize((256,256))
                
                if self.tk=='rotation':
                    #print("yaaaaaaay rotaaa")
                    task='rotation'
                    rotated_imgs = [
                        self.transform(img0),
                        self.transform(rotate_img(img0,  90)),
                        self.transform(rotate_img(img0, 180)),
                        self.transform(rotate_img(img0, 270))
                    ]
                    rotation_labels = torch.LongTensor([0, 1, 2, 3])
                    k=torch.randint(4, size=(1,))
                    return rotated_imgs[int(k[0])],rotation_labels[int(k[0])],task
                
                elif self.tk=='vertical':
                    #print("yaaaaaaay verta")
                    task='vertical'
                    vertic_imgs = [
                        self.transform(img0),
                        self.transform(vertic_img(img0, 1)),
                        self.transform(vertic_img(img0, 2)),
                        self.transform(vertic_img(img0, 3)),
                        self.transform(vertic_img(img0, 4)),
                        self.transform(vertic_img(img0, 5))
                    ]
                    vertical_labels = torch.LongTensor([0, 1, 2, 3, 4, 5])
                    k=torch.randint(6, size=(1,))
                    return vertic_imgs[int(k[0])],vertical_labels[int(k[0])],task
                
                elif self.tk=='horizontal':
                    #print("yaaaaaaay horita")
                    task='horizontal'
                    horizon_imgs = [
                        self.transform(img0),
                        self.transform(horizon_img(img0, 1)),
                        self.transform(horizon_img(img0, 2)),
                        self.transform(horizon_img(img0, 3)),
                        self.transform(horizon_img(img0, 4)),
                        self.transform(horizon_img(img0, 5))
                    ]
                    horizontal_labels = torch.LongTensor([0, 1, 2, 3, 4, 5])
                    k=torch.randint(6, size=(1,))
                    return horizon_imgs[int(k[0])],horizontal_labels[int(k[0])],task
                else:
                    raise ValueError('DATALOADER FACING ISSUES WITH IDX NO')
               
               
                                    
                    
            def _collate_fun(batch):
                batch = default_collate(batch)
                #print("%%%%%%%%%%%%",batch[0].shape,batch[1].shape)
                assert(len(batch)==3)
                #print("$$$$$$$$$$$$$",batch[0].shape)
                rotations=1
                batch_size, channels, height, width = batch[0].size()
                #print("#####################____----_____",batch_size, rotations, channels, height, width)
                
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                batch[2] = batch[2][0]
                return batch
        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size

    
    
def get_loader(image_path, batch_size=3, num_workers=0, split='train',unsupervised=True,tk='rotation'):
    """Builds and returns Dataloader."""

    dataset = GenericDataset('finger',split='train', random_sized_crop=False)
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,tk = tk)
    return data_loader

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = GenericDataset('finger','train', random_sized_crop=False)
    dataloader1 = DataLoader(dataset, batch_size=6, unsupervised=True,tk='rotational')
    dataloader2 = DataLoader(dataset, batch_size=6, unsupervised=True,tk='vertical')
    dataloader3 = DataLoader(dataset, batch_size=6, unsupervised=True,tk='horizontal')
    print(len(dataloader1.dataset))
    
    for step, b in enumerate(dataloader1):
        data, label,task = b
        break
    
    print(label,task,"<==== my task and output")
    inv_transform = dataloader.inv_transform
    for i in range(data.size(0)):
        print("%%%%%%^^^^^^^^%%%%%%%%",label[i])
        plt.subplot(data.size(0)/6,6,i+1)
        fig=plt.imshow(inv_transform(data[i]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()



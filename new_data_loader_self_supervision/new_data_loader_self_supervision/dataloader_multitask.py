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
FINGER_DATA = './data_self_supervision'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds
    
    
class FingerPrintDataset(data.Dataset):
    def __init__(self,root_dir,train='True',transform=None,tk='rotation'):
        self.samples = []
        self.train =train
        self.transform =transform
        self.root_dir = root_dir
        self.data=[]
        self.tk=tk
        self.targets=[]
        self.tasks=[]
        self.__init__dataset()
        
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    
    def __init__dataset(self):
        
        if self.train:
            datadir=self.root_dir+"train/"+self.tk
        else:
            datadir=self.root_dir+"test/"+self.tk
        print(datadir)
        for image_name in os.listdir(datadir):
            #print(image_name)
            img_path = os.path.join(datadir,image_name)
            filename = image_name.split("/")[-1]
            filename = filename.split(".")[0]
            target = filename.split('_')[-1]
            task = filename.split('_')[-2]
            image = Image.open(img_path)
            image = image.convert('RGB')
            image = image.resize((256,256))
            
            #print(filename,target,task)
            if self.transform is not None:
                image = self.transform(image)
            self.data.append(image)
            self.targets.append(int(target))
            self.tasks.append(task)
        self.targets=torch.Tensor(self.targets)
        #self.tasks=torch.Tensor(self.tasks)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx],int(self.targets[idx]),self.tasks[idx]

class MyFingerData(data.Dataset):
    def __init__(self, root, transform=None,task='rotation'):
        
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform
        #self.target_transform = target_transform

    def __getitem__(self, index):
       
        image_path = self.image_paths[index]
        filename = image_path.split('/')[-1]
        img = Image.open(image_path).convert('RGB')
        image_name = filename.split('.')[0]
        target = image_name.split('_')[-1]
        task = image_name.split('_')[-2]

        if self.transform is not None:
            img = self.transform(img)
        return img, target,task

    def __len__(self):
        return len(self.image_paths)

class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None,task = 'rotation'):
        self.task = task
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.num_imgs_per_cat = num_imgs_per_cat
        
        if self.dataset_name =='finger':
            assert(self.split=='train' or self.split=='test')
            self.transform=None
            split_data_dir = FINGER_DATA + '/' + self.split + '/' + self.task
            self.data = MyFingerData(split_data_dir, self.transform,self.task)
        
    def __getitem__(self, index):
        img, label, task = self.data[index]
        return img, int(label), task

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
    
    
    

'''

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
        
        print("****************", self.tk)
        
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        
        
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
                
                img0, label, task = self.dataset[idx]
                #print("########____________",idx,label, task)
                img0=img0.resize((256,256))
                return self.transform(img0), label, task                    
                    
            def _collate_fun(batch):
                batch = default_collate(batch)
                #print("%%%%%%%%%%%%",batch[0].shape,batch[1].shape)
                assert(len(batch)==3)
                #print("$$$$$$$$$$$$$",batch[0].shape)
                
                batch_size, channels, height, width = batch[0].size()
                #print("#####################____----_____",batch_size, rotations, channels, height, width)
                
                batch[0] = batch[0].view([batch_size, channels, height, width])
                batch[1] = batch[1].view([batch_size])
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
'''
    
    
def get_loader(image_path, batch_size=3, num_workers=0, train=True ,transform=None,tk='rotation'):
    """Builds and returns Dataloader."""

    dataset = FingerPrintDataset(image_path,train=train,transform=transform,tk=tk)
    print(len(dataset))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
'''
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

'''

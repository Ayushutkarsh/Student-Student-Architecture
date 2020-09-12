
# coding: utf-8

# In[1]:


import torch
import os
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, random_split
from network import *


# In[2]:


from data_generator import FingerPrintDataset


# In[3]:


contact_data_root= './dataset_contactbased/'
contact_less_data_root= './dataset_contactless/'


# In[4]:



train_dataset = FingerPrintDataset(contact_data_root, train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()]))
train_label_dict=train_dataset.get_label_dict()
#print("train labels",train_label_dict)
test_dataset = FingerPrintDataset(contact_data_root, train=False,
                            transform=transforms.Compose([
                                transforms.ToTensor()]))

test_label_dict=test_dataset.get_label_dict()
#print(test_label_dict)


train_dataset_rgb = FingerPrintDataset(contact_less_data_root, train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),label_dict=train_label_dict)
#print(train_dataset_rgb.get_label_dict())
test_dataset_rgb = FingerPrintDataset(contact_less_data_root, train=False,
                            transform=transforms.Compose([
                                transforms.ToTensor()]),label_dict=test_label_dict)
#print(test_dataset_rgb.get_label_dict())


# In[5]:


i,l =train_dataset[4]
print("Data shape is CxHxW --->",i.shape)


# In[24]:


train_classes_set=set()
test_classes_set=set()
for x in train_dataset:
    train_classes_set.add(x[1])
for x in test_dataset:
    test_classes_set.add(x[1])

# train_classes = len(train_classes_set)
train_classes = 32
# print('train_classes', train_classes)
#test_classes = len(test_classes_set)
test_classes = 32
# print('test_classes', test_classes)
train_classes_samples=2
test_classes_samples=2
print(train_classes,test_classes)


# In[17]:


from losses import TripletLoss
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit
cuda = torch.cuda.is_available()


# In[18]:


from datasets import BalancedBatchSampler

train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=train_classes, n_samples=train_classes_samples)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=test_classes, n_samples=train_classes_samples)

train_batch_sampler_rgb = BalancedBatchSampler(train_dataset_rgb.train_labels, n_classes=train_classes, n_samples=train_classes_samples)
test_batch_sampler_rgb = BalancedBatchSampler(test_dataset_rgb.test_labels, n_classes=test_classes, n_samples=train_classes_samples)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)


online_train_loader_rgb = torch.utils.data.DataLoader(train_dataset_rgb, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader_rgb = torch.utils.data.DataLoader(test_dataset_rgb, batch_sampler=test_batch_sampler, **kwargs)


# In[19]:


# Set up the network and training parameters
import torchvision
from torch import nn
#from networks import EmbeddingNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric

margin = 1.


# In[20]:


#model


# In[21]:


# model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
model=torchvision.models.mobilenet_v2(pretrained=False)
model.load_state_dict(torch.load('mobilenet_v2-b0353104.pth'))
model_infeatures_final_layer=model.classifier[1].in_features
model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
for param in model.parameters():
    param.requires_grad = True
for param in model.features[18].parameters():
    param.requires_grad=True
for param in model.classifier.parameters():
    param.requires_grad=True
model_trained_mobilenet =model
print("All trainable parameters of model are")
for name, param in model_trained_mobilenet.named_parameters():
    if param.requires_grad:
        print (name,param.shape)
contact_model = multi_task_model_classification(model_trained_mobilenet)
contact_less_model = multi_task_model_classification(model_trained_mobilenet)


# In[22]:


if cuda:
    contact_less_model.cuda()
    contact_model.cuda()
AllTripletSelector1 =AllTripletSelector()
loss_fn = OnlineTripletLoss(margin, AllTripletSelector1)
lr = 1e-3
contact_optimizer = optim.Adam(contact_model.parameters(), lr=lr, weight_decay=1e-4)
contact_less_optimizer = optim.Adam(contact_less_model.parameters(), lr=lr, weight_decay=1e-4)
contact_scheduler = lr_scheduler.StepLR(contact_optimizer, 8, gamma=0.1, last_epoch=-1)
contact_less_scheduler = lr_scheduler.StepLR(contact_less_optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 300
log_interval = 5


# In[23]:


fit(online_train_loader,online_train_loader_rgb, online_test_loader,online_test_loader_rgb, contact_model,contact_less_model,loss_fn, contact_optimizer,contact_less_optimizer, contact_scheduler,contact_less_scheduler, n_epochs, cuda, log_interval)


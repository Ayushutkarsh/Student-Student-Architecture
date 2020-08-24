import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torchvision.utils import save_image
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net,U_Net_AB,multi_task_model,multi_task_model_classification
import csv
from tqdm import tqdm
import cv2


#################
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
################


class Solver(object):
    def __init__(self, config, train_loader_rot,train_loader_hor,train_loader_ver, valid_loader_rot,valid_loader_hor,valid_loader_ver, test_loader):

        # Data loader
        self.train_loader_rot = train_loader_rot
        self.train_loader_hor = train_loader_hor
        self.train_loader_ver = train_loader_ver
        
        self.valid_loader_rot = valid_loader_rot
        self.valid_loader_hor = valid_loader_hor
        self.valid_loader_ver = valid_loader_ver
        self.test_loader = test_loader
        print("@@@@@@@@@@@@@@@@@@------> length of dataset",len(self.train_loader_rot.dataset),len(self.train_loader_ver.dataset))

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.CrossEntropyLoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        print("@@@@@@@@@@@@@@@@@@@@@@@ LR B1 & B2 for Adam ------> ",self.lr,self.beta1,self.beta2)

        # Training settings
        self.num_epochs = config.num_epochs
        
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        #self.val_step = config.val_step
        self.val_step = 1

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=3,output_ch=3)
        elif self.model_type =='R2U_Net':
            print("------> using R2U <--------")            
            self.unet = R2U_Net(img_ch=3,output_ch=3,t=self.t)
        elif self.model_type =='AttU_Net':
            print("------> using AttU <--------")
            self.unet = AttU_Net(img_ch=3,output_ch=3)
        elif self.model_type == 'R2AttU_Net':
            print("------> using R2-AttU <--------")
            self.unet = R2AttU_Net(img_ch=3,output_ch=3,t=self.t)
        elif self.model_type == 'ABU_Net':
            print("------> using ABU_Net <--------")
            self.unet = U_Net_AB(img_ch=3,output_ch=1)
        elif self.model_type == 'Multi_Task':
            print("------> using Multi_Task Learning <--------")
            model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
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
            self.unet = multi_task_model_classification(model_trained_mobilenet)
            
            

        self.optimizer = optim.AdamW(list(self.unet.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self,SR,GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img


    def train(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#
        
        #unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

        unet_path='wassup'
        print("-------> started Training <------")# U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.
            
            for epoch in range(self.num_epochs):

                self.unet.train(True)
                epoch_loss_rot = 0
                epoch_loss_ver = 0
                epoch_loss_hor = 0
                count_rot = 0
                count_ver = 0
                count_hor = 0
                acc_rot = 0
                acc_ver = 0
                acc_hor = 0
                
                acc = 0.    # Accuracy
                SE = 0.        # Sensitivity (Recall)
                SP = 0.        # Specificity
                PC = 0.     # Precision
                F1 = 0.        # F1 Score
                JS = 0.        # Jaccard Similarity
                DC = 0.        # Dice Coefficient
                length = 0
                for i, (data,label,task) in enumerate(tqdm(self.train_loader_rot(epoch))):
                    #for i, (data,label,task) in enumerate(tqdm(self.train_loader(epoch))):
                    # GT : Ground Truth
                    #print("%%%%%%%%%", i, "image is", images.shape,GT.shape)
                    #print(i)
                    #----------------- first task is rotation -----------------------
                    self.optimizer.zero_grad()
                    
                    data = data.to(self.device)
                    #print(data.shape)
                    label = label.to(self.device)
                    #print(label,task)
                    #task = task.to(self.device)
                    SR_rot = self.unet(data,'rotation')
                    loss_rot = self.criterion(SR_rot,label)
                    preds_rot = SR_rot.argmax(1, keepdim=True)
                    correct_rot = preds_rot.eq(label.view_as(preds_rot)).sum()
                    count_rot+=data.size(0)
                    acc_rot+=correct_rot.float()
                    epoch_loss_rot += loss_rot.item()
                    self.reset_grad()
                    loss_rot.backward()
                    self.optimizer.step()
                    #----------------- second task is horizontal ----------------------- 
                    for b in self.train_loader_hor(i):
                            (data,label,task) = b
                    self.optimizer.zero_grad()
                    
                    data = data.to(self.device)
                    #print(data.shape)
                    label = label.to(self.device)
                    #print(label,task)
                    #task = task.to(self.device)
                    SR_hor = self.unet(data,'horizontal')
                    loss_hor = self.criterion(SR_hor,label)
                    preds_hor = SR_hor.argmax(1, keepdim=True)
                    correct_hor = preds_hor.eq(label.view_as(preds_hor)).sum()
                    count_hor+=data.size(0)
                    acc_hor+=correct_hor.float()
                    epoch_loss_hor += loss_hor.item()
                    self.reset_grad()
                    loss_hor.backward()
                    self.optimizer.step()
                    #---------------------- third task is vertical -----------------------
                    for b in self.train_loader_ver(i):
                            (data,label,task) = b
                    self.optimizer.zero_grad()
                    
                    data = data.to(self.device)
                    #print(data.shape)
                    label = label.to(self.device)
                    #print(label,task)
                    #task = task.to(self.device)
                    SR_ver = self.unet(data,'vertical')
                    loss_ver = self.criterion(SR_ver,label)
                    preds_ver = SR_ver.argmax(1, keepdim=True)
                    correct_ver = preds_ver.eq(label.view_as(preds_ver)).sum()
                    count_ver+=data.size(0)
                    acc_ver+=correct_ver.float()
                    epoch_loss_ver += loss_ver.item()
                    self.reset_grad()
                    loss_ver.backward()
                    self.optimizer.step()
                    
                    
                    '''if i-1>=0 and (i-1)%3==0:
                        for b in self.train_loader_hor(i):
                            (data,label,task) = b
                    elif i-2>=0 and (i-2)%3==0:
                        for b in self.train_loader_ver(i):
                            (data,label,task) = b
                    self.optimizer.zero_grad()
                    #print("------->", data.shape,task)
                    #data = data.to(self.device)
                    #label = label.to(self.device)
                    #task = task.to(self.device)
                    #print("TASK IS ---->",task,label)
                    # SR : Segmentation Result
                    if task =='rotation':
                        SR = self.unet(data,'rotation')
                    elif task =='horizontal':
                        SR = self.unet(data,'horizontal')
                    elif task =='vertical':
                        SR = self.unet(data,'vertical')
                    #print("output of model is --->", SR)
                    #print("label is --->", label)
                    #print("Shape of unet output",SR.shape)
                    #SR_probs = F.sigmoid(SR)
                    #print("Shape of unet output",SR_probs.shape)
                    #SR_flat = SR.view(SR.size(0),-1)
                    #print("Shape of unet output",SR_flat.shape)
                    #print("Shape of GT output",GT.shape)
                    #GT_flat = label.view(label.size(0),-1)
                    #print("Shape of GT output",GT_flat.shape)
                    loss = self.criterion(SR,label)
                    print("------------->",loss,"<----------------")
                    print("------------->",SR,label,"<----------------")
                    preds = SR.argmax(1, keepdim=True)
                    print("------------->",preds,"<----------------")
                    correct = preds.eq(label.view_as(preds)).sum()
                    #print("task",task,"predictions", preds,"label",label,correct,correct.float())
                    #acc = correct.float()/preds.shape[0]
                    if task =='rotation':
                        count_rot+=data.size(0)
                        acc_rot+=correct.float()
                        epoch_loss_rot += loss.item()
                        
                    elif task =='horizontal':
                        count_hor+=data.size(0)
                        acc_hor+=correct.float()
                        epoch_loss_hor +=loss.item()
                    elif task =='vertical':
                        count_ver +=data.size(0)
                        acc_ver+=correct.float()
                        epoch_loss_ver +=loss.item()

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()'''

                # Print the log info
                if count_rot==0:
                    count_rot=100000
                    print("NOTE ------> No rotation data in this epoch <---------")
                if count_ver==0:
                    count_ver=100000
                    print("NOTE ------> No vertical data in this epoch <---------")
                if count_hor==0:
                    count_hor=100000
                    print("NOTE ------> No horizontal data in this epoch <---------")
                print('Epoch [%d/%d], \n Loss rotation: %.4f, Loss horizontal: %.4f,Loss vertical: %.4f \n acc rotation: %.4f, acc horizontal: %.4f,acc vertical: %.4f' % (
                      epoch+1, self.num_epochs, \
                      epoch_loss_rot,\
                      epoch_loss_hor,epoch_loss_ver,\
                acc_rot/count_rot,acc_hor/count_hor,acc_ver/count_ver))
                #print("rot,hor,ver counts are",count_rot,count_hor,count_ver)
                      
                
            
                if (epoch+1)%200==0:
                    print("saving model")
                    unet_path_epoch = os.path.join(self.model_path, 'epoch-%d-%s-%d-%.4f-%d-%.4f.pkl' %(epoch,self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
                    saved_epoch = self.unet.state_dict()
                    torch.save(saved_epoch,unet_path_epoch)
                # Decay learning rate
                if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decay learning rate to lr: {}.'.format(lr))
                    
                #===================================== Validation ====================================#
                if epoch%10 ==0:
                    self.unet.train(False)
                    self.unet.eval()
                    for i, (data,label,task) in enumerate(tqdm(self.valid_loader_rot(epoch))):
                        
                         #----------------- first task is rotation -----------------------
                        self.optimizer.zero_grad()

                        data = data.to(self.device)
                        label = label.to(self.device)
                        #task = task.to(self.device)
                        SR_rot = self.unet(data,'rotation')
                        loss_rot = self.criterion(SR_rot,label)
                        preds_rot = SR_rot.argmax(1, keepdim=True)
                        correct_rot = preds_rot.eq(label.view_as(preds_rot)).sum()
                        count_rot+=data.size(0)
                        acc_rot+=correct_rot.float()
                        epoch_loss_rot += loss_rot.item()
                        self.reset_grad()
                        loss_rot.backward()
                        self.optimizer.step()
                        #----------------- second task is horizontal ----------------------- 
                        for b in self.valid_loader_hor(i):
                                (data,label,task) = b
                        self.optimizer.zero_grad()

                        data = data.to(self.device)
                        label = label.to(self.device)
                        #task = task.to(self.device)
                        SR_hor = self.unet(data,'horizontal')
                        loss_hor = self.criterion(SR_hor,label)
                        preds_hor = SR_hor.argmax(1, keepdim=True)
                        correct_hor = preds_hor.eq(label.view_as(preds_hor)).sum()
                        count_hor+=data.size(0)
                        acc_hor+=correct_hor.float()
                        epoch_loss_hor += loss_hor.item()
                        self.reset_grad()
                        loss_hor.backward()
                        self.optimizer.step()
                        #---------------------- third task is vertical -----------------------
                        for b in self.valid_loader_ver(i):
                                (data,label,task) = b
                        self.optimizer.zero_grad()

                        data = data.to(self.device)
                        label = label.to(self.device)
                        #task = task.to(self.device)
                        SR_ver = self.unet(data,'vertical')
                        loss_ver = self.criterion(SR_ver,label)
                        preds_ver = SR_ver.argmax(1, keepdim=True)
                        correct_ver = preds_ver.eq(label.view_as(preds_ver)).sum()
                        count_ver+=data.size(0)
                        acc_ver+=correct_ver.float()
                        epoch_loss_ver += loss_ver.item()
                        self.reset_grad()
                        loss_ver.backward()
                        self.optimizer.step()

                   

                    # Print the log info
                    if count_rot==0:
                        count_rot=100000
                        print("NOTE ------> No rotation data in this epoch <---------")
                    if count_ver==0:
                        count_ver=100000
                        print("NOTE ------> No vertical data in this epoch <---------")
                    if count_hor==0:
                        count_hor=100000
                        print("NOTE ------> No horizontal data in this epoch <---------")
                    print('Validation Losses ---------------->  Epoch [%d/%d], Loss rotation: %.4f, Loss horizontal: %.4f,Loss vertical: %.4f, acc rotation: %.4f, acc horizontal: %.4f,acc vertical: %.4f' % (
                          epoch+1, self.num_epochs, \
                          epoch_loss_rot/count_rot,\
                          epoch_loss_hor/count_hor,epoch_loss_ver/count_ver,\
                    acc_rot/count_rot,acc_hor/count_hor,acc_ver/count_ver))


                
                
                    
                
                
            
    def plot_sample(self,X, y, preds, binary_preds, ix=None):
        """Function to plot the results"""
        if ix is None:
            ix = random.randint(0, len(X))

        has_mask = y[ix].max() > 0

        fig, ax = plt.subplots(1, 4, figsize=(20, 10))
        ax[0].imshow(X[ix, ..., 0], cmap='seismic')
        if has_mask:
            ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
        ax[0].set_title('Seismic')

        ax[1].imshow(y[ix].squeeze())
        ax[1].set_title('Salt')

        ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
        if has_mask:
            ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
        ax[2].set_title('Salt Predicted')

        ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
        if has_mask:
            ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
        ax[3].set_title('Salt Predicted binary')
        plt.show()
            
    def test(self):
            print("YAAAAAy Testing")     
            #del self.unet
            #del best_unet
            self.build_model()
            unet_test_load_model='epoch-1-ABU_Net-250-0.0002-162-0.5965.pkl'
            unet_path=os.path.join(self.model_path,unet_test_load_model)
            print("@@@@@@@@@@@",unet_path)
            self.unet.load_state_dict(torch.load(unet_path))
            
            self.unet.train(False)
            self.unet.eval()

            acc = 0.    # Accuracy
            SE = 0.        # Sensitivity (Recall)
            SP = 0.        # Specificity
            PC = 0.     # Precision
            F1 = 0.        # F1 Score
            JS = 0.        # Jaccard Similarity
            DC = 0.        # Dice Coefficient
            length=0
            test_img_dir=self.result_path+'/'+str(self.num_epochs)+"/result_images/"
            if not os.path.exists(test_img_dir):
                os.makedirs(test_img_dir)
            for i, (images, GT) in enumerate(self.valid_loader):

                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = F.sigmoid(self.unet(images))
                print("Img",images.shape)
                print("GT",GT.shape)
                print(SR.shape)
                SR_preds_t = (SR > 0.5)
                print(SR_preds_t.shape)
                for j,GT_per in enumerate(GT):
                    print("@@@@@@@@@", GT_per.shape)
                    GT_name= str(j+1)+'_in_batch'+str(i+1)+'_ground_truth.png'
                    SR_name= str(j+1)+'_in_batch'+str(i+1)+'_output.png'
                    save_image(GT[j],self.result_path+'/'+str(self.num_epochs)+"/result_images/"+GT_name)
                    save_image(SR[j],self.result_path+'/'+str(self.num_epochs)+"/result_images/"+SR_name)
                    #cv2.imwrite(self.result_path+'/'+self.model_type+'/'+str(self.num_epochs)+"/result_images/"+GT_name,np.uint8(GT[j].detach().numpy()*255.0))
                    #cv2.imwrite(self.result_path+'/'+self.model_type+'/'+str(self.num_epochs)+"/result_images/"+SR_name,np.uint8(SR[j].detach().numpy()*255.0))
                
                #self.plot_sample(images.detach().numpy(), GT.detach().numpy(), SR.detach().numpy(), SR_preds_t.detach().numpy(),ix=1)
                
                
                
                acc += get_accuracy(SR,GT)
                SE += get_sensitivity(SR,GT)
                SP += get_specificity(SR,GT)
                PC += get_precision(SR,GT)
                F1 += get_F1(SR,GT)
                JS += get_JS(SR,GT)
                DC += get_DC(SR,GT)
                length += images.size(0)
                
                    
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            unet_score = JS + DC


            f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr])
            f.close()
            

            
        
            


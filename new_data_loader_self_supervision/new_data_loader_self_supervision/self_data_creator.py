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

FINGER_DATA = './data_trial'
FINGER_DATA_SELF_SUPERVISED = './data_self_supervision'


def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 1: # 90 degrees rotation
        return Image.fromarray(np.flipud(np.transpose(img, (1,0,2))).copy())
    elif rot == 2: # 90 degrees rotation
        return Image.fromarray(np.fliplr(np.flipud(img)).copy())
    elif rot == 3: # 270 degrees rotation / or -90
        return Image.fromarray(np.transpose(np.flipud(img), (1,0,2)).copy())
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

if __name__ == '__main__':
    FINGER_DATA = './data_trial/train'
    FF = './data_self_supervision/train'
    image_paths = list(map(lambda x: os.path.join(FINGER_DATA, x), os.listdir(FINGER_DATA)))
    for image_path in image_paths:
        
        print(image_path)
        filename = image_path.split('/')[-1]
        img = Image.open(image_path).convert('RGB')
        target = filename.split('.')[0]
        
        for d in ['rotation','horizontal','vertical']:
            #print(d)
            if d =='rotation':
                idx=[0,1,2,3]
                for i in idx:
                    #print(i)
                    image_new = rotate_img(img,i)
                    target_rot = target+"_rotation_" + str(i)
                    save_file = FF+'/rotation/'+target_rot+'.png'
                    image_new.save(save_file, "PNG")
            if d =='horizontal':
                idx=[0,1,2,3,4,5]
                for i in idx:
                    image_new = horizon_img(img,i)
                    target_rot = target+"_horizontal_" + str(i)
                    save_file = FF+'/horizontal/'+target_rot+'.png'
                    image_new.save(save_file, "PNG")
            if d =='vertical':
                idx=[0,1,2,3,4,5]
                for i in idx:
                    image_new = vertic_img(img,i)
                    target_rot = target+"_vertical_" + str(i)
                    save_file = FF+'/vertical/'+target_rot+'.png'
                    image_new.save(save_file, "PNG")
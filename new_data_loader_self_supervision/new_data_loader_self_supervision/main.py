import argparse
import os
from torchvision import transforms,datasets
from solver import Solver
from dataloader_multitask import get_loader,FingerPrintDataset
from torch.backends import cudnn
import random


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net','ABU_Net','Multi_Task']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/Multi_Task')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = random.choice([100,150,200,250])
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    #config.lr = lr
    config.num_epochs_decay = decay_epoch
    #config.batch_size=5

    print(config)
    print("########_______**********&&&&&&&&&&&& using batch size -------->",config.batch_size)
    
    '''
    train_loader_rot = get_loader(image_path='./data_trial',batch_size=config.batch_size, num_workers=2, split='train',unsupervised=True,tk='rotation')
    train_loader_hor = get_loader(image_path='./data_trial',batch_size=config.batch_size, num_workers=2, split='train',unsupervised=True,tk='horizontal')
    train_loader_ver = get_loader(image_path='./data_trial',batch_size=config.batch_size, num_workers=2, split='train',unsupervised=True,tk='vertical')
    valid_loader_rot = get_loader(image_path='./data_trial',batch_size=config.batch_size, num_workers=2, split='valid',unsupervised=True,tk='rotation')
    valid_loader_hor = get_loader(image_path='./data_trial',batch_size=config.batch_size, num_workers=2, split='valid',unsupervised=True,tk='horizontal')
    valid_loader_ver = get_loader(image_path='./data_trial',batch_size=config.batch_size, num_workers=2, split='valid',unsupervised=True,tk='vertical')
    test_loader = get_loader(image_path='./data_trial',batch_size=config.batch_size, num_workers=2, split='test',unsupervised=True)
    '''
    train_loader_rot = get_loader('./data_self_supervision/', train=True,
    transform=transforms.Compose([
        transforms.ToTensor()]),tk='rotation')
    train_loader_hor = get_loader('./data_self_supervision/', train=True,
    transform=transforms.Compose([
        transforms.ToTensor()]),tk='horizontal')
    train_loader_ver = get_loader('./data_self_supervision/', train=True,
    transform=transforms.Compose([
        transforms.ToTensor()]),tk='vertical')
    valid_loader_rot = get_loader('./data_self_supervision/', train=False,
    transform=transforms.Compose([
        transforms.ToTensor()]),tk='rotation')
    valid_loader_hor = get_loader('./data_self_supervision/', train=False,
    transform=transforms.Compose([
        transforms.ToTensor()]),tk='horizontal')
    valid_loader_ver = get_loader('./data_self_supervision/', train=False,
    transform=transforms.Compose([
        transforms.ToTensor()]),tk='vertical')
    test_loader = get_loader('./data_self_supervision/', train=False,
    transform=transforms.Compose([
        transforms.ToTensor()]),tk='rotation')

    solver = Solver(config, train_loader_rot,train_loader_hor,train_loader_ver, valid_loader_rot,valid_loader_hor,valid_loader_ver, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='Multi_Task', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/ABU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)

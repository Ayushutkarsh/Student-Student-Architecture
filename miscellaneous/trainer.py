import torch
import numpy as np
import os
import skimage.io as io
import warnings
from PIL import Image


def fit(contact_train_loader,contact_less_train_loader, contact_val_loader,contact_less_val_loader,contact_model, contact_less_model, loss_fn, contact_optimizer,contact_less_optimizer, contact_scheduler,contact_less_scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        contact_scheduler.step()
        contact_less_scheduler.step()
        

    for epoch in range(start_epoch, n_epochs):
        contact_scheduler.step()
        contact_less_scheduler.step()

        # Train stage
        print("Epoch Number -------------->",epoch)
        train_loss, metrics = train_epoch(contact_train_loader,contact_less_train_loader,contact_model, contact_less_model, loss_fn, contact_optimizer,contact_less_optimizer, cuda, log_interval, metrics,epoch)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.8f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        
        contact_val_loss,contact_less_val_loss, metrics = test_epoch(contact_val_loader,contact_less_val_loader, contact_model, contact_less_model, loss_fn, cuda, metrics)
        

        message += '\nEpoch: {}/{}. Validation set: \n Average contact loss: {:.8f} \n Average contact less loss: {:.8f}'.format(epoch + 1, n_epochs,contact_val_loss, contact_less_val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        if (epoch+1)%5==0 or epoch==n_epochs-1:
            contact_save_path = './contact_saved_model/model_train_loss_{}_epoch_{}'.format(train_loss,epoch)
            contact_less_save_path = './contact_less_saved_model/model_train_loss_{}_epoch_{}'.format(train_loss,epoch)
            model_saver(contact_model,contact_less_model,contact_save_path,contact_less_save_path)
            print("model saved as", contact_save_path," and ",contact_less_save_path)
        print(message)
        

def model_saver(contact_model,contact_less_model,CONTACT_PATH,CONTACT_LESS_PATH):
    torch.save(contact_model.state_dict(), CONTACT_PATH)
    torch.save(contact_less_model.state_dict(), CONTACT_LESS_PATH)
def train_epoch(contact_train_loader,contact_less_train_loader, contact_model, contact_less_model, loss_fn, contact_optimizer,contact_less_optimizer, cuda, log_interval, metrics,epoch=0):
    for metric in metrics:
        metric.reset()

    contact_model.train()
    contact_less_model.train()
    contact_losses = []
    contact_less_losses = []
    total_loss = 0
    batch_idx=0
    for batch_idx, (contact_data, contact_target) in enumerate(contact_train_loader):
        #print(batch_idx,contact_data.shape,contact_target.shape)
        contact_target = contact_target if len(contact_target) > 0 else None
        if cuda:
            contact_data = contact_data.cuda()
            if contact_target is not None:
                contact_target = contact_target.cuda()
        for batch_2, (cl_data, cl_target) in enumerate(contact_less_train_loader):
            if batch_2 ==batch_idx:
                contact_less_data, contact_less_target = cl_data, cl_target
                break
            else:
                continue
        contact_less_target = contact_less_target if len(contact_less_target) > 0 else None
        if cuda:
            contact_less_data = contact_less_data.cuda()
            if contact_less_target is not None:
                contact_less_target = contact_less_target.cuda()


        #optimizer.zero_grad() #<------------- I dont know how to work with this in student student
        contact_data=torch.autograd.Variable(contact_data)
        contact_less_data=torch.autograd.Variable(contact_less_data)
        contact_outputs = contact_model(contact_data,'student')
        contact_less_outputs = contact_less_model(contact_less_data,'student')
        #contact_outputs = contact_model(contact_data,'student')
        #contact_less_outputs = contact_less_model(contact_less_data,'student')

        if epoch%2==0:
            print("epoch number",epoch, "using contact optimizer")
            contact_optimizer.zero_grad()
            anchor_embeddings=contact_outputs
            search_embeddings=contact_less_outputs 
            if contact_target is not None and contact_less_target is not None:
                anchor_target=contact_target
                search_target=contact_less_target
                
            loss_outputs = loss_fn(anchor_embeddings,search_embeddings, anchor_target,search_target)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            if loss==0:
                print("No backprop as no triplets <------------------------")
                continue
            contact_losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            contact_optimizer.step()
        else:
            print("epoch number",epoch, "using contact less optimizer")
            contact_less_optimizer.zero_grad()
            anchor_embeddings=contact_less_outputs
            search_embeddings=contact_outputs 
            if contact_target is not None and contact_less_target is not None:
                anchor_target=contact_less_target
                search_target=contact_target
                
            loss_outputs = loss_fn(anchor_embeddings,search_embeddings, anchor_target,search_target)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            if loss==0:
                print("No backprop as no triplets <------------------------")
                continue
            contact_less_losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            contact_less_optimizer.step()
            
                
        
        

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\t Contact Loss: {:.6f} \t Contact Less Loss: {:.6f}'.format(
                batch_idx * len(contact_data[0]), len(contact_train_loader.dataset),
                100. * batch_idx / len(contact_train_loader), np.mean(contact_losses),np.mean(contact_less_losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(contact_val_loader,contact_less_val_loader, contact_model, contact_less_model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        contact_model.eval()
        contact_less_model.eval()
        total_contact_less_loss = 0
        total_contact_loss = 0
        batch_idx =0
        for batch_idx, (contact_data, contact_target) in enumerate(contact_val_loader):
            #print(batch_idx,contact_data.shape,contact_target.shape)
            contact_target = contact_target if len(contact_target) > 0 else None
            if cuda:
                contact_data = contact_data.cuda()
                if contact_target is not None:
                    contact_target = contact_target.cuda()
            for batch_2, (cl_data, cl_target) in enumerate(contact_less_val_loader):
                if batch_2 ==batch_idx:
                    contact_less_data, contact_less_target = cl_data, cl_target
                    break
                else:
                    continue
            contact_less_target = contact_less_target if len(contact_less_target) > 0 else None
            if cuda:
                contact_less_data = contact_less_data.cuda()
                if contact_less_target is not None:
                    contact_less_target = contact_less_target.cuda()


            #optimizer.zero_grad() #<------------- I dont know how to work with this in student student
            contact_data=torch.autograd.Variable(contact_data)
            contact_less_data=torch.autograd.Variable(contact_less_data)
            contact_outputs = contact_model(contact_data,'student')
            contact_less_outputs = contact_less_model(contact_less_data,'student')
            #contact_outputs = contact_model(contact_data,'student')
            #contact_less_outputs = contact_less_model(contact_less_data,'student')

            
            print("Validation :  using contact optimizer")
            anchor_embeddings=contact_outputs
            search_embeddings=contact_less_outputs 
            if contact_target is not None and contact_less_target is not None:
                anchor_target=contact_target
                search_target=contact_less_target

            loss_outputs = loss_fn(anchor_embeddings,search_embeddings, anchor_target,search_target)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            if loss==0:
                print("No backprop as no triplets <------------------------")
                continue
            total_contact_loss += loss.item()
            
            
            print("Validation :  using contact less optimizer")
            anchor_embeddings=contact_less_outputs
            search_embeddings=contact_outputs 
            if contact_target is not None and contact_less_target is not None:
                anchor_target=contact_less_target
                search_target=contact_target

            loss_outputs = loss_fn(anchor_embeddings,search_embeddings, anchor_target,search_target)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            total_contact_less_loss += loss.item()
            #print("val losses",total_contact_loss,total_contact_less_loss,batch_idx + 1)
            for metric in metrics:
                metric(outputs, target, loss_outputs)
                
        total_contact_loss/= (batch_idx + 1)
        total_contact_less_loss/= (batch_idx + 1)
        print("val losses",total_contact_loss,total_contact_less_loss,batch_idx + 1)

    return total_contact_loss,total_contact_less_loss, metrics

def final_test_epoch(probe_path, gallery_path, model, metrics,transform=None):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        probes = []
        gallery = []

        for image_name in os.listdir(probe_path):
            img_path = os.path.join(probe_path,image_name)
            filename = image_name.split("/")[-1]
            image = Image.open(img_path)
            image = image.convert('RGB')
            if transform is not None:
                image = transform(image)
            image = image.unsqueeze(0)
            embedding = model.get_embedding(image).data.cpu().numpy()
            probes.append((img_path,embedding))


        for image_name in os.listdir(gallery_path):
            img_path = os.path.join(gallery_path,image_name)
            filename = image_name.split("/")[-1]
            image = Image.open(img_path)
            image = image.convert('RGB')
            if transform is not None:
                image = transform(image)
            image = image.unsqueeze(0)
            embedding = model.get_embedding(image).data.cpu().numpy()
            gallery.append((img_path,embedding))

        top1=[]
        for x in probes:
            emb=x[1]
            probe_path=x[0]
            nns = sorted(gallery, key=lambda l: np.sum(np.square(l[1]-emb)))
            probe_filename=probe_path.split('/')[-1].split('.')[0]
            probe_folder='/'.join(probe_path.split('/')[:-1])
            top1.append(nns[0][0])
            with open(probe_folder+'/'+probe_filename+"_matching.txt",'w') as f:
                for line in nns:
                    f.write(line[0]+" "+str(np.sqrt(np.sum(np.square(line[1]-emb))))+"\n")
        return zip(probes,top1)
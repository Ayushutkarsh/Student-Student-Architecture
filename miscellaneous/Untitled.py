
# coding: utf-8

# In[9]:


import torch
import torchvision.transforms as tf


# In[37]:


input1 = torch.rand(3, 2)
input2 = torch.rand(3, 2)
#cos = torch.nn.CosineSimilarity()
#output = cos(input1, input2)


# In[38]:


input1


# In[39]:


mm=torch.mean(input1,dim=1)


# In[40]:


ip_mean=torch.mean(input1,dim=1)


# In[130]:


iii=torch.tensor([[-1.0,4.0,7.0],[2.0,-14.0,66.0]])


# In[131]:


iii.shape


# In[132]:


i_m=torch.mean(iii,dim=1)


# In[133]:


i_m.unsqueeze_(-1)


# In[134]:


i_m.shape


# In[135]:


iii[:,]-i_m[:,]


# In[136]:


mini = torch.min(iii,dim=1)[0].unsqueeze(-1)
maxi = torch.max(iii,dim=1)[0].unsqueeze(-1)
mami=maxi-mini


# In[137]:


iii


# In[138]:


mini=torch.min(iii)
maxi=torch.max(iii)
maxi-mini


# In[139]:


iii-mini


# In[143]:


iii=(iii-mini)/(maxi-mini)


# In[144]:


iii


# In[142]:


kk


# In[54]:


input1 = torch.nn.functional.normalize(input1,dim=1,p=2)


# In[55]:


input1


# In[31]:


tensor([[0.7409, 0.6716],
        [0.9653, 0.2611],
        [0.4602, 0.8878]])


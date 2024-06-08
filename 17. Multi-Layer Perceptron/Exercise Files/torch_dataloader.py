#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import DataLoader
import torch


# In[2]:


t = torch.arange(7, dtype=torch.float32)
data_loader = DataLoader(t)


# In[3]:


for item in data_loader:
    print(item)


# In[3]:


t


# In[6]:


data_loader = DataLoader(t, batch_size=3, drop_last=False, shuffle=True)

for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', batch)


# In[5]:


torch.manual_seed(23)
data_loader = DataLoader(t, batch_size=3, drop_last=True, shuffle=True)
for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', batch)


# In[7]:


from torch.utils.data import Dataset

class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# In[12]:


torch.manual_seed(23)
t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.arange(4)


# In[13]:


t_x


# In[14]:


t_y


# In[16]:


joint_dataset = JointDataset(t_x, t_y)
joint_dataset.x


# In[17]:


len(joint_dataset)


# In[20]:


joint_dataset[3]


# In[21]:


for example in joint_dataset:
    print('  x: ', example[0],
          '  y: ', example[1])


#  Or use Class `TensorDataset` directly

# In[22]:


torch.manual_seed(23)
from torch.utils.data import TensorDataset
joint_dataset = TensorDataset(t_x, t_y)

for example in joint_dataset:
    print('  x: ', example[0],
          '  y: ', example[1])


# In[23]:


torch.manual_seed(1)
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)

for i, batch in enumerate(data_loader, 1):
        print(f'batch {i}:', 'x:', batch[0],
              '\n         y:', batch[1])

for epoch in range(2):
    print(f'epoch {epoch+1}')
    for i, batch in enumerate(data_loader, 1):
        print(f'batch {i}:', 'x:', batch[0],
              '\n         y:', batch[1])


# In[24]:


import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


# In[25]:


df = pd.read_csv("C:/Training/Academy/Statistics (Python)/Cases/human-resources-analytics/HR_comma_sep.csv")
dum_df = pd.get_dummies(df,drop_first=True)
dum_df.head()


# In[26]:


X = dum_df.drop('left', axis=1)
scaler = MinMaxScaler()

y = dum_df['left'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=23,stratify=y)
X_scl_trn = scaler.fit_transform(X_train) 
X_scl_tst = scaler.transform(X_test) 


# In[27]:


X_torch = torch.from_numpy(X_scl_trn)
y_torch = torch.from_numpy(y_train)
print(X_torch.size())
print(y_torch.size())


# In[28]:


data_loader = DataLoader(y_torch, batch_size=30, drop_last=False)

for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', batch)


# In[29]:


from torch.utils.data import TensorDataset
joint_dataset = TensorDataset(X_torch.float(), y_torch.float())


# In[30]:


type(joint_dataset)


# In[43]:


torch.manual_seed(23)
data_loader = DataLoader(dataset=joint_dataset, batch_size=16, shuffle=True)

#for i, batch in enumerate(data_loader, 1):
#        print(f'batch {i}:', 'x:', batch[0].shape,
#              '\n         y:', batch[1].shape)



# In[32]:


# Create a model
model = nn.Sequential(nn.Linear(in_features=X_scl_trn.shape[1], out_features=5),
                      nn.ReLU(),
                      nn.Linear(5, 3),
                      nn.ReLU(),
                      nn.Linear(3,1),
                      nn.Sigmoid())


# In[33]:


criterion = torch.nn.BCELoss()
# Construct the optimizer (Adam in this case)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
optimizer


# Prediction with Default Weights

# In[34]:


y_pred = model(X_torch.float())
y_torch = y_torch.unsqueeze(1)
print(y_torch.shape)
print(y_pred.shape)


# In[44]:


#for epoch in range(2):
#    print(f'epoch {epoch+1}')
#    for i, batch in enumerate(data_loader, 1):
        #print(f'batch {i}:', 'x:', batch[0],
         #     '\n         y:', batch[1])
#        print((batch[0].shape, batch[1].shape))


# In[54]:


# Gradient Descent

for epoch in np.arange(0,100):
    for i, batch in enumerate(data_loader, 1):
      # Forward pass: Compute predicted y by passing x to the model
      y_pred_prob = model(batch[0].float())

      # Compute and print loss
      loss = criterion(y_pred_prob, batch[1].float().unsqueeze(1))

      # Zero gradients, perform a backward pass, and update the weights.
      optimizer.zero_grad()

      # perform a backward pass (backpropagation)
      loss.backward()

      # Update the parameters
      optimizer.step()
    
    if epoch%10 == 0:
          print('epoch: ', epoch+1,' train loss: ', loss.item())


# In[55]:


X_torch_tst = torch.from_numpy(X_scl_tst)
y_torch_tst = torch.from_numpy(y_test)
y_torch_tst = y_torch_tst.unsqueeze(1)
print(y_torch_tst.shape)


# Prediction with Final Weights

# In[56]:


y_pred = model(X_torch_tst.float())
y_pred[:5]


# In[57]:


y_pred.shape, y_test.shape


# In[58]:


type(y_pred.detach().numpy())


# In[59]:


y_pred = y_pred.detach().numpy()
y_pred.shape


# In[60]:


from sklearn.metrics import log_loss
log_loss(y_test, y_pred)


# In[ ]:





# In[ ]:





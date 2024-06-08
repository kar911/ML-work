#!/usr/bin/env python
# coding: utf-8

# In[37]:


import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler


# In[38]:


df = pd.read_csv("C:/Training/Academy/Statistics (Python)/Cases/human-resources-analytics/HR_comma_sep.csv")
dum_df = pd.get_dummies(df)
dum_df.head()


# In[39]:


X = dum_df.drop('left', axis=1)
scaler = MinMaxScaler()

y = dum_df['left'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=23,stratify=y)
X_scl_trn = scaler.fit_transform(X_train) 
X_scl_tst = scaler.transform(X_test) 


# In[40]:


X_torch = torch.from_numpy(X_scl_trn)
y_torch = torch.from_numpy(y_train)
print(X_torch.size())
print(y_torch.size())


# In[41]:


type(X_torch)


# In[42]:


X_scl_trn.shape[1]


# In[43]:


torch.manual_seed(23)
# Create a model
model = nn.Sequential(nn.Linear(in_features=X_scl_trn.shape[1], out_features=15),
                      nn.ReLU(),
                      nn.Linear(15, 8 ),
                      nn.ReLU(),
                      nn.Linear(8, 5),
                      nn.ReLU(),
                      nn.Linear(5,1),
                      nn.Sigmoid())


# In[44]:


model.parameters


# In[45]:


criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
optimizer


# In[46]:


y_pred = model(X_torch.float())
print(y_torch.shape)


# In[47]:


y_torch = y_torch.unsqueeze(1)
print(y_torch.shape)
print(y_pred.shape)


# In[48]:


y_torch.size()


# Prediction with initialized weights

# In[49]:


y_pred[:5]


# In[ ]:





# ### Initial Log Loss

# In[50]:


loss = criterion(y_pred, y_torch.float())
loss


# In[51]:


for epoch in np.arange(0,1000):
       # Forward pass: Compute predicted y by passing x to the model
       y_pred_prob = model(X_torch.float())

       # Compute and print loss
       loss = criterion(y_pred_prob, y_torch.float())
       if epoch%100 == 0:
          print('epoch: ', epoch+1,' loss: ', loss.item())

       # Zero gradients, perform a backward pass, and update the weights.
       optimizer.zero_grad()

       # perform a backward pass (backpropagation)
       loss.backward()

       # Update the parameters
       optimizer.step()
#print('epoch: ', epoch+1,' loss: ', loss.item())


# ### Training Set Log Loss

# In[52]:


loss


# In[53]:


X_torch_test = torch.from_numpy(X_scl_tst)
type(X_torch_test)


# In[54]:


y_pred_prob = model(X_torch_test.float()) # Equivalent predict_proba / predict
type(y_pred_prob)


# In[55]:


y_pred_prob


# In[56]:


y_pred_prob = y_pred_prob.detach().numpy()
type(y_pred_prob)


# In[57]:


y_pred_prob.shape


# In[58]:


y_pred_prob = y_pred_prob.reshape(y_test.shape[0],)
y_pred_prob.shape


# In[59]:


y_pred = np.where(y_pred_prob >= 0.5,1,0)

y_pred[:5]


# ### Test Set Accuracy Score

# In[60]:


print(accuracy_score(y_test,y_pred))


# ### Test Set Log Loss

# In[61]:


log_loss(y_test, y_pred_prob)


# In[ ]:





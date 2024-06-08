# In[1]:

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
# In[2]:

df = pd.read_csv("Concrete_Data.csv")
df = df.astype(float)
df.head()

# In[4]:

X = df.drop('Strength', axis=1)
y = df['Strength']
scaler = MinMaxScaler()
scalerY = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=23)
y_train = scalerY.fit_transform(y_train.values.reshape(-1, 1))
X_train = scaler.fit_transform(X_train)
#y_test = scalerY.transform(y_test.values.reshape(-1, 1))
X_test = scaler.transform(X_test)

# In[5]:

X_torch = torch.from_numpy(X_train)
y_torch = torch.from_numpy(y_train)
print(X_torch.size())
print(y_torch.size())

# In[6]:
torch.manual_seed(23)
# Create a model
model = nn.Sequential(nn.Linear(in_features=X_train.shape[1], out_features=7),
                      nn.ReLU(),
                      nn.Linear(7,3),
                      nn.ReLU(),
                      nn.Linear(3,1),
                      nn.ReLU())

# In[7]:

criterion = torch.nn.MSELoss()
# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.2)
optimizer

# Initail Weights

# In[8]:


model[0].weight


# In[9]:


model


# In[10]:


y_pred = model(X_torch.float())
#y_torch = y_torch.unsqueeze(1)
print(y_torch.shape)
print(y_pred.shape)


# In[11]:


# Gradient Descent

for epoch in np.arange(0,1000):
   # Forward pass: Compute predicted y by passing x to the model
   y_pred = model(X_torch.float())

   # Compute and print loss
   loss = criterion(y_pred, y_torch.float())
   #print('epoch: ', epoch+1,' loss: ', loss.item())

   # Zero gradients, perform a backward pass, and update the weights.
   optimizer.zero_grad()

   # perform a backward pass (backpropagation)
   loss.backward()

   # Update the parameters
   optimizer.step()
   if epoch % 100 == 0:
       print('epoch: ', epoch+1,' loss: ', loss.item())


# In[12]:


#model = model.eval()
X_torch_test = torch.from_numpy(X_test)
y_pred = model(X_torch_test.float())
y_pred = y_pred.detach().numpy()
y_pred = y_pred.reshape(y_test.shape[0],)
y_pred[:5]

# In[13]:

y_pred = y_pred.reshape(-1,1)
y_pred_orig = scalerY.inverse_transform(y_pred)
y_pred_orig[:5]

# In[14]:
y_test.iloc[:5]

# ### Test Set R2 Score

# In[15]:
print(r2_score(y_test,y_pred_orig))

# ### Test Set MSE

# In[16]:

print(mse(y_test,y_pred_orig))

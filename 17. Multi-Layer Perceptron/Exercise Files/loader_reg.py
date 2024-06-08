# In[1]:

from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

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

# In[27]:


X_torch = torch.from_numpy(X_train)
y_torch = torch.from_numpy(y_train)
print(X_torch.size())
print(y_torch.size())


joint_dataset = TensorDataset(X_torch.float(), 
                              y_torch.float())

type(joint_dataset)

torch.manual_seed(23)
data_loader = DataLoader(dataset=joint_dataset, 
                         batch_size=16, shuffle=True)

# torch.manual_seed(2022)
# # Create a model
# model = nn.Sequential(nn.Linear(in_features=X_train.shape[1], out_features=7),
#                       nn.ReLU(),
#                       nn.Linear(7,3),
#                       nn.ReLU(),
#                       nn.Linear(3,1),
#                       nn.ReLU())

## OR

class MLPRegressor(torch.nn.Module):    
    def __init__(self, num_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features=num_features, 
                                  out_features=7)
        self.linear2 = nn.Linear(7,3)
        self.linear3 = nn.Linear(3,1)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.linear3(x)
        output = self.act(x)
        return output
    
torch.manual_seed(23)
model = MLPRegressor(num_features=X_train.shape[1])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
optimizer


# Prediction with Default Weights

# In[16]:
y_torch.size()

# In[17]:
y_pred = model(X_torch.float())
print(y_torch.shape)
print(y_pred.shape)

# In[18]:

y_pred[:5]

# In[19]:
y_torch.float().size()

# ### Initial Log Loss

# In[20]:


loss = criterion(y_pred, y_torch.float())
loss


# In[21]:


X_torch_test = torch.from_numpy(X_test)
type(X_torch_test)


# In[22]:


y_torch_test = torch.from_numpy(y_test.values)
y_torch_test.size()


### A specific batch calculation
#batch = next(enumerate(data_loader))
#y_pred_prob = model(batch[1][0].float())
#loss = criterion(y_pred_prob, batch[1][1].float())

# Gradient Descent

for epoch in np.arange(0,100):
    for i, batch in enumerate(data_loader):
      # Forward pass: Compute predicted y by passing x to the model
      y_pred_prob = model(batch[0].float())

      # Compute and print loss
      loss = criterion(y_pred_prob, batch[1].float())

      # Zero gradients, perform a backward pass, and update the weights.
      optimizer.zero_grad()

      # perform a backward pass (backpropagation)
      loss.backward()

      # Update the parameters
      optimizer.step()
    
    if epoch%10 == 0:
          print('epoch: ', epoch+1,' train loss: ', loss.item())


# In[55]:


X_torch_tst = torch.from_numpy(X_test)
y_torch_tst = torch.from_numpy(y_test.values)
print(y_torch_tst.shape)


# Prediction with Final Weights
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

from ISLP import load_data
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

BrainCancer = load_data('BrainCancer')
print(BrainCancer.columns)

X = BrainCancer.drop(['time','status'], axis=1)
y = BrainCancer['status']

print(X.isnull().sum())
X['diagnosis'].fillna('Meningioma', inplace=True)
print(X.isnull().sum())

X = pd.get_dummies(X)
y = y.values

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=23,stratify=y)
X_scl_trn = scaler.fit_transform(X_train) 
X_scl_tst = scaler.transform(X_test) 

X_torch = torch.from_numpy(X_scl_trn)
y_torch = torch.from_numpy(y_train)
print(X_torch.size())
print(y_torch.size())

joint_dataset = TensorDataset(X_torch.float(), y_torch.float())

type(joint_dataset)

torch.manual_seed(23)
data_loader = DataLoader(dataset=joint_dataset, 
                         batch_size=16, shuffle=True)

class MLPClassifier(torch.nn.Module):    
    def __init__(self, num_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features=num_features, 
                                  out_features=7)
        self.linear2 = nn.Linear(7,6)
        self.linear3 = nn.Linear(6,3)
        self.linear4 = nn.Linear(3,1)
        self.act1 = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.linear3(x)
        x = self.act(x)
        output = self.linear4(x)
        return output    

torch.manual_seed(23)
model = MLPClassifier(num_features=X_train.shape[1])


criterion = torch.nn.BCEWithLogitsLoss()
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

y_pred = y_pred.squeeze(1)
loss = criterion(y_pred, y_torch.float())
loss


# In[21]:


X_torch_test = torch.from_numpy(X_scl_tst)
type(X_torch_test)


# In[22]:


y_torch_test = torch.from_numpy(y_test)
y_torch_test.size()


### A specific batch calculation
#batch = next(enumerate(data_loader,1))
#y_pred_prob = model(batch[1][0].float())
#loss = criterion(y_pred_prob, batch[1][1].float())

# Gradient Descent

for epoch in np.arange(0,1000):
    for i, batch in enumerate(data_loader, 1):
      # Forward pass: Compute predicted y by passing x to the model
      y_pred_prob = model(batch[0].float())

      # Compute and print loss
      y_pred_prob = y_pred_prob.squeeze(1)
      loss = criterion(y_pred_prob, batch[1].float())

      # Zero gradients, perform a backward pass, and update the weights.
      optimizer.zero_grad()

      # perform a backward pass (backpropagation)
      loss.backward()

      # Update the parameters
      optimizer.step()
    
    if epoch%100 == 0:
          print('epoch: ', epoch+1,' train loss: ', loss.item())


# In[55]:


X_torch_tst = torch.from_numpy(X_scl_tst)
y_torch_tst = torch.from_numpy(y_test)
y_torch_tst = y_torch_tst.unsqueeze(1)
print(y_torch_tst.shape)


# Prediction with Final Weights

y_wt_sum = model(X_torch_test.float()) 
sigmoid = nn.Sigmoid()
pred_proba = sigmoid(y_wt_sum)
pred_proba

print(pred_proba.size())

pred_proba = pred_proba.squeeze(1)
y_pred = np.where(pred_proba.detach().numpy()>0.5,1,0 )

y_pred[:5]


# ### Test Set Accuracy Score

print(accuracy_score(y_test,y_pred))

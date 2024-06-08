# In[1]:
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler
# In[2]:
df = pd.read_csv("Glass.csv")
df.head()
# In[3]:
df['Type'].unique()
# In[4]:
dum_df = pd.get_dummies(df)
dum_df.head()
# In[5]:
X = dum_df.iloc[:,:-6]
X.head()


# In[6]:


y = dum_df.iloc[:,-6:]
y.head()


# In[7]:


le = LabelEncoder()
le_y = le.fit_transform(df['Type'])


# In[8]:


y = y.values  # converting to numpy


# In[9]:


scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=23,stratify=le_y)
X_scl_trn = scaler.fit_transform(X_train) 
X_scl_tst = scaler.transform(X_test) 


# In[10]:


X_torch = torch.from_numpy(X_scl_trn)
y_torch = torch.from_numpy(y_train)
print(X_torch.size())
print(y_torch.size())


# In[11]:


type(X_torch)


# In[12]:


X_scl_trn.shape[1]


# In[13]:


torch.manual_seed(2022)
# Create a model
model = nn.Sequential(nn.Linear(in_features=X_scl_trn.shape[1], out_features=7),
                      nn.ReLU(),
                      nn.Linear(7, 6),
                      nn.ReLU(),
                      nn.Linear(6,3),
                      nn.ReLU(),
                      nn.Linear(3,6))

# In[14]:

print(model)

# In[15]:

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
optimizer

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


X_torch_test = torch.from_numpy(X_scl_tst)
type(X_torch_test)


# In[22]:


y_torch_test = torch.from_numpy(y_test)
y_torch_test.size()


# ### Training Loop

# In[23]:


for epoch in np.arange(0,1000):
       # Forward pass: Compute predicted y by passing x to the model
       y_pred_prob = model(X_torch.float())
       y_pred_prob_test = model(X_torch_test.float())
        
       # Compute and print loss
       loss = criterion(y_pred_prob, y_torch.float())
       tst_loss = criterion(y_pred_prob_test, y_torch_test.float() )
       if epoch%100 == 0:
          print('epoch: ', epoch+1,' train loss: ', loss.item(), " test loss:", tst_loss.item())

       # Zero gradients, perform a backward pass, and update the weights.
       optimizer.zero_grad()

       # perform a backward pass (backpropagation)
       loss.backward()

       # Update the parameters
       optimizer.step()
#print('epoch: ', epoch+1,' train loss: ', loss.item(), " test loss:", tst_loss.item())


# ### Training Set Log Loss after training loop execution

# In[24]:


loss


# In[25]:


#torch.set_printoptions(precision=3, sci_mode=False)


# In[26]:


y_wt_sum = model(X_torch_test.float()) 
softmax = nn.Softmax(dim=1)
pred_proba = softmax(y_wt_sum)
pred_proba


# In[27]:


pred_proba.size()


# In[28]:


y_pred = np.argmax(pred_proba.detach().numpy(), axis=1 )

y_pred[:5]


# In[29]:


y_test_lbl = np.argmax(y_test, axis=1)
y_test_lbl[:5]


# ### Test Set Accuracy Score

# In[30]:


print(accuracy_score(y_test_lbl,y_pred))


# In[ ]:





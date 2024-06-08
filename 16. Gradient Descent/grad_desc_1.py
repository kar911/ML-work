import numpy as np

x = np.array([0.2, 1.2,1, 1.4, -1.5, 0.5, -0.5])
y = np.array([5.6, 8.6, 8. , 9.2, 0.5, 6.5, 3.5])

eta = 0.2
i_w = 0.5
i_b = -0.5

w, b = i_w, i_b
for epoch in range(1, 101):
    
    y_pred = w*x + b
    L = (1/2*len(x))*np.sum((y-y_pred)**2)
    print("Epoch:",epoch)
    print("Loss =", L)
    if L < 0.0001: 
        break
    
    db = -(1/len(x))*np.sum(y-y_pred)
    dw = -(1/len(x))*np.sum((y-y_pred)*x)
    
    new_w = w - eta*dw 
    new_b = b - eta*db 
    print(new_w, new_b)
    w, b = new_w, new_b

###############################
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt 

pizza = pd.read_csv("pizza.csv")
scaler = MinMaxScaler().set_output(transform='pandas')
piz_scaled = scaler.fit_transform(pizza)

x = piz_scaled['Promote'].values
y = piz_scaled['Sales'].values

eta = 0.05
i_w = 0.5
i_b = -0.5
losses = []
w, b = i_w, i_b
epochs = np.arange(1, 100001)
for epoch in epochs:
    
    y_pred = w*x + b
    L = (1/2*len(x))*np.sum((y-y_pred)**2)
    losses.append(L)
    print("Epoch:",epoch)
    print("Loss =", L)
    if L < 0.0001: 
        break
    
    db = -(1/len(x))*np.sum(y-y_pred)
    dw = -(1/len(x))*np.sum((y-y_pred)*x)
    
    new_w = w - eta*dw 
    new_b = b - eta*db 
    print(new_w, new_b)
    w, b = new_w, new_b
    
plt.scatter(epochs, losses, c='red')
plt.plot(epochs, losses)
plt.title("Learning Curve with eta="+str(eta))
plt.show()

####################################################

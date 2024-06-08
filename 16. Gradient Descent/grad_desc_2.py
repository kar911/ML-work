import numpy as np

x = np.array([0.2, 1.2,1, 1.4, -1.5, 0.5, -0.5])
y = np.array([5.6, 8.6, 8. , 9.2, 0.5, 6.5, 3.5])


eta = 0.2
i_w = 0.5
i_b = -0.5

w, b = i_w, i_b
for epoch in range(1, 11):
    
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


#### SGD
eta = 0.2
i_w = 0.5
i_b = -0.5

w, b = i_w, i_b
for epoch in range(1, 11):
    
    y_pred = w*x + b
    L = (1/2*len(x))*np.sum((y-y_pred)**2)
    print("Epoch:",epoch)
    print("Loss =", L)
    if L < 0.0001: 
        break
    
    for i in range(0,len(x)):
        db = -(y[i]-y_pred[i])/2
        dw = -((y[i]-y_pred[i])*x[i])/2
    
        new_w = w - eta*dw 
        new_b = b - eta*db 
        print(new_w, new_b)
        w, b = new_w, new_b
   
    
        
        
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

actual = np.array([0,0,0,1,0,1,1,0,1,0,0,1])
pred = np.array([0,0,0,0,1,1,0,0,1,1,0,0])

print(confusion_matrix(actual, pred))
print(accuracy_score(actual, pred))

print(recall_score(actual, pred, pos_label=0))
print(recall_score(actual, pred, pos_label=1))
print(recall_score(actual, pred, average='macro'))
print(recall_score(actual, pred, average='weighted'))

print(precision_score(actual, pred, pos_label=0))
print(precision_score(actual, pred, pos_label=1))
print(precision_score(actual, pred, average='macro'))
print(precision_score(actual, pred, average='weighted'))

print(f1_score(actual, pred, pos_label=0))
print(f1_score(actual, pred, pos_label=1))
print(f1_score(actual, pred, average='macro'))
print(f1_score(actual, pred, average='weighted'))

print(classification_report(actual, pred))

    



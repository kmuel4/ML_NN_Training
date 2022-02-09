# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 13:43:45 2021

@author: kurtm
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
#from sklearn.multiclass import OneVsOneClassifier
#from sklearn import model_selection
import neurolab as nl


# Input file containing data
input_file = 'svm_income_data.txt'

# Read the data
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 1000
q_count = 0

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            print('break here');
            break

        if '?' in line:
            q_count += 1
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1


print(q_count, count_class1, count_class2)

#print(X.shape)
# Convert to numpy array
X = np.array(X)
print(X.shape)

label_encoder = [] 
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit(): 
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

print(X[:4,:])	
print(X_encoded[:4,:])	

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

print(X[:4,:])
print(y[:4])

y = np.expand_dims(y, axis=1)
minmax_X = []
X1, y1 = X.tolist(), y.tolist();
for i,item in enumerate(X[0]):
	minmax_X.append([X[:,i].min(),X[:,i].max()])
print(minmax_X)

nn = nl.net.newff(minmax_X, [15, 6, 1])
nn.trainf = nl.train.train_gd
error_progress = nn.train(X1, y1, epochs=10, goal=0.01)

plt.figure()
plt.plot(error_progress)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Training progress')
plt.show()

input_data = [['52', 'Self-emp-not-inc', '209642', 'HS-grad', '9', 'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White', 'Male', '0', '0', '45', 'United-States'], \
			['42', 'Private', '159449', 'Bachelors', '13', 'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White', 'Male', '5178', '0', '40', 'United-States']]

input_data = np.array(input_data)
input_data_encoded = np.empty(input_data.shape)

count = 0
for i, item in enumerate(input_data[0]):
    if item.isdigit():
        input_data_encoded[:,i] = input_data[:,i]
    else:
        input_data_encoded[:,i] = label_encoder[count].transform(input_data[:,i])
        count += 1 
input_data_encoded = input_data_encoded.astype(int)
print(input_data_encoded)

for t in input_data_encoded.tolist():
    predicted_class = nn.sim([t])[0]
    print(t, '-->', predicted_class)

# model_selection.train_test_split(X, y)

# accuracy = 100.0 * (y == y1).sum() / X.shape[0]
# print("Accuracy =", round(accuracy, 2), "%")

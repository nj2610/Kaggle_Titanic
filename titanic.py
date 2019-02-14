#Titanic

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('titanic_clean.csv')
X_train = train.iloc[0:1308,[1,4,5,6,7,9,11]]
Y_train = train.iloc[0:1308,2]

test = pd.read_csv('test.csv')
X_test = test.iloc[:,[1,3,4,5,6,8,10]]

X_train['age'] = X_train ['age'].fillna(X_train['age'].median())

X_train.loc[X_train['sex'] == 'male','sex'] = 0
X_train.loc[X_train['sex'] == 'female','sex'] = 1

X_train.loc[X_train['embarked'] == 'S','embarked'] = 2
X_train.loc[X_train['embarked'] == 'C','embarked'] = 1
X_train.loc[X_train['embarked'] == 'Q','embarked'] = 0

X_train['embarked'] = X_train ['embarked'].fillna(X_train['embarked'].median())

X_test['Age'] = X_test['Age'].fillna(X_test['Age'].median())

X_test.loc[X_test['Sex'] == 'male','Sex'] = 0
X_test.loc[X_test['Sex'] == 'female','Sex'] = 1

X_test.loc[X_test['Embarked'] == 'S','Embarked'] = 0
X_test.loc[X_test['Embarked'] == 'C','Embarked'] = 1
X_test.loc[X_test['Embarked'] == 'Q','Embarked'] = 2

X_test['Embarked'] = X_test ['Embarked'].fillna(X_test['Embarked'].median())
X_test['Fare'] = X_test ['Fare'].fillna(X_test['Fare'].median())

null_columns=X_train.columns[X_train.isnull().any()] 
X_train[null_columns].isnull().sum()
print(X_train[X_train["fare"].isnull()][null_columns])
X_train = X_train.drop([1225])
Y_train = Y_train.drop([1225])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [0])
onehotencoder = OneHotEncoder(categorical_features = [1])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()
onehotencoder = OneHotEncoder(categorical_features = [0])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_test = onehotencoder.fit_transform(X_test).toarray()

#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.fit_transform(X_test)

#importing keras
import keras
from keras.models import Sequential    #Used to initialise neural net
from keras.layers import Dense         #Used for hidden layers

'''Two methods for initialising 1. defining as sequence of layers
2. defining as graph
here we are using sequence of layers'''
#Initialising ANN
classifier = Sequential()

'''Generally rectifier activation function is preferred for hidden layer and in output layer sigmoid activation function
is preferred as it provides probabilities'''

'''TIP: Choose number of nodes of hidden layer equal to average f number of nodes in input and out put layers.
Or we can use parameter tuning'''

'''Arguements:
    output_dim : No of nodes in hidden layer , init : weight initial,  activation: activation function used
    in first step input dim is a compulsory arguement because we have just intialised the neural net, 
    input_dim:no of independent variables'''
    
'''If dependent variable has more than two categories then activation fn is softmax for output layer'''

#Adding input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu',input_dim = 9))

#Adding second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

'''Compiling means applying stochastic gradient descent on whole ANN
Arguements :
    optimizer: adam is efficient form of stochastic gradient descent
    loss : binary for two output, for more than two categorical
    metrics : criterion for improving models performence'''
#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

'''epochs means number of rounds
   also we have to add the batch size '''

#Fitting ANN to the dataset
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 200)

#Making Prediction and evaluating model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = ((y_pred >0.5)*1)

predictions = pd.DataFrame(y_pred, columns=['Survived']).to_csv('prediction.csv')

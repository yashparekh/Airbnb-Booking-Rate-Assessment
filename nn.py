# -*- coding: utf-8 -*-
"""
Created on Mon May 07 22:52:08 2018

@author: ypare
"""

# =============================================================================
# Import Libraries
# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================
# Import Dataset
# =============================================================================

df_X = pd.read_csv("clean_training.csv")
a=np.ones(99991)
df_X['isTraining']=a
df_X_test=pd.read_csv("clean_test.csv")
b=np.zeros(12206)
df_X_test['isTraining']=b
df_X=df_X.append(df_X_test)
df_am=pd.read_csv('amenities3.csv')
df_X=df_X.reset_index()
df_X1=df_X.iloc[:,[3,5,9,10,11,12,13,15,16,23,26,27,28,29,39,42,45,49,55,56,60]]
df_X=pd.concat([df_X1,df_am], axis=1)
X=df_X.iloc[:,].values
df_y=pd.read_csv("clean_train_y.csv")
y=df_y.iloc[:,[2]].values

# =============================================================================
# Step 4: Handling Missing Values. Missing values are either replaced with mean or median 
# =============================================================================

from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN', strategy='median')
imp.fit(X[:, [2,5,8,17,18]]) 
X[:, [2,5,8,17,18]]=imp.transform(X[:, [2,5,8,17,18]])
imp.fit(y[:,])
y[:,]=imp.transform(y[:,])

# =============================================================================
# Handling Categorical data
# =============================================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#bed_type
label_encoder_3=LabelEncoder()
label_encoder_3.fit(X[:,3])
X[:,3]=label_encoder_3.transform(X[:,3])

#cancellation_policy
label_encoder_6=LabelEncoder()
label_encoder_6.fit(X[:,6])
X[:,6]=label_encoder_6.transform(X[:,6])

#city_name
label_encoder_7=LabelEncoder()
label_encoder_7.fit(X[:,7])
X[:,7]=label_encoder_7.transform(X[:,7])

#host_has_profile_pic
label_encoder_10=LabelEncoder()
label_encoder_10.fit(X[:,10])
X[:,10]=label_encoder_10.transform(X[:,10])

#host_identity_verified
label_encoder_11=LabelEncoder()
label_encoder_11.fit(X[:,11])
X[:,11]=label_encoder_11.transform(X[:,11])

#host_is_superhost
label_encoder_12=LabelEncoder()
label_encoder_12.fit(X[:,12])
X[:,12]=label_encoder_12.transform(X[:,12])

#instant_bookable
label_encoder_14=LabelEncoder()
label_encoder_14.fit(X[:,14])
X[:,14]=label_encoder_14.transform(X[:,14])

#is_location_exact
label_encoder_15=LabelEncoder()
label_encoder_15.fit(X[:,15])
X[:,15]=label_encoder_15.transform(X[:,15])

#license
label_encoder_16=LabelEncoder()
label_encoder_16.fit(X[:,16])
X[:,16]=label_encoder_16.transform(X[:,16])

#property_type
label_encoder_19=LabelEncoder()
label_encoder_19.fit(X[:,19])
X[:,19]=label_encoder_19.transform(X[:,19])

#room_type
label_encoder_20=LabelEncoder()
label_encoder_20.fit(X[:,20])
X[:,20]=label_encoder_20.transform(X[:,20])

onehot_encoder=OneHotEncoder(categorical_features=[3,6,7,10,11,12,14,15,16,19,20], sparse=False)
onehot_encoder.fit(X)
X=onehot_encoder.transform(X)
X=np.delete(X,onehot_encoder.feature_indices_[:-1],1)


Xt=X[:99991,]
X_test=X[99991:,]                

# =============================================================================
# Split data into training and test
# =============================================================================

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(Xt,y,test_size=0.3, random_state=0)

# =============================================================================
# Building artificial neural network
# =============================================================================

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 91 , init = 'uniform', activation = 'relu', input_dim = 274))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = 137, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 91, init = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 2)

# =============================================================================
# Prediction using Test Data
# =============================================================================

y_pred=classifier.predict(X_val)
y_pred_prob=classifier.predict_proba(X_val)
y_pred2=(y_pred_prob>0.5)
j=0
y_pred2=y_pred2[:,1]
y_pred3=[]
while (j<29998):
    for i in y_pred2:
        if i==True:
            print(i)
            y_pred3[j]=1    
        else:
            y_pred3[j]=0
        j=j+1

    
y_final=nb.predict(X_test)

# =============================================================================
# Classification Metrics
# =============================================================================

from sklearn.metrics import accuracy_score
accuracy_score(y_val, y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_val, y_pred)

# =============================================================================
# Write to csv file
# =============================================================================

final=pd.DataFrame(y_final)
final.to_csv('high_booking_rate_pred2.csv', sep=',', encoding='utf-8', header = True)
# -*- coding: utf-8 -*-

#Data Decription
'''Data were extracted from images that were taken from genuine and forged banknote-like specimens. 
For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels.  
Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.
'''

import pandas as pd 

df = pd.read_csv("D:/AppliedAICourse/Projects/Machine Learning/Bank_Note_Authentication/BankNote_Authentication.csv")
df.head()

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

### Implement logistic regression classifier
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(solver='lbfgs')
classifier.fit(X_train,y_train)

## Prediction
y_pred=classifier.predict(X_test)

### Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
score

### Create a Pickle file using serialization 
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

#predicting the output
#classifier.predict([[2,3,4,1]])
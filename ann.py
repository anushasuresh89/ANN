import pandas as pd
import numpy as np

#import the data set
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:, 3:13]
y = data.iloc[:, 13]

#preprocessing:
#Geography
X = pd.concat([X.drop('Geography', axis=1), pd.get_dummies(X['Geography'])], axis=1)
#Gender
X = pd.concat([X.drop('Gender', axis=1), pd.get_dummies(X['Gender'])], axis=1)


#splitting into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Artificial Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=len(X.columns)))
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='relu'))

#compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

classifier.save('model_v1.h5')


# Load and run the test set predictions
# from keras.models import load_model
# classifier = load_model('model_v1.h5')

# y_pred = classifier.predict(X_test)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print("y_test")
# print(y_test)
# print("y_pred")
# print(y_pred)
# print(cm)












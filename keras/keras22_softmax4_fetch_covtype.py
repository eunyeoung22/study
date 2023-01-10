import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

# print(datasets.DESCR)
# print(datasets.feature_names)
# # print(x.shape, y.shape) #(581012, 54) (581012,)
# print(np.unique(y, return_counts=True))
# # #(array([1, 2, 3, 4, 5, 6, 7])
# # #array[211840, 283301,  35754,   2747,   9493,  17367,  20510]
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Initialize the OneHotEncoder
ohe = OneHotEncoder()

# Fit the OneHotEncoder on the target variable 'y'
y = ohe.fit_transform(y.reshape(-1,1)).toarray()

# Print the new shape of the target variable 'y'
print(y.shape)
print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(
                x, y, shuffle=True,
                random_state=123,
                test_size=0.2,
                stratify=y
)
#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='linear', input_shape = (54,)))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_accuracy',
                              mode='max',
                              patience=5,
                              verbose=2)
model.fit(x_train, y_train, epochs=10, batch_size=32,
          callbacks=[earlyStopping],validation_split=0.2, verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy : ', accuracy)


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=0)
print(y_predict)
y_test = np.argmax(y_test, axis=0)
print(y_test)
# acc = accuracy_score(y_test, y_predict)
# print(acc)

"""
loss:  0.664304256439209
accuracy :  0.7103431224822998
"""
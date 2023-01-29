#01 ~14까지 conv1D로 만들기(boston ~ cifa100)

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=123
)


scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)#(354, 13) (152, 13)
x_train = x_train.reshape(354,13,1)
x_test = x_test.reshape(152,13,1)

print(x_train.shape, x_test.shape)

# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))

# 2. 모델 구성(순차형)
model = Sequential()
model.add(Conv1D(100, 2, input_shape =(13,1), activation = 'relu'))
model.add(LSTM(100,  activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(200, activation= 'linear'))
model.add(Dense(300, activation= 'linear'))
model.add(Dense(400, activation= 'linear'))
model.add(Dense(500, activation= 'linear'))
model.add(Dense(400, activation= 'linear'))
model.add(Dense(300, activation= 'linear'))
model.add(Dense(200, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(1, activation= 'linear'))
model.summary()

#3.컴파일, 훈련
model.compile(loss='mse' , optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=300, batch_size=50, validation_split=0.3)

#4.평가, 예측
mae, mse = model.evaluate(x_test, y_test)
print('mae : ', mae)
print('mse : ', mse)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
print("예측 값 : ", y_predict)

r2 = r2_score(y_test, y_predict)
print("***********************************")
print("R2 : ", r2)
print("***********************************")


"""
R2 :  0.6099399653377051

R2 :  0.6311076332724015
"""

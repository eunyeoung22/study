# 1. train 0.7 이상
# 2. R2 : 0.8 이상 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import sklearn as sk
print(sk.__version__)

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=123
)

print(x)
print(x.shape)
print(y)
print(y.shape)

print(dataset.feature_names)

print(dataset.DESCR)

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(60))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mae' , optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=3000, batch_size=50)

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

# print("=========================")
# print(y_test)
# print(y_predict)
# print("=========================")

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))

def adj_r2_score(y_test, y_predict, p=x.shape[1]):
    return 1-(1-r2_score(y_test, y_predict)) * (len(y_test)-1) / (len(y_test) - p - 1)

print("***********************************")
print("RMSE : ", RMSE(y_test, y_predict))
print("***********************************")

r2 = r2_score(y_test, y_predict)
print("***********************************")
print("R2 : ", r2)
print("***********************************")


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))


x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=123
)


# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=13, activation= 'linear'))
model.add(Dense(100, activation= 'sigmoid'))
model.add(Dense(200, activation= 'linear'))
model.add(Dense(300, activation= 'linear'))
model.add(Dense(400, activation= 'linear'))
model.add(Dense(500, activation= 'linear'))
model.add(Dense(400, activation= 'linear'))
model.add(Dense(300, activation= 'linear'))
model.add(Dense(200, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(1, activation= 'linear'))

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

# print("=========================")
# print(y_test)
# print(y_predict)
# print("=========================")

# 변경 전
# mae :  57.251380920410156
# mse :  5.495046615600586
# R2 :  0.2916895715796184
# 변경 후 
# MinMaxScaler --> 싸이킷런 데이터 전처리 방법(다양한 데이터를 부동소수점으로 만들어 정리)
# mae :  33.76993942260742
# mse :  4.570913314819336
# R2 :  0.5822004358015083
# StandardScaler --> 싸이킷런 데이터 전처리 방법(편중된 데이터를 정리)
# mae :  29.02393913269043
# mse :  3.6788923740386963
# R2 :  0.6409176542899131

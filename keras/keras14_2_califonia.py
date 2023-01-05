
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets= fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x)
# print(x.shape) # (20640, 8)

# print(y)
# print(y.shape) # (20640, 8)

# print('결과: ', datasets.feature_names)

# print('결과: ', datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.9, shuffle=True, random_state=123
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(1))

#3.컴파일, 훈련
import time
model.compile(loss='mse' , optimizer='adam', metrics=['mae'])
start = time.time()
model.fit(x_train, y_train, epochs=3000, batch_size=500)
end = time.time()

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

# def adj_r2_score(y_test, y_predict, p=x.shape[1]):
#     return 1-(1-r2_score(y_test, y_predict)) * (len(y_test)-1) / (len(y_test) - p - 1)

print("***********************************")
print("RMSE : ", RMSE(y_test, y_predict))
print("걸리는 시간 : ", end - start)
print("***********************************")

r2 = r2_score(y_test, y_predict)
print("***********************************")
print("R2 : ", r2)
print("걸리는 시간 : ", end - start)
print("***********************************")


"""
1. CPU
RMSE :  0.7946094173495656
걸리는 시간 :  469.38778018951416

R2 :  0.5468672671923229
걸리는 시간 :  469.38778018951416

2. GPU
RMSE :  0.7947311636072167
걸리는 시간 :  468.9245343208313

R2 :  0.5467284028912904
걸리는 시간 :  468.9245343208313




"""
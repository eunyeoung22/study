import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

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
model.compile(loss='mae' , optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=3000, batch_size=50, validation_split=0.3)

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print("예측 값 : ", y_predict)

# print("=========================")
# print(y_test)
# print(y_predict)
# print("=========================")

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))


print("***********************************")
print("RMSE : ", RMSE(y_test, y_predict))
print("***********************************")

r2 = r2_score(y_test, y_predict)
print("***********************************")
print("R2 : ", r2)
print("***********************************")

"""
1.
RMSE :  5.163125996943811
R2 :  0.6701905301890563

2.
RMSE :  5.328105266084274 #relu
R2 :  0.6487767416120316

3.
RMSE :  5.042976070204978
R2 :  0.6853617706610756
"""
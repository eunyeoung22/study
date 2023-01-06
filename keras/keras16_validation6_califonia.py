
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets= fetch_california_housing()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.9, shuffle=True, random_state=123
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=8, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(500, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(50, activation= 'linear'))
model.add(Dense(1, activation= 'linear'))

#3.컴파일, 훈련
import time
model.compile(loss='mse' , optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=300, batch_size=50)


#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)


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
1. CPU
loss :  [1.3962271213531494, 0.937720537185669]

RMSE :  1.1816205764085699    #relu 3회 사용
R2 :  -0.0020147390357134753

RMSE :  1.1830157025661103    #relu 2회 사용
R2 :  -0.004382270991142079


"""
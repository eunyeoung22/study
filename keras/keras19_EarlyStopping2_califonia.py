
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
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_loss',
                              mode='min',
                              patience = 100, 
                              restore_best_weights=True
                              )
hist = model.fit(x_train, y_train, epochs=3000, batch_size=50, validation_split=0.2,
                 callbacks = [earlyStopping], verbose=1)


#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("=============================================")
print(hist)#(<keras.callbacks.History object at 0x000002965CCF2790>)
print("=============================================")
print(hist.history)
print("=============================================")
print(hist.history['loss'])
print("=============================================")
print(hist.history['val_loss'])
print("=============================================")


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


import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) #그래프 사이즈
plt.plot(hist.history['loss'], c = 'red', marker = '.', 
        label='loss') # c : 그래프 선 color / label : 그래프 선 이름
plt.plot(hist.history['val_loss'], c = 'blue', marker = '.', 
        label = 'val loss')
plt.grid() #격자
plt.xlabel('epochs') #x축
plt.ylabel('loss') #y축
plt.title('califonia loss') # 그래프 타이틀
plt.legend() # 범례(알아서 빈곳에 현출)
# plt.legend(loc='upper right') #범례(그래프 오른쪽)
# plt.legend(loc='upper left') #범례(그래프 왼쪽)
plt.show() # 그래프 보여줘


"""
loss :  [1.285967469215393, 0.9281428456306458]
***********************************
RMSE :  1.134004911803205
***********************************
***********************************
R2 :  0.07711435462414407
***********************************
"""
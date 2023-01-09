
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#1. 데이터
datasets= load_diabetes()
x = datasets.data
y = datasets.target

# # print(x)
# # print(x.shape) # (442, 10)

# print(y)
# print(y.shape) # (442,)

print('결과: ', datasets.feature_names)

print('결과: ', datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle = True, test_size=0.4)
x_test, x_validation, y_test, y_validation = train_test_split(
    x_test, y_test, random_state=66, shuffle = True, test_size=0.5)

# 2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=10, activation= 'sigmoid'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(500, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(200, activation= 'linear'))
model.add(Dense(1, activation= 'linear'))

#3.컴파일, 훈련
model.compile(loss='mse' , optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_loss',
                              mode='min',
                              patience = 100, 
                              restore_best_weights=True
                              )
hist = model.fit(x_train, y_train, epochs=1500, batch_size=32, validation_split= 0.25,
                callbacks=[earlyStopping], verbose=1)

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))

# def adj_r2_score(y_test, y_predict, p=x.shape[1]):
#     return 1-(1-r2_score(y_test, y_predict)) * (len(y_test)-1) / (len(y_test) - p - 1)

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
plt.title('diabets loss') # 그래프 타이틀
plt.legend() # 범례(알아서 빈곳에 현출)
# plt.legend(loc='upper right') #범례(그래프 오른쪽)
# plt.legend(loc='upper left') #범례(그래프 왼쪽)
plt.show() # 그래프 보여줘


"""
1. 
RMSE :  55.76194151066935 # 1 : sigmoid
                            2~6 :linear
R2 :  0.41316194017455843

RMSE :  54.58584732508851
R2 :  0.4571201541426064

EarlyStopping 
loss :  [3171.578369140625, 46.11053466796875]
***********************************
RMSE :  56.31677337911578
***********************************
***********************************
R2 :  0.4752787459502814
"""

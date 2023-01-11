
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터
path = './_data/bike/' #  데이터 위치
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)

# print(train_csv.shape)  #(10886, 11)
# print(test_csv.shape)  #(6493, 8)
# print(submission.shape) #(6493, 1)
# print(train_csv.info())
# print(test_csv.info())

x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv['count']
print(y)
print(y.shape) #(10,886,0) 


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9 , shuffle=True, random_state=123
)
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# # print(x)
# # print(type(x)) #<class 'numpy.ndarray'>
# # print('최소값 : ', np.min(x))
# # print('최대값 : ',np.max(x))

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8, activation = 'linear'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'linear'))
model.add(Dense(100, activation = 'linear'))
model.add(Dense(90, activation = 'linear'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(50, activation = 'linear'))
model.add(Dense(40, activation = 'linear'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
import time
model.compile(loss='mse' , optimizer='adam')

hist = model.fit(x_train, y_train, epochs=50, batch_size=10, validation_split=0.25)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('y_predict : ' , y_predict)

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))**0.5

    

print("***********************************")
print("RMSE : ", RMSE(y_test, y_predict))
print("***********************************")

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6)) #그래프 사이즈
# plt.plot(hist.history['loss'], c = 'red', marker = '.', 
#         label='loss') # c : 그래프 선 color / label : 그래프 선 이름
# plt.plot(hist.history['val_loss'], c = 'blue', marker = '.', 
#         label = 'val loss')
# plt.grid() #격자
# plt.xlabel('epochs') #x축
# plt.ylabel('loss') #y축
# plt.title('kaggle loss') # 그래프 타이틀
# plt.legend() # 범례(알아서 빈곳에 현출)
# # plt.legend(loc='upper right') #범례(그래프 오른쪽)
# # plt.legend(loc='upper left') #범례(그래프 왼쪽)
# plt.show() # 그래프 보여줘

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

submission['count'] = y_submit
print(submission)
submission.to_csv(path + 'submission_01111730.csv')


"""
RMSE :  168.51801356609832 #relu
RMSE :  149.02541250872042
RMSE :  149.10960514117627 #last
RMSE :  150.33808468361812


loss :  23392.48828125
RMSE :  12.367135363460482
변경전
RMSE :  12.402475969794056

변경 후
RMSE :  12.221096450729014(민맥스)
RMSE :  12.255723279848867(스탠다드)


"""
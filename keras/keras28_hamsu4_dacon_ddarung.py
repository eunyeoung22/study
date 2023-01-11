import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#1. 데이터
path = './_data/ddarung/' #데이터 위치가 현재 작성파일과 동일한 위치에 있을때(현재위치)
# path = '../_data/ddarung/' #상위폴더에 데이터 위치할때
# path = 'c:/study/_data/ddarung/' #파일 위치 강제경로 잡을때
train_csv = pd.read_csv(path + 'train.csv' , index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

# print(train_csv)
# print(train_csv.shape) #(1459, 9) -> 열 10이지만 'count'는 y값이므로 1을 빼준다.
# print(train_csv.columns)
# # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',        
# #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],   
# #       dtype='object')
# print(train_csv.info())
# print(test_csv.info())
# print(train_csv.describe())
print(submission.shape) #(715, 1)


## 결측치 처리 1. 제거 ##
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape) #(1328, 10)
x = train_csv.drop(['count'], axis=1)
print(x) #[1328 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape) #(1328,)


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9, shuffle=True, random_state=123
)

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv) 

# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))

#2. 모델 구성(순차형)
# model = Sequential()
# model.add(Dense(100, input_dim = 9, activation= 'linear'))
# model.add(Dense(100, activation= 'linear'))
# model.add(Dense(100, activation= 'linear'))
# model.add(Dense(100, activation= 'linear'))
# model.add(Dense(100, activation= 'linear'))
# model.add(Dense(100, activation= 'linear'))
# model.add(Dense(100, activation= 'linear'))
# model.add(Dense(100, activation= 'linear'))
# model.add(Dense(90, activation= 'linear'))
# model.add(Dense(90, activation= 'linear'))
# model.add(Dense(80, activation= 'linear'))
# model.add(Dense(50, activation= 'linear'))
# model.add(Dense(40, activation= 'linear'))
# model.add(Dense(30, activation= 'linear'))
# model.add(Dense(20, activation= 'linear'))
# model.add(Dense(1, activation= 'linear'))
# model.summary()
# Total params: 104,221
# Trainable params: 104,221
# Non-trainable params: 0

# 2. 모델 구성(함수형)
input1 = Input(shape=(9,))
dence1 = Dense(100, activation= 'linear')(input1)
dence2 = Dense(100, activation= 'linear')(dence1)
dence3 = Dense(100, activation= 'linear')(dence2)
dence4 = Dense(100, activation= 'linear')(dence3)
dence5 = Dense(100, activation= 'linear')(dence4)
dence6 = Dense(100, activation= 'linear')(dence5)
dence7 = Dense(100, activation= 'linear')(dence6)
dence8 = Dense(100, activation= 'linear')(dence7)
dence9 = Dense(90, activation= 'linear')(dence8)
dence10 = Dense(90, activation= 'linear')(dence9)
dence11 = Dense(80, activation= 'linear')(dence10)
dence12 = Dense(50, activation= 'linear')(dence11)
dence13 = Dense(40, activation= 'linear')(dence12)
dence14 = Dense(30, activation= 'linear')(dence13)
dence15 = Dense(20, activation= 'linear')(dence14)
output1 = Dense(1, activation= 'linear')(dence15)
model1 = Model(inputs = input1, outputs = output1)
model1.summary()
# Total params: 104,221
# Trainable params: 104,221
# Non-trainable params: 0

#3. 컴파일, 훈련
import time
model1.compile(loss='mse' , optimizer='adam', metrics='mae')
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor = 'val_loss',
                              mode = 'min',
                              patience = 300,
                              restore_best_weights=True)
hist = model1.fit(x_train, y_train, epochs=3000, batch_size=32, validation_split=0.2,
                callbacks=[earlyStopping], verbose=2)

#4. 평가, 예측
loss = model1.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model1.predict(x_test)
print('y_predict : ', y_predict)


def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))

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
# plt.title('dacon_ddarung loss') # 그래프 타이틀
# plt.legend() # 범례(알아서 빈곳에 현출)
# # plt.legend(loc='upper right') #범례(그래프 오른쪽)
# # plt.legend(loc='upper left') #범례(그래프 왼쪽)
# plt.show() # 그래프 보여줘

# .to_csv()를 사용하여 
# submission_0105.csv를 완성하시오.
y_submit = model1.predict(test_csv)
# print(y_submit)
# print(y_submit.shape) #(715, 1)

submission['count'] = y_submit
# print(submission)

submission.to_csv(path + 'submission_01111956.csv')



"""
RMSE :  48.92284822242941
"""


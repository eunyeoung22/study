
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))

# 2. 모델 구성(순차형)
# model = Sequential()
# model.add(Dense(500, input_dim=10, activation= 'sigmoid'))
# model.add(Dense(100, activation= 'linear'))
# model.add(Dense(100, activation= 'linear'))
# model.add(Dense(500, activation= 'linear'))
# model.add(Dense(100, activation= 'linear'))
# model.add(Dense(200, activation= 'linear'))
# model.add(Dense(1, activation= 'linear'))
# model.summary()
# # Total params: 186,701
# # Trainable params: 186,701
# # Non-trainable params: 0

# 2. 모델 구성(함수형)
input1 = Input(shape=(10,))
dence1 = Dense(500, activation= 'sigmoid')(input1)
dence2 = Dense(100, activation= 'linear')(dence1)
dence3 = Dense(100, activation= 'linear')(dence2)
dence4 = Dense(500, activation= 'linear')(dence3)
dence5 = Dense(100, activation= 'linear')(dence4)
dence6 = Dense(200, activation= 'linear')(dence5)
output1 = Dense(1, activation= 'linear')(dence6)
model1 = Model(inputs = input1, outputs = output1)
model1.summary()
# Total params: 186,701
# Trainable params: 186,701
# Non-trainable params: 0

#3.컴파일, 훈련
model1.compile(loss='mse' , optimizer='adam', metrics=['mae'])
hist = model1.fit(x_train, y_train, epochs=1500, batch_size=80)

#4.평가, 예측
loss = model1.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model1.predict(x_test)


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

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6)) #그래프 사이즈
# plt.plot(hist.history['loss'], c = 'red', marker = '.', 
#         label='loss') # c : 그래프 선 color / label : 그래프 선 이름


# plt.grid() #격자
# plt.xlabel('epochs') #x축
# plt.ylabel('loss') #y축
# plt.title('diabets loss') # 그래프 타이틀
# plt.legend() # 범례(알아서 빈곳에 현출)
# # plt.legend(loc='upper right') #범례(그래프 오른쪽)
# # plt.legend(loc='upper left') #범례(그래프 왼쪽)
# plt.show() # 그래프 보여줘

"""
loss :  [4370.814453125, 52.534339904785156]
***********************************
RMSE :  66.11214041795621
***********************************
***********************************
R2 :  0.27687135895762827

"""


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout,LSTM,Flatten
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

print(x_train.shape, x_test.shape)#(265, 10) (88, 10)
x_train = x_train.reshape(265,5,2)
x_test = x_test.reshape(88,5,2)


# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))


# 2. 모델 구성(순차형)
model = Sequential()
model.add(LSTM(100, input_shape=(5,2), activation= 'relu'))
model.add(Dense(500, activation= 'sigmoid'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(500, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(200, activation= 'linear'))
model.add(Dense(1, activation= 'linear'))
# model.summary()
# # Total params: 186,701
# # Trainable params: 186,701
# # Non-trainable params: 0


#3.컴파일, 훈련
model.compile(loss='mse' , optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor= 'val_loss', 
                             patience= 100, mode = 'min', 
                             restore_best_weights=True, 
                             verbose=1)
import datetime
date = datetime.datetime.now()
print(date) #2023-01-12 14:57:51.908060
print(type(date)) #class 'datetime.datetime' 
date = date.strftime("%m%d_%H%M") #0112_1502 ->스트링 문자열 형식으로 바꿔주기
print(date)
print(type(date)) #<class 'str'>->스트링 문자열 형태임 

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # epoch는 정수 4자리까지 val_loss는 소수점 4자리 이하까지 .hdf5 파일 만들기

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, #가장 좋은 지점을 저장
                      filepath= filepath +'k31_03_'+ '_' + date + '_' + filename) #파일 저장 경로 지정                
                    #   filepath= path +'MCP/keras30_ModelCheckPoint13.hdf5') #파일 저장 경로 지정
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.3, 
        callbacks=[es, mcp], verbose=1)


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


"""
loss :  [4370.814453125, 52.534339904785156]
***********************************
RMSE :  66.11214041795621
***********************************
***********************************
R2 :  0.27687135895762827


loss :  [3194.0751953125, 46.190433502197266]
***********************************
RMSE :  56.51615307908093
***********************************
***********************************
R2 :  0.4715568012384689

LSTM 적용
loss :  [4041.5302734375, 51.408687591552734]
RMSE :  63.573028216225936
R2 :  0.33134988272724386

loss :  [3401.465087890625, 47.072166442871094]
RMSE :  58.322083416958115
R2 :  0.43724522767912155

"""

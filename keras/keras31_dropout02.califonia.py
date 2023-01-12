
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets= fetch_california_housing()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.9, shuffle=True, random_state=123
)

# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# # x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델 구성(순차형)
model = Sequential()
model.add(Dense(500, input_dim=8, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(500, activation= 'linear'))
model.add(Dropout(0.5))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dropout(0.2))
model.add(Dense(50, activation= 'linear'))
model.add(Dense(1, activation= 'linear')) 
model.summary()
# # Total params: 210,801
# # Trainable params: 210,801
# # Non-trainable params: 0

# 2. 모델 구성(함수형)
# input1 = Input(shape=(8,))
# dence1 = Dense(500, activation= 'relu')(input1)
# dence2 = Dense(100, activation= 'linear')(dence1)
# dence3 = Dense(100, activation= 'linear')(dence2)
# dence4 = Dense(500, activation= 'linear')(dence3)
# dence5 = Dense(100, activation= 'linear')(dence4)
# dence6 = Dense(100, activation= 'linear')(dence5)
# dence7 = Dense(100, activation= 'linear')(dence6)
# dence8 = Dense(100, activation= 'linear')(dence7)
# dence9 = Dense(100, activation= 'linear')(dence8)
# dence10 = Dense(50, activation= 'linear')(dence9)
# output1 = Dense(1, activation= 'linear')(dence10)
# model = Model(inputs = input1, outputs = output1)
# model.summary()
# Total params: 210,801
# Trainable params: 210,801
# Non-trainable params: 0

#3.컴파일, 훈련
import time
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
                      filepath= filepath +'k31_02_'+ '_' + date + '_' + filename) #파일 저장 경로 지정                
                    #   filepath= path +'MCP/keras30_ModelCheckPoint13.hdf5') #파일 저장 경로 지정
model.fit(x_train, y_train, epochs=100, batch_size=50, validation_split= 0.3,
                  callbacks=[es, mcp], verbose=1)


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
dropout 적용
RMSE :  1.1503178728746182
R2 :  0.05037146751141797
"""
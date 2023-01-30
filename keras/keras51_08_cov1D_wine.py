import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
y = to_categorical(y)

# print(x.shape, y.shape) #(178, 13) (178,)
# print(np.unique(y)) #[0 1 2]
# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(
                x, y, shuffle=True,
                random_state=123,
                test_size=0.2,
                stratify=y
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)#(142, 13) (36, 13)

x_train = x_train.reshape(142,13,1)
x_test = x_test.reshape(36,13,1)

# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))

#2. 모델 구성(순차형)
model = Sequential()
model.add(Conv1D(100, 2, input_shape =(13,1), activation = 'relu'))
model.add(LSTM(200, input_shape=(13,1), activation= 'relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(3, activation='softmax'))
model.summary()
# # Total params: 1,195,803
# # Trainable params: 1,195,803
# # Non-trainable params: 0


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor= 'val_loss',
                              mode='min',
                              patience = 50, 
                              restore_best_weights=True
                              )
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
                      filepath= filepath +'k31_08_'+ '_' + date + '_' + filename) #파일 저장 경로 지정                
                    #   filepath= path +'MCP/keras30_ModelCheckPoint13.hdf5') #파일 저장 경로 지정
model.fit(x_train, y_train, epochs=200, batch_size=1,
          callbacks=[es, mcp],validation_split=0.2, verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy : ', accuracy)

"""
loss:  1.0164193554373924e-05
accuracy :  1.0

loss:  0.11479764431715012
accuracy :  0.9722222089767456

cnn 적용후
loss:  2.9503255518648075e-06
accuracy :  1.0

LSTM적용
loss:  0.023019473999738693
accuracy :  1.0

conv1D
loss :  0.17966678738594055
accuracy :  0.9824561476707458
accuracy_score :  0.9122807017543859
"""


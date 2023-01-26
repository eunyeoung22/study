import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)#60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)#(10000, 28, 28) (10000,)


# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
print(x_train.shape, y_train.shape)#(60000, 784) (60000,)
print(x_test.shape, y_test.shape)#(10000, 784) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,Dropout

#2. 모델
model = Sequential()
model.add(Dense(100, input_shape =(784, ), activation='relu')) 
model.add(Dropout(0.3))
model.add(Dense(90, activation='relu')) 
model.add(Dense(80, activation='linear')) 
model.add(Dense(70, activation='linear')) 
model.add(Dense(60, activation='linear')) 
model.add(Dense(50, activation='linear')) 
model.add(Dense(20, activation='linear')) 
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', 
            metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=20,
                   verbose=1)
# import datetime
# date = datetime.datetime.now()
# print(date) #2023-01-12 14:57:51.908060
# print(type(date)) #class 'datetime.datetime' 
# date = date.strftime("%m%d_%H%M") #0112_1502 ->스트링 문자열 형식으로 바꿔주기
# print(date)
# print(type(date))

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
#                       save_best_only=True, 
#                       filepath= filepath +'k34_1_'+ '_' + date + '_' + filename)               
#                     # filepath= path +'MCP/keras30_ModelCheckPoint13.hdf5') #파일 저장 경로 지정
model.fit(x_train, y_train, epochs= 100, verbose=1, batch_size=32
          , callbacks =[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

"""
loss :  0.12990501523017883
acc :  0.974399983882904
"""

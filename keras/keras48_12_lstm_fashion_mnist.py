from tensorflow.keras.datasets import fashion_mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)#60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)#(10000, 28, 28) (10000,)
# print(x_train[1000])
# print(y_train[1000])


print(np.unique(y_train, return_counts=True))

x_train = x_train / 255.
x_test = x_test / 255.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#2. 모델
model = Sequential()
model.add(LSTM(100, input_shape=(28,28), activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(100, activation='linear')) 
model.add(Dense(100, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='linear')) 
model.add(Dense(50, activation='relu')) 
model.add(Dense(50, activation='linear'))
model.add(Dense(30, activation='linear')) 
model.add(Dense(20, activation='linear')) 
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', 
            metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=10,
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
model.fit(x_train, y_train, epochs= 50, verbose=1, batch_size=32,
          callbacks = [es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


# import matplotlib.pyplot as plt
# plt.imshow(x_train[1000], 'gray')
# plt.show()

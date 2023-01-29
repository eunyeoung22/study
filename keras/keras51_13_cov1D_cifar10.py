import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()



print(x_train.shape, y_train.shape)#(50000, 32, 32, 3) (50000, 1)


x_train = x_train.reshape(50000,3072,1)
x_test = x_test.reshape(10000,3072,1)

print(x_train.shape, y_train.shape) #(50000, 3072,1) (50000,1)
print(x_test.shape, y_test.shape)   #(10000, 3072,1) (10000,1)

# print(np.unique(y_train, return_counts=True))
# # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# # array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
# #       dtype=int64))





#2. 모델
model = Sequential()
model.add(Conv1D(100, 3, input_shape = (3072, 1), activation='relu'))
model.add(LSTM(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', 
            metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=50,
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
#                       filepath= filepath +'k34_2_'+ '_' + date + '_' + filename)               
#                     #   filepath= path +'MCP/keras30_ModelCheckPoint13.hdf5') #파일 저장 경로 지정
model.fit(x_train, y_train, epochs= 200, verbose=1, batch_size=100, validation_split=0.2,
          callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

"""

loss :  0.36694279313087463
acc :  0.8730000257492065

"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000,1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000,1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#       dtype=int64))



#2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape = (32,32,3),
          activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=64, kernel_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(2,2))) 
model.add(Flatten()) 
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', 
            metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=150,
                   verbose=1)

import datetime
date = datetime.datetime.now()
print(date) #2023-01-12 14:57:51.908060
print(type(date)) #class 'datetime.datetime' 
date = date.strftime("%m%d_%H%M") #0112_1502 ->스트링 문자열 형식으로 바꿔주기
print(date)
print(type(date))

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, 
                      filepath= filepath +'k34_2_'+ '_' + date + '_' + filename)               
                    #   filepath= path +'MCP/keras30_ModelCheckPoint13.hdf5') #파일 저장 경로 지정
model.fit(x_train, y_train, epochs= 300, verbose=1, batch_size=100, validation_split=0.2,
          callbacks=[es, mcp])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

"""
loss :  2.3025925159454346
acc :  0.10000000149011612

loss :  1.3097537755966187
acc :  0.5479000210762024

loss :  1.6605857610702515
acc :  0.5126000046730042

loss :  1.1182035207748413
acc :  0.6129000186920166

loss :  1.169961929321289
acc :  0.607200026512146

loss :  2.8691251277923584
acc :  0.6071000099182129
"""
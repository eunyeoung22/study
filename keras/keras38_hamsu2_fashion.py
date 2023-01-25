from tensorflow.keras.datasets import fashion_mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)#60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)#(10000, 28, 28) (10000,)
print(x_train[1000])
print(y_train[1000])

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# print(x_train.shape, y_train.shape)#(60000, 28, 28, 1) (60000,)
# print(x_test.shape, y_test.shape)#(10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten,MaxPooling2D,Input


#2. 모델 구성(순차형)
# model = Sequential()
# model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape = (28,28,1),
#           activation = 'relu')) # (27,27,128)
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=64, kernel_size=(2,2)))# (26,26,64)
# model.add(Conv2D(filters=64, kernel_size=(2,2))) # (25,25,64)
# model.add(Flatten()) #->40000
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# model.add(Dense(50, activation='relu')) #input_shape(400000, )
#                #(6만 4만)이 인풋이야   (batch_size, input_dim)
# model.add(Dense(40, activation='relu')) 
# model.add(Dense(30, activation='linear')) 
# model.add(Dense(20, activation='linear')) 
# model.add(Dense(20, activation='linear')) 
# model.add(Dense(20, activation='linear')) 
# model.add(Dense(20, activation='linear')) 
# model.add(Dense(10, activation='softmax'))

#2. 모델 구성(함수형)
input1 = Input(shape=(784,))
dence1 = Dense(128, activation= 'relu')(input1)
dence2 = Dense(64, activation= 'relu')(dence1)
dence3 = Dense(64, activation= 'relu')(dence2)
dence4 = Dense(50, activation= 'relu')(dence3)
dence5 = Dense(40, activation= 'linear')(dence4)
dence6 = Dense(20, activation= 'linear')(dence5)
dence7 = Dense(20, activation= 'linear')(dence6)
dence8 = Dense(20, activation= 'linear')(dence7)
dence9 = Dense(20, activation= 'linear')(dence8)
dence10 = Dense(50, activation= 'linear')(dence9)
output1 = Dense(10, activation= 'softmax')(dence10)
model = Model(inputs = input1, outputs = output1)

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

"""
loss :  0.6417582631111145
acc :  0.89230000972747

함수형
loss :  0.4106460511684418
acc :  0.8871999979019165
"""


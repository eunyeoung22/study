#53_01 복붙

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = 'D:/_data/' #  데이터 위치
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)


# np.save('./_data/brain/brain_x_train.npy', arry = xy_train[0][0])
# np.save('./_data/brain/brain_y_train.npy', arry = xy_train[0][1])
# # np.save('./_data/brain/brain_y_tranin.npy', arry = xy_train[0])#안먹힘 쓰지 말것
# np.save('./_data/brain/brain_x_test.npy', arry = xy_test[0][0])
# np.save('./_data/brain/brain_y_test.npy', arry = xy_test[0][1])

x_train = np.load('D:/_data/cat_dog_x_train.npy')
y_train = np.load('D:/_data/cat_dog_y_train.npy')
x_test= np.load('D:/_data/cat_dog_x_test.npy')
y_test= np.load('D:/_data/cat_dog_y_test.npy')


print(x_train.shape, x_test.shape)#(500, 200, 200, 3) (500, 200, 200, 3)
print(y_train.shape, y_test.shape)#(500,) (500,)


# # 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,LSTM,MaxPooling2D

model = Sequential()
model.add(Conv2D(100, (2,2), input_shape = (200,200,3), activation='relu'))
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# #3. 컴파일, 훈련(배치사이즈 넣기)
model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(#xy_train[0][0], xy_train[0][1],
                 x_train, y_train, #xy_train 통으로 사용할 수 있음
                 batch_size =16,#steps_per_epoch=16, 
                 epochs=50,
                 validation_data=(x_test,y_test),#validation_data : 검증데이터셋을 제공할 제네레이터를 지정
                # validation_steps=4)#validation_steps : epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정
                 validation_split=0.2)
accuracy = hist.history['acc']
val_acc =  hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('accuracy: ', accuracy[-1])
print('val_acc: ', val_acc[-1])

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0][0][159], 'rgb') #Tuple(한덩어리 안에 [xy가 같이 존재][x인지 y인지 구분][batch_size만큼의 x,y각각 존재])
# plt.show()


y_submit = model.predict(x_test)
print(y_submit)
print(y_submit.shape)

submission = submission[:500]
submission['label'] = y_submit
print(submission)
submission.to_csv(path + 'submission_01311526.csv')

# print(type(xy_train))#<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))#<class 'tuple'> :한번 생성되면 바꿀수 없다.
# print(type(xy_train[0][0]))#<class 'numpy.ndarray'>
# print(type(xy_train[0][1]))#<class 'numpy.ndarray'>

#2. 모델구성

"""
loss:  2.5059489416889846e-05
val_loss:  21.22486114501953
accuracy:  1.0
val_acc:  0.550000011920929
"""




import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터
#사진 같은 이미지를 .....증폭도 가능하다 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, #이미지 수평 반전
    vertical_flip=True, # 이미지 수직 반전
    width_shift_range=0.1, # 수평 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range=0.5, #이미지회전 
    zoom_range=1.2, #확대
    shear_range=0.7, # 이미지를 역방향 힘으로 변형
    fill_mode='nearest' # 옆으로 옮겨졌으면 빈곳을 가까운 값으로 채워 넣어라

)
# test 데이터는 rescale만 한다 : 평가데이터 이기에 때문에 증폭 할 필요없으므로 원데이터만 쓴다.
test_datagen = ImageDataGenerator(
    rescale=1./255
)
xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',
    target_size=(100,100),# 전체 이미지를 동일하게 (200,200)으로 맞춘다(이미지 사이즈들이 모두 같을 수 없기 때문)
    batch_size=1000,
    class_mode='binary',# 수치화
    color_mode='grayscale',
    shuffle=True # 0,1를 적당히 섞여서 테스트 하기 위해 'True'사용
    # Found 160 images belonging to 2 classes.
    ) #폴더 구조를 자동으로 이미지가 들어있는 폴더를 0,1로 구분함 X = (160,150,150,1), y = (160,) ad:80장, normal:80장
xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(100,100),# 전체 이미지를 동일하게 (200,200)으로 맞춘다(이미지 사이즈들이 모두 같을 수 없기 때문)
    batch_size=1000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
    # Found 120 images belonging to 2 classes.
    )

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,LSTM,MaxPooling2D

model = Sequential()
model.add(Conv2D(100, (2,2), input_shape = (100,100,1), activation='relu'))
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))#0,1로만 구성되어 있는 이진분류로 마지막엔 sigmoid 또는 softmax 사용

#3. 컴파일, 훈련(배치사이즈 넣기)
model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(xy_train[0][0], xy_train[0][1],#steps_per_epoch=16, 
                epochs=50,
                validation_data=(xy_test[0][0], xy_test[0][1]),#validation_data : 검증데이터셋을 제공할 제네레이터를 지정
                # validation_steps=4)#validation_steps : epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정
)
accuracy = hist.history['acc']
val_acc =  hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('accuracy: ', accuracy[-1])
print('val_acc: ', val_acc[-1])

import matplotlib.pyplot as plt
plt.imshow(xy_train[1][0][1], 'gray') #Tuple(한덩어리 안에 [xy가 같이 존재][x인지 y인지 구분][batch_size만큼의 x,y각각 존재])
plt.show()


"""
loss:  0.00017916594515554607
val_loss:  1.2281023263931274
accuracy:  1.0
val_acc:  0.6416666507720947


loss:  5.169044925423805e-06
val_loss:  1.9013832807540894
accuracy:  1.0
val_acc:  0.7333333492279053
"""
#4. 평가, 예측







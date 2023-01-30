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
    batch_size=10, # 컴파일 시 배치사이즈 넣어줄 필요없고, 미리 분리해줌/ 크게 잡아주면 데이터 전체를 훈련시킬수 있음
    class_mode='binary',# 수치화
    color_mode='grayscale',
    shuffle=True # 0,1를 적당히 섞여서 테스트 하기 위해 'True'사용
    # Found 160 images belonging to 2 classes.
    ) #폴더 구조를 자동으로 이미지가 들어있는 폴더를 0,1로 구분함 X = (160,150,150,1), y = (160,) ad:80장, normal:80장
xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(100,100),# 전체 이미지를 동일하게 (200,200)으로 맞춘다(이미지 사이즈들이 모두 같을 수 없기 때문)
    batch_size=10, # 컴파일 시 배치사이즈 넣어줄 필요없고, 미리 분리해줌
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
    # Found 120 images belonging to 2 classes.
    )

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,LSTM

model = Sequential()
model.add(Conv2D(100, (3,3), input_shape = (100,100,1), activation='relu'))
model.add(Conv2D(100,(2,2), padding='same', activation='relu'))
model.add(Conv2D(100,(2,2), padding ='same', activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(1, activation='sigmoid'))#0,1로만 구성되어 있는 이진분류로 마지막엔 sigmoid 또는 softmax 사용

#3. 컴파일, 훈련
model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit_generator(xy_train, steps_per_epoch=10, epochs=300,
                    validation_data=xy_test,#validation_data : 검증데이터셋을 제공할 제네레이터를 지정
                    validation_steps=5)#validation_steps : epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정

accuracy = hist.history['acc']
val_acc =  hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('accuracy: ', accuracy[-1])
print('val_acc: ', val_acc[-1])



import matplotlib.pyplot as plt
plt.imshow(xy_train[3][0][9], 'gray') #Tuple(한덩어리 안에 [xy가 같이 존재][x인지 y인지 구분][batch_size만큼의 x,y각각 존재])
plt.show()

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6)) #그래프 사이즈
# plt.plot(hist.history['loss'], c = 'red', marker = '.', 
#         label='loss') # c : 그래프 선 color / label : 그래프 선 이 름
# plt.plot(hist.history['val_loss'], c = 'blue', marker = '.', 
#         label = 'val loss')
# plt.plot(hist.history['acc'], c = 'orange', marker = '.', 
#         label = 'acc')
# plt.plot(hist.history['val_acc'], c = 'green', marker = '.', 
#         label = 'val_acc')

# plt.grid() #격자
# plt.xlabel('epochs') #x축
# plt.ylabel('loss') #y축
# plt.title('fit_generator') # 그래프 타이틀
# plt.legend() # 범례(알아서 빈곳에 현출)
# # plt.legend(loc='upper right') #범례(그래프 오른쪽)
# # plt.legend(loc='upper left') #범례(그래프 왼쪽)
# plt.show() # 그래프 보여줘

"""
loss:  0.6931695342063904
val_loss:  0.6932075619697571
accuracy:  0.5
val_acc:  0.4749999940395355

loss:  0.6933556199073792
val_loss:  0.6929908990859985
accuracy:  0.4699999988079071
val_acc:  0.5600000023841858

"""

#4. 평가, 예측







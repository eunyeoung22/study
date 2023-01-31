import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#사진 같은 이미지를 .....증폭도 가능하다 
train_datagen = ImageDataGenerator(
    rescale=1./255   
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# test 데이터는 rescale만 한다 : 평가데이터 이기에 때문에 증폭 할 필요없으므로 원데이터만 쓴다.
# test_datagen = ImageDataGenerator(
#     rescale=1./255
# )
xy_train = train_datagen.flow_from_directory(
    'D:/_data/train/',
    target_size=(200,200),# 전체 이미지를 동일하게 (200,200)으로 맞춘다(이미지 사이즈들이 모두 같을 수 없기 때문)
    batch_size=500, # 컴파일 시 배치사이즈 넣어줄 필요없 고, 미리 분리해줌/ 크게 잡아주면 데이터 전체를 훈련시킬수 있음
    class_mode='binary',# 수치화
    # class_mode='categorical',# 수치화
    color_mode='rgb',
    shuffle=True
    # 
    ) 
    #폴더 구조를 자동으로 이미지가 들어있는 폴더를 0,1로 구분함 X = (160,150,150,1), y = (160,) ad:80장, normal:80장
xy_test = test_datagen.flow_from_directory(
    'D:/_data/test1/',
    target_size=(200,200),# 전체 이미지를 동일하게 (200,200)으로 맞춘다(이미지 사이즈들이 모두 같을 수 없기 때문)
    batch_size=500, # 컴파일 시 배치사이즈 넣어줄 필요없고, 미리 분리해줌
    class_mode='binary',# 수치화
    # class_mode='categorical',
    color_mode='rgb',
    shuffle=True
    # 
    )

print(xy_train[0][0])
print(xy_train[0][0].shape)#binary (500, 200, 200, 1)
print(xy_train[0][1])#
print(xy_train[0][1].shape)#(10, 2) / (500,)
print(xy_test[0][0].shape)#(500, 200, 200, 3)
print(xy_test[0][1].shape)#(500,)


np.save('D:/_data/cat_dog_x_train.npy', arr = xy_train[0][0])
np.save('D:/_data/cat_dog_y_train.npy', arr = xy_train[0][1])
np.save('D:/_data/cat_dog_x_test.npy', arr =  xy_test[0][0])
np.save('D:/_data/cat_dog_y_test.npy', arr = xy_test[0][1])

# np.save('./_data/brain/brain_y_tranin.npy', arry = xy_train[0])#xy가 전체 들어가지만 빼서 쓸때 나눠서 빼줘야함



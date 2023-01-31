#53_01 복붙

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    target_size=(200,200),# 전체 이미지를 동일하게 (200,200)으로 맞춘다(이미지 사이즈들이 모두 같을 수 없기 때문)
    batch_size=10, # 컴파일 시 배치사이즈 넣어줄 필요없 고, 미리 분리해줌/ 크게 잡아주면 데이터 전체를 훈련시킬수 있음
    class_mode='binary',# 수치화
    # class_mode='categorical',# 수치화
    color_mode='grayscale',
    shuffle=True
    # Found 160 images belonging to 2 classes.
    ) #폴더 구조를 자동으로 이미지가 들어있는 폴더를 0,1로 구분함 X = (160,150,150,1), y = (160,) ad:80장, normal:80장
xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(200,200),# 전체 이미지를 동일하게 (200,200)으로 맞춘다(이미지 사이즈들이 모두 같을 수 없기 때문)
    batch_size=10, # 컴파일 시 배치사이즈 넣어줄 필요없고, 미리 분리해줌
    class_mode='binary',# 수치화
    # class_mode='categorical',
    color_mode='grayscale',
    shuffle=True
    # Found 120 images belonging to 2 classes.
    )
print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000235FF011AF0>

# from sklearn.datasets import load_iris

# datasets = load_iris
# print(datasets)

# print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][0].shape)#(10, 200, 200, 1) / binary (10, 200, 200, 1)
print(xy_train[0][1])#
print(xy_train[0][1].shape)#(10, 2) / binary (10,)
# print(xy_train[15][0].shape)#(10, 200, 200, 1)
# print(xy_train[15][1].shape)#(10,)
# 통데이터로 할려면 배치 사이즈를 크게 넣어서 전체 데이터 훈련 시킬수 있다.

# print(type(xy_train))#<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))#<class 'tuple'> :한번 생성되면 바꿀수 없다.
# print(type(xy_train[0][0]))#<class 'numpy.ndarray'>
# print(type(xy_train[0][1]))#<class 'numpy.ndarray'>

#2. 모델구성






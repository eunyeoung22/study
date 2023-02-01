# 이미지를 뒤집고 늘리고 돌리고 움직여서 데이터 수를 증폭시키자!

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size = 40000  # 전체 데이터 중 몇 장을 증폭할 건지

randidx = np.random.randint(x_train.shape[0],  # x_train의 데이터 수
                            size=augument_size)
print(randidx)  # 랜덤하게 골라진 40000개의 데이터
print(len(randidx))  # 40000

x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()
# 원본을 건드리지 않도록 안전하게 copy() 사용
# 전체 데이터 60000개 중 augument_size만큼의 데이터를 랜덤하게 골라 복사함

x_augument = x_augument.reshape(40000, 28, 28, 1)




train_datagen = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True, vertical_flip=True,
    width_shift_range=0.1, height_shift_range=0.1,
    rotation_range=5, zoom_range=1.2, shear_range=0.7,
    fill_mode='nearest')

x_augumented = train_datagen.flow(
    x_augument,  # x
    y_augument,  # y
    batch_size=augument_size,
    shuffle=True
)

print(x_augumented[0][0].shape)  # (40000, 28, 28, 1)
print(x_augumented[0][1].shape)  # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1]))

print(x_train.shape, y_train.shape)
# (100000, 28, 28, 1) (100000,)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten #Conv2D(이미지(그림))

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2),#kernel_size : 조각내는 기준
          input_shape=(5, 5, 1)))# filter : 히든레이어의 갯수 개념
model.add(Conv2D(filters=5, kernel_size=(2,2)))
model.add(Flatten())#dence 형식으로 바꾸기(행렬)
model.add(Dense(10))
model.add(Dense(1))
model.summary()


                    
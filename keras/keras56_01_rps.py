# 실습
# 가위바위보 모델 만들기


import numpy as np


# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(rescale=1./255)
# xy_train = datagen.flow_from_directory(
#     'D:/_data/rps/',  
#     target_size=(100, 100),
#     batch_size=2600,  # 너무 큰 수를 주면 memory overflow 발생
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True,
# )  # Found 25000 images belonging to 2 classes.
# np.save('./_data/rps/rps_x_train.npy', arr=xy_train[0][0])
# np.save('./_data/rps/rps_y_train.npy', arr=xy_train[0][1])



x_train = np.load('./_data/rps/rps_x_train.npy')
y_train = np.load('./_data/rps/rps_y_train.npy')

print(x_train.shape, y_train.shape)
# (2520, 200, 200, 3) (2520,)



##2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(30, 30, input_shape=(100, 100, 3)))
model.add(Conv2D(80, 20, activation='relu'))
model.add(Conv2D(40, 20, activation='relu'))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='softmax'))



##3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train,
                 batch_size=40, epochs=50,
                 validation_split=0.1,
                 verbose=3)



##4. 평가, 예측
loss = hist.history['loss']
val_loss = hist.history['val_loss']
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('accuracy:', accuracy[-1])
print('val_acc:', val_acc[-1])

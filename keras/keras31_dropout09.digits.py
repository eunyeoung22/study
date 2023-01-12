import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1.데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
y = to_categorical(y)

# print(x.shape, y.shape) #(1797, 64) (1797,)
# print(np.unique(y, return_counts=True))
# #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# #array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[3])
# plt.show()
x_train, x_test, y_train, y_test = train_test_split(
                x, y, shuffle=True,
                random_state=123,
                test_size=0.2,
                stratify=y
)
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))

#2. 모델 구성(순차형)
# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape = (64,)))
# model.add(Dense(100, activation='linear'))
# model.add(Dense(100, activation='linear'))
# model.add(Dense(100, activation='linear'))
# model.add(Dense(100, activation='linear'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='linear'))
# model.add(Dense(50, activation='linear'))
# model.add(Dense(10, activation='softmax'))
# model.summary()
# Total params: 57,810
# Trainable params: 57,810
# Non-trainable params: 0

# 2. 모델 구성(함수형)
input1 = Input(shape=(64,))
dence1 = Dense(10, activation= 'relu')(input1)
dence2 = Dense(100, activation= 'linear')(dence1)
dence3 = Dense(100, activation= 'linear')(dence2)
dence4 = Dense(100, activation= 'linear')(dence3)
dence5 = Dense(100, activation= 'linear')(dence4)
dence6 = Dense(100, activation= 'relu')(dence5)
dence7 = Dense(100, activation= 'linear')(dence6)
dence8 = Dense(50, activation= 'linear')(dence7)
output1 = Dense(10, activation= 'softmax')(dence8)
model1 = Model(inputs = input1, outputs = output1)
model1.summary()
# Total params: 57,810
# Trainable params: 57,810
# Non-trainable params: 0

#3. 컴파일, 훈련
model1.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_accurucy',
                              mode = 'max',
                              patience = 70,
                              restore_best_weights=True,
                              verbose=1)
model1.fit(x_train, y_train, epochs=100, batch_size=20,
                 callbacks=[earlyStopping], validation_split=0.2, verbose=1)

#4. 평가, 예측
loss, accuracy = model1.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score
y_predict = model1.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)
# acc = accuracy_score(y_test, y_predict)
# print(acc)

"""
loss:  0.8500062823295593
accuracy :  0.9472222328186035

"""
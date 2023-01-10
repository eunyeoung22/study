import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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
#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_shape = (64,)))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_accurucy',
                              mode = 'max',
                              patience = 70,
                              restore_best_weights=True,
                              verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=20,
                 callbacks=[earlyStopping], validation_split=0.2, verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)
# acc = accuracy_score(y_test, y_predict)
# print(acc)

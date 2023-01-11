import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
y = to_categorical(y)

# print(x.shape, y.shape) #(178, 13) (178,)
# print(np.unique(y)) #[0 1 2]
# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(
                x, y, shuffle=True,
                random_state=123,
                test_size=0.2,
                stratify=y
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))

#2. 모델 구성
model = Sequential()
model.add(Dense(10000, activation='linear', input_shape = (13,)))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_loss',
                              mode='min',
                              patience = 100, 
                              restore_best_weights=True
                              )
model.fit(x_train, y_train, epochs=10000, batch_size=1,
          callbacks=[earlyStopping],validation_split=0.2, verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy : ', accuracy)


"""
변경 전
loss:  0.25725018978118896
accuracy :  0.9722222089767456

변경 후
loss:  0.002826852258294821
accuracy :  1.0

스켈링 한 후 더 좋아짐

"""
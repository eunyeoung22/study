from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout,Conv2D,Flatten,LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']
y = to_categorical(y) # 노드 마지막 아웃풋이 개수 오류에 대한 치환을 해줄 경우
# print(y)
# print(y.shape)
# print(datasets.DESCR) #판다스 .describe() / .info() 사용
# print(datasets.feature_names) #판다스 .columns 사용
# print(x.shape, y.shape) #(150, 4) (150,)
x_train, x_test, y_train, y_test = train_test_split(
                                x, y, shuffle = True, #False일 경우 문제점 : y_test와 y_train 성능에 문제발생
                                random_state=123, 
                                test_size=0.2,
                                stratify=y) # y에 대한 경우에만 사용(균일하게 분배)

# print(y_train)
# print(y_test)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)#(120, 4) (30, 4)

x_train = x_train.reshape(120,2,2)
x_test = x_test.reshape(30,2,2)



# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))

# 2. 모델 구성(순차형)
model = Sequential()
model.add(LSTM(100, input_shape=(2,2), return_sequences= 'True', activation='relu'))                
model.add(LSTM(100))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(40, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(30, activation = 'sigmoid'))
model.add(Dense(3, activation = 'softmax'))#다중분류 시 activation = 'softmax' 사용, 노드개수 3 : y 클래스 종류
#                                            #최종 아웃풋 레이어에 액티베이션은 softmax다
# model.summary()
# # Total params: 8,583
# # Trainable params: 8,583
# # Non-trainable params: 0

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
# es = EarlyStopping(monitor= 'val_loss' ,
#                               mode='min',
#                               patience = 10, 
#                               restore_best_weights=True, # patience 10번 중 마지막 최소값이 아닌 그중 제일 최소값 반환
#                               verbose=1
#                               )

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2,
                verbose=1)
#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)


"""
loss:  0.22739247977733612
accuracy :  0.8666666746139526
[1 2 0 0 2 2 0 2 0 2 2 2 0 1 0 1 2 2 2 1 2 0 0 1 0 0 1 1 2 1]
[1 2 0 0 2 2 0 1 0 2 2 1 0 1 0 1 2 2 2 2 2 0 0 1 0 0 1 1 1 1]
0.8666666666666667

drop
loss:  0.3513701260089874
accuracy :  0.8999999761581421

cnn적용 후
loss:  0.1940150260925293
accuracy :  0.8666666746139526

LSTM 적용
loss:  0.32076606154441833
accuracy :  0.8999999761581421
[1 2 0 0 2 2 0 1 0 1 1 1 0 1 0 1 2 2 2 1 2 0 0 1 0 0 1 1 1 1]
[1 2 0 0 2 2 0 1 0 2 2 1 0 1 0 1 2 2 2 2 2 0 0 1 0 0 1 1 1 1]
0.9

"""
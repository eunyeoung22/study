from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']
# y = to_categorical(y) # 노드 마지막 아웃풋이 개수 오류에 대한 치환을 해줄 경우
# print(y)
# print(y.shape)
# print(datasets.DESCR) #판다스 .describe() / .info() 사용
# print(datasets.feature_names) #판다스 .columns 사용
# print(x.shape, y.shape) #(150, 4) (150,)
x_train, x_test, y_train, y_test = train_test_split(
                                x, y, shuffle = True, #False일 경우 문제점 : y_test와 y_train 성능에 문제발생
                                random_state=123, 
                                test_size=0.2,
                                stratify=y) # 분류에서만 사용 y에 대한 경우에만 사용(균일하게 분배)

# print(y_train)
# print(y_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(50, input_shape = (4,), activation = 'relu'))
model.add(Dense(40, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(30, activation = 'sigmoid'))
model.add(Dense(3, activation = 'softmax'))#다중분류 시 activation = 'softmax' 사용, 노드개수 3 : y 클래스 개수
                                           #최종 아웃풋 레이어에 액티베이션은 softmax다

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', #sparse_categorical_crossentropy 사용 시 위 모델구성 마지막 노드의 개수도 클래스 개수로 작성
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=1,
         validation_split=0.2,
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
# y_test = np.argmax(y_test, axis=1) # 원핫을 쓰지 않았으므로 사용 필요 없음
print(y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)


"""
loss:  0.19589351117610931
accuracy :  0.9333333373069763
"""
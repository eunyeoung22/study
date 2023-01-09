import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets['data']
y = datasets['target']
# print(x.shape , y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)


#2. 모델구성
model = Sequential()
model.add(Dense(50,  activation= 'linear', input_shape=(30,)))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid')) # 이진분류에선 꼭 마지막에 sigmoid 사용


#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['accuracy']) # 이진분류에선 꼭 이 훈련 쓰기
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_loss' ,
                              mode='min',
                              patience = 10, 
                              restore_best_weights=True,
                              verbose=1
                              )
model.fit(x_train, y_train, epochs=10000, batch_size= 15,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
intarr = list(map(int, y_predict)) 
print(intarr[:10])  #--> 정수형으로 바꿔줘야함 
print(y_test[:10])

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, intarr)
print("accuracy_score : ", acc)

"""
1. patience = 50
loss :  0.0944017618894577
accuracy :  0.9736841917037964

2.patience = 100
loss :  0.08319780975580215
accuracy :  0.9649122953414917

3.patience = 300
loss :  0.11062361299991608
accuracy :  0.9912280440330505

4.patience = 500
loss :  0.08158644288778305
accuracy :  0.9824561476707458
"""
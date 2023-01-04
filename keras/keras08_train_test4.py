import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])
# print(range(10))
x = x.T
print(x.shape)   #(3, 10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = y.T
print(y.shape)

#[실습] train_test_split를 이용하여
#7:3으로 잘라서 모델 구현

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    # test_size=0.3, 
    shuffle=True,
    random_state=777
    )

#2.모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(1))



#3.컴파일, 훈련
model.compile(loss='mae' ,  optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4.평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ' , loss)
result = model.predict([[9,30,210]])
print('[9,30,210]의 결과값 : ' , result)


# print('x_train : ', x_train)
# print('y_train : ' , y_train)
# print('x_test : ' , x_test)
# print('y_test : ', y_test)




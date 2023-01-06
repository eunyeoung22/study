import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2)

print(x_train.shape)
print(x_test.shape)


#2.모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse' , optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
            validation_split=0.25)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
 
y_predict = model.predict(x_test)
print('예측 값 : ', y_predict)
 
# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict)) 
 
# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)




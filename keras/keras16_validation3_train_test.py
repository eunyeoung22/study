import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))


#[실습]
# train_test_split로 잘라라
# 10:3:3 으로 나눠라
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle = False, test_size=0.375)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, random_state=66, shuffle = False, test_size=0.5)
print(x_train)
print(x_test)
print(x_validation)

# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_validation = np.array([14,15,16])
# y_validation = np.array([14,15,16])



# #2.모델구성
# model = Sequential()
# model.add(Dense(5, input_dim=1))
# model.add(Dense(3, activation= 'relu'))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse' , optimizer='adam')
# model.fit(x_train, y_train, epochs=100, batch_size=1,
#             validation_data=(x_validation, y_validation))

# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
 
# y_predict = model.predict(x_test)
# print(y_predict)
 
# # RMSE 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print('RMSE : ', RMSE(y_test, y_predict)) 
 
# # R2 구하기
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test, y_predict)
# print('R2 : ', r2_y_predict)




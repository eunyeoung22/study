import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

# print(datasets.DESCR)
# print(datasets.feature_names)
# # print(x.shape, y.shape) #(581012, 54) (581012,)
# print(np.unique(y, return_counts=True))
# # #(array([1, 2, 3, 4, 5, 6, 7])
# # #array[211840, 283301,  35754,   2747,   9493,  17367,  20510]

##############################1.케라스 투 카테고리컬###################################
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# # print(y.shape) #(581012, 8)
# # print(type(y))# y컬럼 타입이 무엇인지 :  class 'numpy.ndarray'
# # print(y[:10])# y 9번째 컬럼까지 출력
# # print(np.unique(y[:,0], return_counts=True)) #0번째 컬럼이 뭐가 있는지 확인
# # print(np.unique(y[:,1], return_counts=True)) #1번째 컬럼이 뭐가 있는지 확인
# y = np.delete(y, 0, axis=1) # 0번째 열 삭제
# print(y.shape)
# print(np.unique(y[:,0], return_counts=True)) #return_counts=True 컬럼갯수 카운트

##############################2.판다스 겟 더미스###########################################

# 1)첫번째 방법
# import pandas as pd
# y = pd.get_dummies(y)
# print(y[:10])
# y = y.to_numpy()
# print(type(y))

# 2)두번째 방법
# y= pd.get_dummies(y, drop_first=False)
# y = y.values # 둘중 하나 쓰기 '.values  또는 .to numpy()'
# y = np.array(y)
# print(y.shape)

#########################3.sklearn OneHotEncoder  ################################
# 1. 첫번째

print(y.shape) # (581012,0)
y = y.reshape(581012,1) # reshape 시 안의 내용은 바뀌지 않는다
print(y.shape)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder() #원핫인코딩 정의
# ohe.fit(y)
# y = ohe.transform(y)
y = ohe.fit_transform(y) #위 두개 명령어를 합침
print(type(y)) # class 'scipy.sparse._csr.csr_matrix' 형태
y = y.toarray() # class 'numpy.ndarray' 로 변경
print(type(y)) # 타입변경 확인



# 2. 두번째
#from sklearn.preprocessing import OneHotEncoder
# Initialize the OneHotEncoder
# ohe = OneHotEncoder()
# y = ohe.fit_transform(y.reshape(-1,1)).toarray()
# # Fit the OneHotEncoder on the target variable 'y'
# # Print the new shape of the target variable 'y'
# print(y.shape)
# print(x.shape)


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

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='linear', input_shape = (54,)))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor= 'val_accuracy',
                              mode='max',
                              patience=100,
                              verbose=2)
model.fit(x_train, y_train, epochs=100, batch_size=32,
          callbacks=[earlyStopping],validation_split=0.2, verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy : ', accuracy)


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1) #numpy 자료형이 pandas를 못받아 들임
# print(y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)

"""
loss:  0.664304256439209
accuracy :  0.7103431224822998

변경후 
loss:  0.6798371076583862
accuracy :  0.7041126489639282

loss:  0.6398959159851074
accuracy :  0.7209022045135498
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. (정제된) 데이터 
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])


#2. 모델 구성 
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련(batch_size의 경우 정제된 데이터를 단위로 쪼개여 훈련시키는 명령어)
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200, batch_size = 3)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([6])
print('6의 결과 : ', result)




#batch_size의 defalut 값은 "32"
#들여쓰기(Tab 키 사용) : 문장의 상위 소속 여부 
    #model.compile(loss='mae', optimizer='adam')
        #model.fit(x, y, epochs=10, batch_size = 3)
#들여쓰기 취소(shift + Tab)

# 블럭 주석 처리(주석처리 할 대상이 많을 시 쌍따옴표 3개)
"""
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=10)
result = model.predict([6])
print('6의 결과 : ', result)
"""

# 판단 기준 : loss

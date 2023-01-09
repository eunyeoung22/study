from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets =  load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)

#2. 모델구성
model = Sequential()
# model.add(Dense(5, input_dim=13))
model.add(Dense(5, input_shape=(13,)))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련
import time
model.compile(loss = 'mse' , optimizer='adam')
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2,
         verbose=3)
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

print("걸린시간 : ", end - start)
# 1. verbose=1, 걸린시간 :  13.388022184371948(실제 실행된 결과)
# 2. verbose=0, 걸린시간 :  11.288203001022339(결과만 보여 줌)
# 3. verbose=2, 걸린시간 :  10.991064071655273(프로그레스 bar "============="만 생략된 결과값)
# 4. verbose=3, 걸린시간 :  11.399138927459717(프로그레스 bar "=============" 생략 + Epoch 50/50 결과값)


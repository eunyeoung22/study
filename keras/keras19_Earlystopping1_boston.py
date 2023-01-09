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
model.add(Dense(100, input_shape=(13,)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam') 

from tensorflow.keras.callbacks import EarlyStopping #파이썬 class
earlyStopping = EarlyStopping(monitor = 'val_loss',
                              mode='min',
                              patience = 100, 
                              restore_best_weights=True
                              )
hist = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2,
         callbacks = [earlyStopping], verbose=1)
# hist : history(loss의 변화값 사전(딕셔너리) 형식의 list 로 내부에 저장되어 있음)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
print("=============================================")
print(hist)#(<keras.callbacks.History object at 0x000002965CCF2790>)
print("=============================================")
print(hist.history)
print("=============================================")
print(hist.history['loss'])
print("=============================================")
print(hist.history['val_loss'])
print("=============================================")


import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) #그래프 사이즈
plt.plot(hist.history['loss'], c = 'red', marker = '.', 
        label='loss') # c : 그래프 선 color / label : 그래프 선 이름
plt.plot(hist.history['val_loss'], c = 'blue', marker = '.', 
        label = 'val loss')
plt.grid() #격자
plt.xlabel('epochs') #x축
plt.ylabel('loss') #y축
plt.title('boston loss') # 그래프 타이틀
plt.legend() # 범례(알아서 빈곳에 현출)
# plt.legend(loc='upper right') #범례(그래프 오른쪽)
# plt.legend(loc='upper left') #범례(그래프 왼쪽)
plt.show() # 그래프 보여줘


"""
loss =  79.52824401855469
loss =  30.279895782470703
"""
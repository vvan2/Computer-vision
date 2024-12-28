import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 샘플 궤적 데이터 (x, y 좌표)
trajectory_data = np.array([
    [0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25]
])

# 입력(X)과 출력(Y) 데이터 준비
X = trajectory_data[:-1]
Y = trajectory_data[1:]

# 모델 생성
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dense(2))  # 출력: x, y 좌표

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='mse')
model.fit(X, Y, epochs=50, batch_size=1)

# 궤적 예측
predicted = model.predict(np.array([[5, 25]]))  # 새로운 좌표 예측
print("다음 궤적 좌표:", predicted)
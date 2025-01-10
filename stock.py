import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import matplotlib.pyplot as plt

# 주식 데이터 다운로드
data = yf.download('AAPL', start='2017-01-01', end='2024-01-01')
close_prices = data['Close']

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

# 학습을 위한 데이터 생성 함수
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# 데이터 분할
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# 학습 및 테스트 데이터 생성
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# LSTM 입력 형태로 데이터 변형
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# LSTM 모델 구축
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))  # 과적합 방지를 위한 Dropout
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))  # 과적합 방지를 위한 Dropout
model.add(Dense(units=1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 과적합 방지를 위한 EarlyStopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 예측 수행
predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions)

# Streamlit 앱 UI 설정
st.title('AAPL 주식 가격 예측')
st.subheader('예측된 주식 가격')

# 예측 가격과 실제 가격을 화면에 표시
st.write(f"예측된 가격: {predicted_prices[:10]}")
st.write(f"실제 가격: {scaler.inverse_transform(y_test.reshape(-1, 1))[:10]}")

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='실제 가격')
plt.plot(predicted_prices, color='red', label='예측 가격')
plt.title('AAPL 주식 가격 예측')
plt.xlabel('날짜')
plt.ylabel('가격 (USD)')
plt.legend()

# Streamlit에서 플롯 표시
st.pyplot(plt)
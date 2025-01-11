import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import datetime

# 주식 데이터 다운로드
symbol = 'AAPL'
data = yf.download(symbol, start='2017-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))
close_prices = data['Close']

# USD to KRW 환율 (Streamlit 실행 시 기준)
usd_to_krw = 1300  # 예시 환율 (실제 환율을 API로 가져올 수도 있음)

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

# 차원 확인
st.write("X_train 차원: ", X_train.shape)
st.write("y_train 차원: ", y_train.shape)
st.write("X_test 차원: ", X_test.shape)
st.write("y_test 차원: ", y_test.shape)

# LSTM 입력 형태로 데이터 변형 (3D 형식: 샘플, 시간 단계, 특성 수)
if len(X_train.shape) == 2:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
else:
    st.error("데이터 차원에 문제가 있습니다. 데이터를 확인해주세요.")
    st.stop()

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
predicted_prices = scaler.inverse_transform(predictions) * usd_to_krw  # KRW로 변환
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)) * usd_to_krw

# 최근 7일 데이터 및 미래 1일 예측
date_index = data.index[-(len(y_test_original) + 60):]  # 테스트 데이터 날짜
recent_data = scaled_data[-67:]  # 최근 7일 + 1일 예측

X_recent = []
for i in range(60, len(recent_data)):
    X_recent.append(recent_data[i-60:i, 0])
X_recent = np.array(X_recent).reshape(len(X_recent), 60, 1)

future_predictions = model.predict(X_recent)
future_prices = scaler.inverse_transform(future_predictions) * usd_to_krw

# 날짜 생성
recent_dates = date_index[-7:]
future_dates = [date_index[-1] + datetime.timedelta(days=i) for i in range(1, 2)]

# 실제 데이터와 예측 데이터 병합
actual_prices = close_prices[-7:] * usd_to_krw
all_dates = list(recent_dates) + future_dates
all_prices = list(actual_prices) + list(future_prices)

# Streamlit 앱 UI 설정
st.title(f'{symbol} 주식 가격 예측 (KRW)')
st.subheader('최근 일주일 및 미래 1일 예측')

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(recent_dates, actual_prices, label='실제 가격', color='blue', marker='o')
plt.plot(all_dates, all_prices, label='예측 가격', color='red', linestyle='--', marker='x')
plt.title(f'{symbol} 주식 가격 예측')
plt.xlabel('날짜')
plt.ylabel('가격 (KRW)')
plt.legend()

# Streamlit에서 플롯 표시
st.pyplot(plt)

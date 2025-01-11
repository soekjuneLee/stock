import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 환율 설정 (USD -> KRW 변환)
usd_to_krw = 1300  # 환율 값 (사용자가 필요에 따라 변경)

# Streamlit UI 설정
st.title("주식 가격 예측 및 시각화")
symbol = st.text_input("주식 코드를 입력하세요 (예: AAPL)", "AAPL")

# 오늘 날짜와 일주일 전 날짜 계산
today = datetime.today()
one_week_ago = today - timedelta(days=7)
tomorrow = today + timedelta(days=1)

# 주식 데이터 다운로드
st.subheader(f"{symbol} 주식 데이터 로드 중...")
data = yf.download(symbol, start=one_week_ago.strftime("%Y-%m-%d"), end=tomorrow.strftime("%Y-%m-%d"))

if data.empty:
    st.error("주식 데이터를 가져올 수 없습니다. 코드를 확인해주세요.")
else:
    # 종가 데이터 가져오기
    close_prices = data["Close"]
    st.write("최근 데이터:")
    st.dataframe(close_prices)

    # 데이터 전처리
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

    # 학습 데이터 준비
    train_data = []
    train_labels = []

    for i in range(5, len(scaled_data) - 1):  # 최근 5일 데이터를 기반으로 다음 날 예측
        train_data.append(scaled_data[i - 5:i, 0])  # 이전 5일 데이터
        train_labels.append(scaled_data[i, 0])  # 다음 날 데이터

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

    # LSTM 모델 구축
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))  # 예측 값 하나 출력
    model.compile(optimizer="adam", loss="mean_squared_error")

    # 모델 학습
    model.fit(train_data, train_labels, epochs=10, batch_size=1, verbose=2)

    # 내일 예측
    last_5_days_scaled = scaled_data[-5:]  # 최근 5일 데이터
    last_5_days_scaled = np.reshape(last_5_days_scaled, (1, 5, 1))  # LSTM 입력 형태로 변환
    predicted_scaled_price = model.predict(last_5_days_scaled)
    predicted_price = scaler.inverse_transform(predicted_scaled_price)  # 원래 값으로 변환

    # 실제 데이터와 예측 데이터 결합
    actual_prices = close_prices[-7:].values * usd_to_krw  # 최근 7일 데이터
    predicted_prices = np.append(actual_prices[-1], predicted_price[0][0] * usd_to_krw)  # 예측 값 추가

    # 날짜 생성
    recent_dates = pd.date_range(start=one_week_ago, periods=7).strftime("%Y-%m-%d")
    future_date = pd.date_range(start=today, periods=2).strftime("%Y-%m-%d")

    # 그래프 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(recent_dates, actual_prices, label="실제 가격", color="blue", marker="o")
    plt.plot([recent_dates[-1], future_date[1]], predicted_prices, label="예측 가격", color="red", linestyle="--", marker="o")
    plt.title(f"{symbol} 주식 가격 예측")
    plt.xlabel("날짜")
    plt.ylabel("가격 (KRW)")
    plt.xticks(rotation=45)
    plt.legend()

    # 결과 출력
    st.subheader("예측 결과")
    st.write(f"내일 예측 가격: {predicted_price[0][0] * usd_to_krw:.2f} KRW")
    st.pyplot(plt)
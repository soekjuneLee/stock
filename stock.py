import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Streamlit UI 구성
st.title("AI 기반 주식 투자 추천 프로그램")

# 주식 종목 딕셔너리
ticker_dict = {
    "삼성전자": "005930.KS",
    "기아": "000270.KS",
    "현대차": "005380.KS",
    "LG전자": "066570.KS",
    "네이버": "035420.KS",
    "카카오": "035720.KS",
    # 필요한 종목 추가 가능
}

# 사용자 입력 받기
stock_name = st.selectbox("주식 종목 선택", list(ticker_dict.keys()))
ticker = ticker_dict[stock_name]

# 기간 및 데이터 옵션
period = st.selectbox("기간 선택", ["1개월", "3개월", "6개월", "1년"])
period_mapping = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}
interval = st.selectbox("데이터 간격", ["일봉", "주봉"])
interval_mapping = {"일봉": "1d", "주봉": "1wk"}

# LSTM 기반 AI 주가 예측
if st.button("AI 기반 주가 예측"):
    # 주식 데이터 다운로드
    data = yf.download(ticker, period=period_mapping[period], interval=interval_mapping[interval])
    data = data[['Close']]  # 'Close' 컬럼만 사용

    # 데이터 전처리
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 훈련 및 테스트 데이터 분리
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # LSTM 입력 데이터 준비
    def create_dataset(dataset, time_step=50):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 50
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # 데이터 차원 변환
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # LSTM 모델 정의
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

    # 예측 및 시각화
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # 예측 결과 출력
    st.write("예측된 주가 결과:")
    plt.figure(figsize=(14, 7))
    plt.plot(scaler.inverse_transform(test_data[time_step:]), label="Actual Price")  # 실제 주가
    plt.plot(predictions, label="Predicted Price")  # 예측 주가
    plt.xlabel("Date")  # X축: 날짜
    plt.ylabel("Price")  # Y축: 주가
    plt.legend()
    st.pyplot(plt)

    # 예측된 주가를 기반으로 투자 판단
    last_actual_price = scaler.inverse_transform(test_data[-1].reshape(1, -1))[0][0]
    last_predicted_price = predictions[-1][0]

    st.write(f"현재 주가: {last_actual_price:.2f} 원")
    st.write(f"예측된 주가: {last_predicted_price:.2f} 원")

    if last_predicted_price > last_actual_price:
        st.write("예측에 따르면 주가는 상승할 것으로 예상됩니다. 투자를 고려해보세요.")
    else:
        st.write("예측에 따르면 주가는 하락할 것으로 예상됩니다. 신중히 판단하세요.")

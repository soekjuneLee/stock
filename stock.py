import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import pipeline
import requests
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
    "POSCO홀딩스": "005490.KS",
    "SK하이닉스": "000660.KS",
    "삼성SDI": "006400.KS",
    "LG화학": "051910.KS",
    "현대모비스": "012330.KS",
    "SK이노베이션": "096770.KS",
    "셀트리온": "068270.KQ",
    "삼성바이오로직스": "207940.KS",
    "한국전력": "015760.KS",
    "KT&G": "033780.KS",
    "아모레퍼시픽": "090430.KS",
    "현대건설": "000720.KS",
    "GS건설": "006360.KS",
    "두산에너빌리티": "034020.KS",
    "한화솔루션": "009830.KS",
    "LG이노텍": "011070.KS",
    "SK텔레콤": "017670.KS",
    "카카오뱅크": "323410.KQ",
    "카카오페이": "377300.KQ",
    "HMM": "011200.KS",
    "삼성중공업": "010140.KS",
    "한국조선해양": "009540.KS",
    "현대미포조선": "010620.KS",
    "오리온": "271560.KS",
    "롯데케미칼": "011170.KS",
    "LG유플러스": "032640.KS",
    "SK": "034730.KS",
    "한온시스템": "018880.KS",
    "S-Oil": "010950.KS",
    "코스맥스": "192820.KQ",
    "CJ제일제당": "097950.KS",
    "CJ ENM": "035760.KQ",
    "넷마블": "251270.KQ",
    "펄어비스": "263750.KQ",
    "NC소프트": "036570.KQ",
    "하이브": "352820.KQ",
    "포스코인터내셔널": "047050.KS",
    "한전KPS": "051600.KS",
    "한화에어로스페이스": "012450.KS",
    "대한항공": "003490.KS",
    "제주항공": "089590.KQ",
    "신한지주": "055550.KS",
    "KB금융": "105560.KS",
    "하나금융지주": "086790.KS",
    "우리금융지주": "316140.KS",
    "현대제철": "004020.KS",
    "DB손해보험": "005830.KS",
    "삼성화재": "000810.KS",
    "메리츠금융지주": "138040.KQ",
    "롯데쇼핑": "023530.KS",
    "신세계": "004170.KS",
    "이마트": "139480.KS",
    "BGF리테일": "282330.KQ",
    "F&F": "007700.KS",
    "LF": "093050.KS",
    "한샘": "009240.KS",
    "현대홈쇼핑": "057050.KS",
    "CJ프레시웨이": "051500.KS",
    "롯데제과": "280360.KQ",
    "빙그레": "005180.KS",
    "매일유업": "267980.KQ",
    "풀무원": "017810.KS",
    "동원F&B": "049770.KQ",
    "대상": "001680.KS",
    "남양유업": "003920.KS",
    "한국타이어앤테크놀로지": "161390.KS",
    "금호타이어": "073240.KS",
    "코오롱인더스트리": "120110.KS",
    "효성티앤씨": "298020.KS",
    "SKC": "011790.KS",
    "LS ELECTRIC": "010120.KS",
    "일진머티리얼즈": "020150.KQ",
    "한솔케미칼": "014680.KS",
}

# 사용자 입력 받기
stock_name = st.selectbox("주식 종목 선택", list(ticker_dict.keys()))
ticker = ticker_dict[stock_name]

# 기간 및 데이터 옵션
period = st.selectbox("기간 선택", ["1개월", "3개월", "6개월", "1년"])
period_mapping = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}
interval = st.selectbox("데이터 간격", ["일봉", "주봉"])
interval_mapping = {"일봉": "1d", "주봉": "1wk"}

# 데이터 가져오기
if st.button("데이터 가져오기"):
    data = yf.download(ticker, period=period_mapping[period], interval=interval_mapping[interval])
    st.write(data)

# 뉴스 및 소셜 분석 감정 점수 계산
sentiment_score = 0
if st.button("뉴스 및 소셜 분석"):
    query = stock_name + " 뉴스"
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=cd5098d67b0242df91db6a121493ef4f"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()["articles"]
        st.write("최신 뉴스:")
        for article in articles[:5]:
            st.write(f"- [{article['title']}]({article['url']})")
        
        # 뉴스 감정 분석
        sentiment_analyzer = pipeline("sentiment-analysis")
        for article in articles[:5]:
            title = article["title"]
            sentiment = sentiment_analyzer(title)[0]
            if sentiment["label"] == "POSITIVE":
                sentiment_score += sentiment["score"]
            elif sentiment["label"] == "NEGATIVE":
                sentiment_score -= sentiment["score"]
        st.write(f"총 감정 점수: {sentiment_score}")
    else:
        st.write("뉴스 데이터를 불러올 수 없습니다.")

# LSTM 기반 AI 주가 예측
if st.button("AI 기반 주가 예측"):
    data = yf.download(ticker, period="1y", interval="1d")  # 1년치 데이터로 예측
    data = data[['Close']]  # 'Close' 컬럼만 사용
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 데이터 분리
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

    st.write("테스트 데이터 예측 결과:")
    plt.figure(figsize=(14, 7))
    plt.plot(scaler.inverse_transform(test_data[time_step:]), label="실제 주가")
    plt.plot(predictions, label="예측 주가")
    plt.legend()
    st.pyplot(plt)

    # 뉴스 감정 점수 기반 조정
    sentiment_adjustment = sentiment_score * 0.01  # 감정 점수를 기반으로 주가 조정
    adjusted_predictions = predictions + sentiment_adjustment
    st.write("감정 점수 반영 후 예측 결과:")
    plt.figure(figsize=(14, 7))
    plt.plot(scaler.inverse_transform(test_data[time_step:]), label="실제 주가")
    plt.plot(adjusted_predictions, label="조정된 예측 주가")
    plt.legend()
    st.pyplot(plt)
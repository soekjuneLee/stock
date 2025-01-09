import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import requests
from transformers import pipeline

# UI 구성
st.title("주식 투자 추천 프로그램")

# 주식 종목 딕셔너리
ticker_dict = {
    "삼성전자": "005930.KS",
    "기아": "000270.KS",
    "현대차": "005380.KS",
    "LG전자": "066570.KS",
    "네이버": "035420.KS",
    "카카오": "035720.KS",
}

# 사용자 입력 받기
stock_name = st.selectbox("주식 종목 선택", list(ticker_dict.keys()))
ticker = ticker_dict[stock_name]

# 기간 및 데이터 옵션
period = st.selectbox("기간 선택", ["1개월", "3개월", "6개월", "1년"])
period_mapping = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}
interval = st.selectbox("데이터 간격", ["일봉", "주봉", "분봉"])
interval_mapping = {"일봉": "1d", "주봉": "1wk", "분봉": "1h"}

# 주식 데이터 가져오기
if st.button("데이터 가져오기"):
    data = yf.download(ticker, period=period_mapping[period], interval=interval_mapping[interval])
    st.write(data)

# AI 기반 예측
if st.button("AI 기반 주가 예측"):
    st.write("데이터 준비 중...")
    data = yf.download(ticker, period="1y", interval="1d")  # 1년치 데이터로 예측
    data['Prediction'] = data['Close'].shift(-30)  # 30일 후 예측

    # 피처와 타겟 설정
    X = np.array(data[['Close']])
    X = X[:-30]  # 마지막 30일 데이터 제외
    y = np.array(data['Prediction'])
    y = y[:-30]

    # Train/Test 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression 모델 훈련
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측
    X_forecast = np.array(data[['Close']])[-30:]
    predictions = model.predict(X_forecast)

    # 결과 표시
    st.write("30일 이후 주가 예측:")
    st.line_chart(predictions)

# 뉴스 및 소셜 분석
if st.button("뉴스 및 소셜 분석"):
    query = stock_name + " 뉴스"
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=cd5098d67b0242df91db6a121493ef4f"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()["articles"]
        st.write("최신 뉴스:")
        for article in articles[:5]:
            st.write(f"- [{article['title']}]({article['url']})")
        
        # 뉴스 기사 감정 분석
        try:
            sentiment_analyzer = pipeline("sentiment-analysis")
        except RuntimeError as e:
            st.write(f"Sentiment analysis pipeline error: {e}")

        sentiments = []
        for article in articles[:5]:
            title = article["title"]
            sentiment = sentiment_analyzer(title)[0]
            sentiment_score = sentiment["score"] if sentiment["label"] == "POSITIVE" else -sentiment["score"]
            sentiments.append({"Title": title, "Sentiment": sentiment["label"], "Score": sentiment_score})
        
        st.write("감정 분석 결과:")
        sentiment_df = pd.DataFrame(sentiments)
        st.write(sentiment_df)
    else:
        st.write("뉴스 데이터를 불러올 수 없습니다.")

# 데이터 저장
if st.button("CSV 저장"):
    data.to_csv(f"{stock_name}_data.csv")
    st.write(f"{stock_name} 데이터가 저장되었습니다.")
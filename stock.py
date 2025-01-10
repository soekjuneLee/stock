import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 주식 데이터 가져오기
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")  # 최근 1년간 데이터
    return data

# 뉴스 감정 분석
def get_news_sentiment(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch'  # Yahoo Finance 뉴스 페이지
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    news = soup.find_all('li', class_='js-stream-content')

    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for article in news[:10]:  # 최근 10개의 뉴스만 가져오기
        headline = article.find('h3').text
        sentiment_score = analyzer.polarity_scores(headline)
        sentiments.append(sentiment_score['compound'])
    
    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return average_sentiment

# 투자 여부 예측 모델 학습
def train_model(stock_data, sentiment_data):
    # 데이터 전처리
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data['Target'] = (stock_data['Return'] > 0).astype(int)  # 수익률이 0 이상이면 1 (구매 신호), 아니면 0 (판매 신호)

    # 특성 데이터 (주가 및 감정 분석)
    X = stock_data[['Open', 'High', 'Low', 'Volume']].dropna()
    X['Sentiment'] = sentiment_data
    y = stock_data['Target'].dropna()

    # 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"모델 정확도: {accuracy:.2f}")
    return model

# 예측
def predict_investment(ticker):
    # 주식 데이터 가져오기
    stock_data = get_stock_data(ticker)

    # 뉴스 감정 분석
    sentiment_score = get_news_sentiment(ticker)

    # 모델 훈련 (이 부분은 첫 실행 시에만 필요)
    # sentiment_data는 주가 데이터와 동일한 크기를 가져야 함
    sentiment_data = [sentiment_score] * len(stock_data)  # 전체 기간에 대한 감정 점수 할당
    model = train_model(stock_data, sentiment_data)

    # 예측
    latest_data = stock_data[['Open', 'High', 'Low', 'Volume']].iloc[-1:].copy()
    latest_data['Sentiment'] = sentiment_score
    prediction = model.predict(latest_data)
    
    if prediction == 1:
        st.write(f"{ticker} 주식에 투자하세요!")
    else:
        st.write(f"{ticker} 주식은 투자하지 마세요.")
    
    # 주식 차트 표시
    st.write(f"{ticker}의 주식 데이터 차트")
    fig, ax = plt.subplots()
    stock_data['Close'].plot(ax=ax)
    st.pyplot(fig)

# 프로그램 실행
if __name__ == "__main__":
    ticker = st.text_input("주식 종목을 입력하세요:", 'AAPL')  # 예시: 애플 주식
    if ticker:
        predict_investment(ticker)

import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from newspaper import Article
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
}

# 사용자 입력 받기
stock_name = st.selectbox("주식 종목 선택", list(ticker_dict.keys()))
ticker = ticker_dict[stock_name]

# 기간 및 데이터 옵션
period = st.selectbox("기간 선택", ["1개월", "3개월", "6개월", "1년"])
period_mapping = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}
interval = st.selectbox("데이터 간격", ["일봉", "주봉"])
interval_mapping = {"일봉": "1d", "주봉": "1wk"}

# 뉴스 감정 분석 함수
def get_news_sentiment(ticker):
    # 주식 종목에 대한 최신 뉴스 URL을 가져오기
    news_url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
    article = Article(news_url)
    article.download()
    article.parse()
    
    # 뉴스 텍스트 가져오기
    news_text = article.text
    
    # 감정 분석
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(news_text)
    
    return sentiment['compound']  # -1 (부정) ~ 1 (긍정) 점수 반환

# LSTM 기반 AI 주가 예측
if st.button("AI 기반 주가 예측"):
    # 주식 데이터 다운로드
    data = yf.download(ticker, period=period_mapping[period], interval=interval_mapping[interval])
    data = data[['Close']]  # 'Close' 컬럼만 사용

    if len(data) < 50:  # 데이터가 부족한 경우 예외 처리
        st.error("데이터가 너무 적습니다. 기간을 더 길게 설정해주세요.")
    else:
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

        # 데이터가 충분한지 확인
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            st.error("시간 스텝을 너무 크게 설정했습니다. 더 작은 값으로 설정해 주세요.")
        else:
            # X_train과 X_test reshape
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # 뉴스 감정 분석 추가
            news_sentiment = get_news_sentiment(ticker)  # 뉴스 감정 분석 결과 얻기

            # 감정 점수를 LSTM 모델에 추가 (예시: 마지막 데이터에 추가)
            news_sentiment_scaled = scaler.transform([[news_sentiment]])  # 스케일링
            X_train = np.concatenate([X_train, np.repeat(news_sentiment_scaled, X_train.shape[0], axis=0)], axis=2)
            X_test = np.concatenate([X_test, np.repeat(news_sentiment_scaled, X_test.shape[0], axis=0)], axis=2)

            # LSTM 모델 정의
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 2)))  # 2는 추가된 뉴스 감정의 채널 수
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

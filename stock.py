import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import lstm  # Ensure this module is available and compatible with your project setup

# 1. Load the data
X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 50, True)
# Show the training data preview
st.write("Training Data Preview:", X_train[:5])

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

# 2. Build the model
model = Sequential()
model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
st.write(f"Compilation time: {time.time() - start:.2f} seconds")

# 3. Train the model
model.fit(X_train, y_train, batch_size=512, epochs=1, validation_split=0.05)

# 4. Make predictions and plot results
predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)

# Streamlit to show the plot
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
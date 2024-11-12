import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import pandas as pd
import numpy as np

class DataRepository:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data

    def preprocess_data(self, data, sequence_length):
        sequences = []
        targets = []
        for i in range(len(data) - sequence_length):
            sequences.append(data.iloc[i:i+sequence_length].values)
            targets.append(data.iloc[i+sequence_length].values)
        return np.array(sequences), np.array(targets)

def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # Define file path and sequence length
    file_path = 'path/to/your/data.csv'  # Update this path to your CSV file
    sequence_length = 10

    # Initialize data repository
    data_repo = DataRepository(file_path)

    # Load and preprocess data
    data = data_repo.load_data()
    X, y = data_repo.preprocess_data(data, sequence_length)

    # Create RNN model
    input_shape = (X.shape[1], X.shape[2])  # (sequence_length, number_of_features)
    model = create_rnn_model(input_shape)

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32)

if __name__ == "__main__":
    main() 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from gdrive_utils import download_file_from_gdrive  # Assuming this function is defined in gdrive_utils.py

class DataRepository:
    def __init__(self, drive_service):
        self.drive_service = drive_service

    def load_data(self):
        # Download the file from Google Drive
        download_file_from_gdrive(self.file_id, self.local_file_path)
        data = pd.read_csv(self.local_file_path)
        return data

    def preprocess_data(self, data, max_sequence_length):
        # Tokenize the text data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data['game_state'] + data['instructions'])

        # Convert texts to sequences
        game_state_sequences = tokenizer.texts_to_sequences(data['game_state'])
        instruction_sequences = tokenizer.texts_to_sequences(data['instructions'])

        # Pad sequences
        game_state_sequences = pad_sequences(game_state_sequences, maxlen=max_sequence_length)
        instruction_sequences = pad_sequences(instruction_sequences, maxlen=max_sequence_length)

        # Prepare decoder target data
        decoder_target_data = np.zeros_like(instruction_sequences)
        decoder_target_data[:, :-1] = instruction_sequences[:, 1:]

        return game_state_sequences, instruction_sequences, decoder_target_data, tokenizer

def create_seq2seq_model(vocab_size, embedding_dim, max_sequence_length):
    # Encoder
    encoder_inputs = Input(shape=(max_sequence_length,))
    encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(100, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_sequence_length,))
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Define Google Drive IDs
    gdrive_root_folder_id = '14rV_AfSINfFyUQgZN4wJEgGtJCyzlv0a'  # Root folder ID
    
    # Initialize Google Drive service
    drive_service = initialize_drive_service()  # You'll need this function
    
    # Initialize data repository with drive service
    data_repo = DataRepository(drive_service)  # Modify DataRepository to accept drive_service
    
    # Load and preprocess data directly from Google Drive
    data = data_repo.load_data_from_drive(gdrive_root_folder_id)
    X, y, decoder_target_data, tokenizer = data_repo.preprocess_data(data, max_sequence_length)
    
    # Create seq2seq model
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token
    embedding_dim = 50  # Adjust based on your data
    model = create_seq2seq_model(vocab_size, embedding_dim, max_sequence_length)

    # Train the model
    model.fit([X, y], decoder_target_data, epochs=10, batch_size=32)

if __name__ == "__main__":
    main() 
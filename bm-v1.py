import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from training import DataRepository  # Assuming DataRepository is in training.py
from gdrive_utils import download_file_from_gdrive  # Assuming this function is defined in gdrive_utils.py
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def load_trained_model(model_path):
    return load_model(model_path)

def preprocess_input(game_state_descriptions, tokenizer, max_sequence_length):
    # Tokenize and pad the input game state descriptions
    sequences = tokenizer.texts_to_sequences(game_state_descriptions)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

def predict_actions(model, game_state_sequences, tokenizer):
    # Predict the actions
    predictions = model.predict(game_state_sequences)
    predicted_sequences = np.argmax(predictions, axis=-1)

    # Convert predicted sequences back to text
    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
    predicted_actions = []
    for sequence in predicted_sequences:
        actions = [reverse_word_index.get(index, '') for index in sequence if index != 0]
        predicted_actions.append(' '.join(actions))
    return predicted_actions

def authenticate_gdrive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Creates local webserver and automatically handles authentication
    drive = GoogleDrive(gauth)
    return drive

def upload_to_gdrive(drive, file_path, gdrive_folder_id):
    file = drive.CreateFile({'parents': [{'id': gdrive_folder_id}]})
    file.SetContentFile(file_path)
    file.Upload()
    print(f"Uploaded {file_path} to Google Drive folder {gdrive_folder_id}")

def main():
    # Authenticate Google Drive
    drive = authenticate_gdrive()

    # Load the trained model
    model_path = 'gdrive/My Drive/path/to/your/saved_model.h5'  # Update with your model's path
    model = load_trained_model(model_path)

    # Example: Upload the model to Google Drive
    gdrive_folder_id = 'your_gdrive_folder_id'  # Replace with your Google Drive folder ID
    upload_to_gdrive(drive, model_path, gdrive_folder_id)

    # Load tokenizer and max_sequence_length from training
    # You might need to save and load these from a file or use a consistent method to retrieve them
    tokenizer = ...  # Load your tokenizer
    max_sequence_length = ...  # Set your max sequence length

    # Example game state descriptions
    game_state_descriptions = [
        "Example game state 1",
        "Example game state 2"
    ]

    # Preprocess the input
    game_state_sequences = preprocess_input(game_state_descriptions, tokenizer, max_sequence_length)

    # Predict actions
    predicted_actions = predict_actions(model, game_state_sequences, tokenizer)

    # Output the predicted actions
    for i, actions in enumerate(predicted_actions):
        print(f"Game State {i+1}: {game_state_descriptions[i]}")
        print(f"Predicted Actions: {actions}")

if __name__ == "__main__":
    main() 
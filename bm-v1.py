import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from training import DataRepository  # Assuming DataRepository is in training.py
from gdrive_utils import download_file_from_gdrive  # Assuming this function is defined in gdrive_utils.py
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

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

def load_model_and_tokenizer():
    """
    Load the trained model, tokenizer, and configuration settings from Google Drive
    Returns:
        tuple: (model, tokenizer, max_sequence_length)
    """
    model_folder_id = '14rV_AfSINfFyUQgZN4wJEgGtJCyzlv0a'
    model_file_id = 'your_specific_model_file_id'
    max_sequence_length = 100  # Or load this from config
    
    # Initialize Drive service (you'll need this)
    drive_service = initialize_drive_service()
    
    # Download and load model from Google Drive
    request = drive_service.files().get_media(fileId=model_file_id)
    model = tf.keras.models.load_model(request)
    
    # Similarly for tokenizer if it's stored in Drive
    tokenizer = load_tokenizer(drive_service, model_folder_id)
    
    return model, tokenizer, max_sequence_length

def main():
    
    # Authenticate Google Drive
    drive = authenticate_gdrive()

    # Load model and configuration
    model, tokenizer, max_sequence_length = load_model_and_tokenizer()
    
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
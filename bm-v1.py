import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gdrive_utils import authenticate_gdrive
import io
from googleapiclient.http import MediaIoBaseDownload

def preprocess_input(text, tokenizer, max_sequence_length):
    """Preprocess the input text for prediction."""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

def predict_actions(model, input_sequence, tokenizer):
    """Predict actions from the input sequence using the model."""
    predictions = model.predict(input_sequence)
    predicted_sequences = np.argmax(predictions, axis=-1)

    # Convert predicted sequences back to text
    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
    predicted_actions = []
    for sequence in predicted_sequences:
        actions = [reverse_word_index.get(index, '') for index in sequence if index != 0]
        predicted_actions.append(' '.join(actions))
    return predicted_actions

def main():
    print("Checking for local model files...")
    need_model = not os.path.exists('model.keras')
    need_tokenizer = not os.path.exists('tokenizer.pickle')
    need_game_state = not os.path.exists('game_state.txt')
    
    if need_model or need_tokenizer or need_game_state:
        print("Some files missing locally. Initializing Google Drive connection...")
        drive_service = authenticate_gdrive()
        
        if need_model:
            print("Downloading model from Drive...")
            # Download model
            query = "name='model.keras' and trashed=false"
            results = drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            files = results.get('files', [])
            if not files:
                raise FileNotFoundError("model.keras not found in Google Drive")
            
            model_file_id = files[0]['id']
            request = drive_service.files().get_media(fileId=model_file_id)
            model_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(model_buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            model_buffer.seek(0)
            
            # Save model temporarily
            with open('model.keras', 'wb') as f:
                f.write(model_buffer.getvalue())
        
        if need_tokenizer:
            print("Downloading tokenizer from Drive...")
            # Download tokenizer
            query = "name='tokenizer.pickle' and trashed=false"
            results = drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            files = results.get('files', [])
            if not files:
                raise FileNotFoundError("tokenizer.pickle not found in Google Drive")
            
            tokenizer_file_id = files[0]['id']
            request = drive_service.files().get_media(fileId=tokenizer_file_id)
            tokenizer_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(tokenizer_buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            tokenizer_buffer.seek(0)
            
            # Save tokenizer temporarily
            with open('tokenizer.pickle', 'wb') as f:
                f.write(tokenizer_buffer.getvalue())
        
        if need_game_state:
            print("Downloading game state from Drive...")
            query = "name='game_state.txt' and trashed=false"
            results = drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            files = results.get('files', [])
            if not files:
                raise FileNotFoundError("game_state.txt not found in Google Drive")
            
            game_state_file_id = files[0]['id']
            request = drive_service.files().get_media(fileId=game_state_file_id)
            game_state_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(game_state_buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            game_state_buffer.seek(0)
            
            # Save game state temporarily
            with open('game_state.txt', 'wb') as f:
                f.write(game_state_buffer.getvalue())
    else:
        print("Using existing local files...")

    # Now load the model
    model = tf.keras.models.load_model('model.keras')
    
    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Read the game state
    with open('game_state.txt', 'r') as f:
        game_state_description = f.read()

    # Preprocess the input
    max_sequence_length = 100
    input_sequence = preprocess_input(game_state_description, tokenizer, max_sequence_length)

    # Predict actions
    predicted_actions = predict_actions(model, input_sequence, tokenizer)

    # Output the predicted actions
    print("Predicted Actions:")
    for actions in predicted_actions:
        print(actions)

    # Clean up downloaded files
    if os.path.exists('model.keras'):
        os.remove('model.keras')
    if os.path.exists('tokenizer.pickle'):
        os.remove('tokenizer.pickle')
    if os.path.exists('game_state.txt'):
        os.remove('game_state.txt')

    # After loading the model and tokenizer, add:
    print("Model config:", model.get_config())
    print("\nTokenizer config:", tokenizer.get_config())
    print("\nTokenizer word index:", dict(list(tokenizer.word_index.items())[:5]))
    print("\nTokenizer special tokens:", {
        'pad_token': tokenizer.pad_token,
        'start_token': getattr(tokenizer, 'start_token', None),
        'end_token': getattr(tokenizer, 'end_token', None)
    })

if __name__ == "__main__":
    main()

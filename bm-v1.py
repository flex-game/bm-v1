import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gdrive_utils import authenticate_gdrive
import io
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

def preprocess_input(text, tokenizer, max_sequence_length):
    """Preprocess the input text for prediction."""
    # Create main sequence
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    
    # Create decoder input sequence (shifted target sequence starting with start token)
    # Assuming start token is index 1 (common practice), adjust if different
    decoder_input = np.zeros((1, max_sequence_length))
    decoder_input[:, 0] = 1  # Set start token
    
    return [padded_sequences, decoder_input]

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

    # After predicting actions
    print("\nPredicted Actions:")
    with open('action_prediction.txt', 'w') as f:
        for actions in predicted_actions:
            print(actions)  # Print each prediction
            f.write(actions + '\n')
    
    print("\nUploading to Google Drive...")
    drive_service = authenticate_gdrive()
    
    try:
        media = MediaFileUpload('action_prediction.txt', 
                              mimetype='text/plain',
                              resumable=True)
        
        # Set the specific folder ID where models are stored
        file_metadata = {
            'name': 'action_prediction.txt',
            'mimeType': 'text/plain',
            'parents': ['14rV_AfSINfFyUQgZN4wJEgGtJCyzlv0a']  # Specific folder ID
        }
        
        # Check if file exists in this specific folder
        query = "name='action_prediction.txt' and '14rV_AfSINfFyUQgZN4wJEgGtJCyzlv0a' in parents and trashed=false"
        results = drive_service.files().list(q=query).execute()
        files = results.get('files', [])
        
        if files:
            file = drive_service.files().update(
                fileId=files[0]['id'],
                media_body=media,
                fields='id'
            ).execute()
            print(f"Updated existing file in Drive (ID: {file.get('id')})")
        else:
            file = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            print(f"Created new file in Drive (ID: {file.get('id')})")
    except Exception as e:
        print(f"Upload failed: {str(e)}")

    # Clean up local files
    os.remove('action_prediction.txt')  # Add this to your existing cleanup
    if os.path.exists('model.keras'):
        os.remove('model.keras')
    if os.path.exists('tokenizer.pickle'):
        os.remove('tokenizer.pickle')
    if os.path.exists('game_state.txt'):
        os.remove('game_state.txt')

    # After loading the model and tokenizer, add:
    print("Model config:", model.get_config())
    print("\nTokenizer special tokens:", {
        'oov_token': tokenizer.oov_token,
        'word_counts': len(tokenizer.word_counts),
        'document_count': tokenizer.document_count,
        'num_words': tokenizer.num_words
    })

if __name__ == "__main__":
    main()

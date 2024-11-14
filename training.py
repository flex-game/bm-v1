import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gdrive_utils import authenticate_gdrive
import io
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import shutil
import pickle
import os
import numpy as np

def create_seq2seq_model(vocab_size, embedding_dim, max_sequence_length):
    # Encoder
    encoder_inputs = tf.keras.Input(shape=(max_sequence_length,))
    encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(100, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = tf.keras.Input(shape=(max_sequence_length,))
    decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_and_preprocess_data(drive_service, dataset_file_id, max_sequence_length=100):
    """Load and preprocess the dataset from Google Drive."""
    print("Downloading dataset from Drive...")
    request = drive_service.files().get_media(fileId=dataset_file_id)
    dataset_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(dataset_buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dataset_buffer.seek(0)
    
    print("Loading dataset into pandas...")
    df = pd.read_csv(dataset_buffer)
    print(f"Loaded {len(df)} training examples")
    
    print("Creating and fitting tokenizer...")
    tokenizer = Tokenizer()
    all_texts = df['Frame Analysis'].tolist() + df['Action Analysis'].tolist()
    tokenizer.fit_on_texts(all_texts)
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1} words")
    
    print("Converting texts to sequences and padding...")
    X = tokenizer.texts_to_sequences(df['Frame Analysis'])
    y = tokenizer.texts_to_sequences(df['Action Analysis'])
    
    X = pad_sequences(X, maxlen=max_sequence_length, padding='post')
    y = pad_sequences(y, maxlen=max_sequence_length, padding='post')
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    return X, y, tokenizer

def find_dataset_file(drive_service, folder_id):
    """
    Search for dataset.csv in the specified Google Drive folder.
    Returns the file ID if found, raises an exception if not found.
    """
    query = f"'{folder_id}' in parents and name='dataset.csv' and trashed=false"
    results = drive_service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    
    files = results.get('files', [])
    if not files:
        raise FileNotFoundError("dataset.csv not found in the specified folder")
    
    return files[0]['id']

def save_to_drive(drive_service, model, tokenizer, folder_id):
    """Save the model and tokenizer to Google Drive."""
    # Save model with .keras extension
    model.save('model.keras')
    
    # Save tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Upload model to Drive
    model_metadata = {
        'name': 'model.keras',
        'parents': [folder_id]
    }
    with open('model.keras', 'rb') as model_file:
        media = MediaIoBaseUpload(model_file, mimetype='application/octet-stream')
        drive_service.files().create(
            body=model_metadata,
            media_body=media,
            fields='id'
        ).execute()
    
    # Upload tokenizer to Drive
    tokenizer_metadata = {
        'name': 'tokenizer.pickle',
        'parents': [folder_id]
    }
    with open('tokenizer.pickle', 'rb') as tokenizer_file:
        media = MediaIoBaseUpload(tokenizer_file, mimetype='application/octet-stream')
        drive_service.files().create(
            body=tokenizer_metadata,
            media_body=media,
            fields='id'
        ).execute()
    
    # Cleanup temporary files
    try:
        os.remove('model.keras')
        os.remove('tokenizer.pickle')
        print("Temporary files deleted successfully.")
    except Exception as e:
        print(f"Error deleting temporary files: {e}")

def preprocess_input(text, tokenizer, max_sequence_length):
    """Preprocess the input text for prediction."""
    # Encoder input
    encoder_sequences = tokenizer.texts_to_sequences([text])
    encoder_input = pad_sequences(encoder_sequences, maxlen=max_sequence_length)
    
    # Decoder input - start with START token
    # Adjust START_TOKEN_ID based on your tokenizer
    START_TOKEN_ID = 1  # Ensure this matches your tokenizer's start token
    decoder_input = np.array([[START_TOKEN_ID]])
    decoder_input = pad_sequences(decoder_input, maxlen=max_sequence_length, padding='post')
    
    return [encoder_input, decoder_input]

def predict_actions(model, input_sequences, tokenizer):
    """Predict actions from the input sequence using the model."""
    encoder_input, decoder_input = input_sequences
    
    # Initialize variables for beam search or greedy search
    max_length = 100  # Maximum length of generated sequence
    target_sequence = decoder_input
    
    # Generate the sequence one word at a time
    generated_sequence = []
    for i in range(max_length):
        # Predict next token
        output = model.predict([encoder_input, target_sequence])
        sampled_token_index = np.argmax(output[0, -1, :])
        
        # Exit condition: either hit max length or end token
        if sampled_token_index == 0:  # Assuming 0 is padding/end token
            break
            
        generated_sequence.append(sampled_token_index)
        
        # Update the decoder input for the next iteration
        target_sequence = np.zeros((1, max_length))
        target_sequence[0, :len(generated_sequence)] = generated_sequence
        target_sequence = pad_sequences(target_sequence, maxlen=max_length, padding='post')

    # Convert predicted sequences back to text
    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
    predicted_actions = [reverse_word_index.get(idx, '') for idx in generated_sequence if idx != 0]
    return [' '.join(predicted_actions)]

def main():
    print("Initializing Google Drive connection...")
    drive_service = authenticate_gdrive()
    root_folder_id = '14rV_AfSINfFyUQgZN4wJEgGtJCyzlv0a'
    
    print("Looking for dataset.csv...")
    dataset_file_id = find_dataset_file(drive_service, root_folder_id)
    print(f"Found dataset.csv, proceeding with training...")
    
    print("\nPreparing data for training...")
    X, y, tokenizer = load_and_preprocess_data(
        drive_service, 
        dataset_file_id
    )
    
    print("\nCreating model...")
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    max_sequence_length = 100
    model = create_seq2seq_model(vocab_size, embedding_dim, max_sequence_length)
    
    print("\nBeginning training...")
    print(f"Training on {len(X)} examples with {vocab_size} unique words")
    model.fit(
        [X, y], 
        y,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1  # This shows the progress bar
    )
    
    print("\nTraining complete. Saving model to Drive...")
    save_to_drive(drive_service, model, tokenizer, root_folder_id)
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    main() 
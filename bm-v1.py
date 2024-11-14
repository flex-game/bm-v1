import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gdrive_utils import authenticate_gdrive, download_file_from_gdrive

def load_model_and_tokenizer():
    """Load the trained model and tokenizer from local files."""
    model = tf.keras.models.load_model('model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

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
    # Authenticate Google Drive
    drive_service = authenticate_gdrive()

    # Ask for the file ID of the .txt file
    file_id = input("Enter the Google Drive file ID of the game state description (.txt): ")

    # Download the .txt file
    download_file_from_gdrive(drive_service, file_id, 'game_state.txt')

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Read the game state description
    with open('game_state.txt', 'r') as file:
        game_state_description = file.read()

    # Preprocess the input
    max_sequence_length = 100  # Ensure this matches the training configuration
    input_sequence = preprocess_input(game_state_description, tokenizer, max_sequence_length)

    # Predict actions
    predicted_actions = predict_actions(model, input_sequence, tokenizer)

    # Output the predicted actions
    print("Predicted Actions:")
    for actions in predicted_actions:
        print(actions)

    # Clean up the downloaded file
    os.remove('game_state.txt')

if __name__ == "__main__":
    main()

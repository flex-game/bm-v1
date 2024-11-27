import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Concatenate, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

# Load and preprocess the dataset
def load_and_preprocess_data(csv_path, max_sequence_length=50, num_words=10000):
    df = pd.read_csv(csv_path)

    # Text preprocessing
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df['stats_shot'])
    sequences = tokenizer.texts_to_sequences(df['stats_shot'])
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)

    # Image preprocessing
    def preprocess_image_from_url(url, target_size=(224, 224)):
        response = requests.get(url)
        img = load_img(BytesIO(response.content), target_size=target_size)
        img_array = img_to_array(img)
        img_array = resnet_preprocess_input(img_array)
        return img_array

    image_urls = df['image_path'].values
    images = np.array([preprocess_image_from_url(url) for url in image_urls])

    # Action labels
    actions = df.drop(columns=['image_path', 'stats_shot']).values

    return images, padded_sequences, actions, tokenizer

# Create the multi-modal model
def create_multimodal_model(vocab_size, embedding_dim, max_sequence_length):
    # Image model
    image_input = Input(shape=(224, 224, 3), name='input_layer')
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    image_features = base_model(image_input)

    # Text model
    text_input = Input(shape=(max_sequence_length,), name='input_layer_2')
    text_embedding = Embedding(vocab_size, embedding_dim)(text_input)
    text_features = LSTM(128)(text_embedding)

    # Combine features
    combined_features = Concatenate()([image_features, text_features])
    dense_output = Dense(166, activation='sigmoid')(combined_features)

    # Model
    model = tf.keras.Model(inputs=[image_input, text_input], outputs=dense_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def setup_google_drive():
    """Authenticate and return a Google Drive client."""
    scope = ["https://www.googleapis.com/auth/drive"]
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set or is empty.")
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    service = build('drive', 'v3', credentials=creds)
    return service

def download_from_drive(service, folder_id, filename):
    # Search for the file in the specified folder
    query = f"'{folder_id}' in parents and name='{filename}'"
    results = service.files().list(q=query, spaces='drive').execute()
    files = results.get('files', [])
    
    if not files:
        raise FileNotFoundError(f"'{filename}' not found in the specified Google Drive folder")
    
    file_id = files[0]['id']
    request = service.files().get_media(fileId=file_id)
    
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%")
    
    # Save the file locally
    output_path = os.path.join(os.getcwd(), filename)
    with open(output_path, 'wb') as f:
        f.write(fh.getvalue())
    
    return output_path

# Main function to load data, create model, and train
def main():
    # Download dataset from Google Drive
    service = setup_google_drive()
    folder_id = "14rV_AfSINfFyUQgZN4wJEgGtJCyzlv0a"
    csv_path = download_from_drive(service, folder_id, 'dataset.csv')
    
    images, text_sequences, actions, tokenizer = load_and_preprocess_data(csv_path)

    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    max_sequence_length = 50

    model = create_multimodal_model(vocab_size, embedding_dim, max_sequence_length)

    # Create TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        {'input_layer': images, 'input_layer_2': text_sequences}, 
        actions
    ))
    dataset = dataset.shuffle(buffer_size=1000).batch(32)

    # Train the model
    model.fit(dataset, epochs=10)

if __name__ == "__main__":
    main() 
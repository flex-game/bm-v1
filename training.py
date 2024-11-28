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
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io
import os
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import logging
import time
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info("Starting image preprocessing for url: " + url)
        response = requests.get(url)
        img = load_img(BytesIO(response.content), target_size=target_size)
        img_array = img_to_array(img)
        img_array = resnet_preprocess_input(img_array)
        return img_array

    image_urls = df['image_path'].values
    images = np.array([preprocess_image_from_url(url) for url in image_urls])

    # Action labels
    actions = df.drop(columns=['image_path', 'stats_shot']).values

    logger.info("Image preprocessing complete")
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
    logger.info(f"Starting download of {filename} from Drive...")
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
    
    logger.info(f"Finished downloading {filename}")
    return output_path

def upload_model_to_drive(service, folder_id, model_path):
    try:
        file_metadata = {
            'name': 'trained_model.keras',
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(model_path, 
                              mimetype='application/octet-stream',
                              resumable=True)
        
        file = service.files().create(body=file_metadata,
                                    media_body=media,
                                    fields='id').execute()
        print(f"Model uploaded to Drive with file ID: {file.get('id')}")
        
        # Clean up local file
        os.remove(model_path)
        print(f"Local model file {model_path} removed")
        
    except Exception as e:
        print(f"Error uploading model to Drive: {str(e)}")

def check_preprocessed_images(service, folder_id):
    logger.info("Checking for existing preprocessed images in Drive...")
    try:
        results = service.files().list(
            q=f"name='preprocessed_images.zip' and '{folder_id}' in parents",
            fields="files(id, name)").execute()
        files = results.get('files', [])
        
        if files:
            logger.info("Found existing preprocessed images, downloading...")
            file_id = files[0]['id']
            request = service.files().get_media(fileId=file_id)
            
            # Download the zip file
            with open('preprocessed_images.zip', 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    _, done = downloader.next_chunk()
                    logger.info("Downloading preprocessed images...")
            
            # Extract zip file
            with zipfile.ZipFile('preprocessed_images.zip', 'r') as zip_ref:
                zip_ref.extractall('preprocessed_images')
            
            # Clean up zip file
            os.remove('preprocessed_images.zip')
            logger.info("Successfully extracted preprocessed images")
            return True
            
        return False
    except Exception as e:
        logger.error(f"Error checking/downloading preprocessed images: {str(e)}")
        return False

def zip_and_upload_preprocessed_images(service, folder_id):
    logger.info("Zipping preprocessed images...")
    with zipfile.ZipFile('preprocessed_images.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('preprocessed_images'):
            for file in files:
                zipf.write(os.path.join(root, file))
    
    logger.info("Uploading preprocessed images to Drive...")
    file_metadata = {
        'name': 'preprocessed_images.zip',
        'parents': [folder_id]
    }
    
    media = MediaFileUpload('preprocessed_images.zip',
                           mimetype='application/zip',
                           resumable=True)
    
    file = service.files().create(body=file_metadata,
                                media_body=media,
                                fields='id').execute()
    
    # Clean up local zip file
    os.remove('preprocessed_images.zip')
    logger.info("Preprocessed images uploaded and local zip removed")

# Main function to load data, create model, and train
def main():
    start_time = time.time()
    logger.info("Starting initialization...")
    
    logger.info("Setting up Google Drive connection...")
    service = setup_google_drive()
    
    logger.info("Downloading dataset from Drive...")
    folder_id = "14rV_AfSINfFyUQgZN4wJEgGtJCyzlv0a"
    csv_path = download_from_drive(service, folder_id, 'dataset.csv')
    
    logger.info("Loading and preprocessing dataset...")
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

    # Check for existing preprocessed images before processing
    if not check_preprocessed_images(service, folder_id):
        logger.info("No existing preprocessed images found, starting preprocessing...")
        # Your existing image preprocessing code here
        process_images(image_paths)
        # After preprocessing is complete, zip and upload
        zip_and_upload_preprocessed_images(service, folder_id)
    else:
        logger.info("Using existing preprocessed images")
    
    logger.info("Starting model training...")
    # Train the model
    model.fit(dataset, epochs=10)

    logger.info("Training complete. Saving model...")
    model.save('trained_model.keras')  # Save locally first
    upload_model_to_drive(service, "14rV_AfSINfFyUQgZN4wJEgGtJCyzlv0a", 'trained_model.keras')

    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time/60:.2f} minutes")

if __name__ == "__main__":
    main() 
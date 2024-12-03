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
import pickle

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
    action_columns = df.drop(columns=['image_path', 'stats_shot']).columns
    action_mapping = {i: action for i, action in enumerate(action_columns)}

    logger.info("Image preprocessing complete")
    return images, padded_sequences, actions, tokenizer, action_mapping

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
            
            # Create directory if it doesn't exist
            if not os.path.exists('preprocessed_images'):
                os.makedirs('preprocessed_images')
            
            # Download and extract zip file
            zip_path = 'preprocessed_images/preprocessed_images.zip'
            with open(zip_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    _, done = downloader.next_chunk()
                    logger.info("Downloading preprocessed images...")
            
            # Extract zip file, flattening the directory structure
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for zip_info in zip_ref.filelist:
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, 'preprocessed_images')
            
            # Clean up zip file
            os.remove(zip_path)
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
    
    # First check if preprocessed images exist in Drive
    if check_preprocessed_images(service, folder_id):
        logger.info("Loading preprocessed images from local directory...")
        # Use relative paths
        images = np.load('preprocessed_images/images.npy')
        text_sequences = np.load('preprocessed_images/text_sequences.npy')
        actions = np.load('preprocessed_images/actions.npy')
        
        # Still need to process the CSV to get the tokenizer and action_mapping
        df = pd.read_csv(csv_path)
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(df['stats_shot'])
        action_columns = df.drop(columns=['image_path', 'stats_shot']).columns
        action_mapping = {i: action for i, action in enumerate(action_columns)}
    else:
        logger.info("Processing images from scratch...")
        images, text_sequences, actions, tokenizer, action_mapping = load_and_preprocess_data(csv_path)

        # Save preprocessed images
        if not os.path.exists('preprocessed_images'):
            os.makedirs('preprocessed_images')
        np.save('preprocessed_images/images.npy', images)
        np.save('preprocessed_images/text_sequences.npy', text_sequences)
        np.save('preprocessed_images/actions.npy', actions)
        
        # Zip and upload preprocessed images
        zip_and_upload_preprocessed_images(service, folder_id)
    
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    max_sequence_length = 50

    '''
    # Original model training code
    model = create_multimodal_model(vocab_size, embedding_dim, max_sequence_length)
    dataset = tf.data.Dataset.from_tensor_slices((
        {'input_layer': images, 'input_layer_2': text_sequences}, 
        actions
    ))
    dataset = dataset.shuffle(buffer_size=1000).batch(32)
    model.fit(dataset, epochs=10)
    model.save('trained_model.keras')
    upload_model_to_drive(service, folder_id, 'trained_model.keras')
    '''
    
    # Save preprocessing info
    preprocessing_info = {
        'tokenizer': tokenizer,
        'max_sequence_length': max_sequence_length,
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'action_mapping': action_mapping
    }
    
    with open('preprocessing_info.pkl', 'wb') as f:
        pickle.dump(preprocessing_info, f)
    logger.info("Saved preprocessing information")

    # Upload preprocessing info to Drive
    file_metadata = {
        'name': 'preprocessing_info.pkl',
        'parents': [folder_id]
    }
    media = MediaFileUpload('preprocessing_info.pkl',
                           mimetype='application/octet-stream',
                           resumable=True)
    service.files().create(body=file_metadata,
                          media_body=media,
                          fields='id').execute()
    logger.info("Uploaded preprocessing information to Drive")
    
    # Clean up local preprocessing file
    os.remove('preprocessing_info.pkl')

    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time/60:.2f} minutes")

if __name__ == "__main__":
    main() 
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
from utils.text_preprocessing import preprocess_texts
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.processing import ScriptProcessor
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Load and preprocess the dataset
def load_and_preprocess_data(csv_path, max_sequence_length=50, num_words=10000):
    df = pd.read_csv(csv_path)

    # Text preprocessing
    padded_sequences, tokenizer = preprocess_texts(df['stats_shot'], num_words, max_sequence_length)

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

def get_matching_files_from_s3(image_bucket, text_bucket, actions_bucket):
    """Get lists of matching files from all three buckets"""
    s3_client = boto3.client('s3')
    
    def list_bucket_objects(bucket):
        objects = []
        paginator = s3_client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=bucket):
                if 'Contents' in page:
                    # Extract just the filenames without paths
                    filenames = [Path(obj['Key']).stem for obj in page['Contents']]
                    objects.extend(filenames)
            return set(objects)
        except ClientError as e:
            logger.error(f"Error accessing bucket {bucket}: {str(e)}")
            raise

    # Get all filenames (without extensions) from each bucket
    image_files = list_bucket_objects(image_bucket)
    text_files = list_bucket_objects(text_bucket)
    action_files = list_bucket_objects(actions_bucket)
    
    # Find common files across all buckets
    common_files = sorted(list(image_files & text_files & action_files))
    
    if not common_files:
        raise ValueError("No matching files found across all three buckets")
    
    logger.info(f"Found {len(common_files)} matching files across all buckets")
    return common_files

def load_and_preprocess_data_from_s3(image_bucket, text_bucket, actions_bucket, 
                                    max_sequence_length=50, num_words=10000):
    """Load and preprocess data from S3 buckets using matching filenames"""
    logger.info("Starting data loading from S3 buckets...")
    s3_client = boto3.client('s3')
    
    # Get list of matching files
    common_files = get_matching_files_from_s3(image_bucket, text_bucket, actions_bucket)
    
    # Initialize lists for data
    texts = []
    images = []
    actions = []
    
    # Load data for each matching set of files
    for filename in common_files:
        try:
            # Load text
            text_response = s3_client.get_object(
                Bucket=text_bucket, 
                Key=f"{filename}.txt"
            )
            text_content = text_response['Body'].read().decode('utf-8').strip()
            texts.append(text_content)
            
            # Load image
            img_response = s3_client.get_object(
                Bucket=image_bucket, 
                Key=f"{filename}.jpg"
            )
            img_data = img_response['Body'].read()
            img = load_img(BytesIO(img_data), target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = resnet_preprocess_input(img_array)
            images.append(img_array)
            
            # Load action
            action_response = s3_client.get_object(
                Bucket=actions_bucket, 
                Key=f"{filename}.txt"
            )
            action_data = action_response['Body'].read().decode('utf-8').strip()
            action_values = [int(x) for x in action_data.split(',')]
            actions.append(action_values)
            
        except ClientError as e:
            logger.error(f"Error loading files for {filename}: {str(e)}")
            continue
        except Exception as e:
            logger.error(f"Error processing files for {filename}: {str(e)}")
            continue
        
        # Log progress periodically
        if len(texts) % 100 == 0:
            logger.info(f"Processed {len(texts)} files...")
    
    # Convert to numpy arrays
    images = np.array(images)
    actions = np.array(actions)
    
    # Preprocess texts
    padded_sequences, tokenizer = preprocess_texts(texts, num_words, max_sequence_length)
    
    # Create action mapping (assuming actions are binary and consistent across files)
    action_mapping = {i: f"action_{i}" for i in range(actions.shape[1])}
    
    logger.info(f"Successfully processed {len(texts)} files")
    logger.info(f"Final shapes - Images: {images.shape}, Text: {padded_sequences.shape}, Actions: {actions.shape}")
    
    return images, padded_sequences, actions, tokenizer, action_mapping

def prepare_action_labels():
    """Extract and prepare unique action labels from training files"""
    s3_client = boto3.client('s3')
    actions_bucket = 'bm-v1-training-actions'
    unique_actions = set()
    
    # List all files in actions bucket
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=actions_bucket):
        for obj in page['Contents']:
            response = s3_client.get_object(Bucket=actions_bucket, Key=obj['Key'])
            action_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Add each action to our set of unique actions
            for action in action_data['actions_by_player']:
                unique_actions.add(action)
    
    # Convert to sorted list for consistent ordering
    action_list = sorted(list(unique_actions))
    action_mapping = {action: idx for idx, action in enumerate(action_list)}
    
    # Save action mapping to S3
    s3_client.put_object(
        Bucket='bm-v1-model',
        Key='action_mapping.json',
        Body=json.dumps(action_mapping)
    )
    
    return action_mapping, len(action_list)

def setup_sagemaker_training():
    """Configure and start SageMaker training job"""
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Get number of output classes
    _, num_actions = prepare_action_labels()
    
    # Define hyperparameters
    hyperparameters = {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_actions': num_actions,
        'embedding_dim': 50,
        'max_sequence_length': 50,
        'num_words': 10000
    }
    
    # Configure estimator
    estimator = TensorFlow(
        entry_point='model_train.py',  # Your training script
        role=role,
        instance_count=1,
        instance_type='ml.p3.2xlarge',  # GPU instance
        framework_version='2.11',
        py_version='py39',
        hyperparameters=hyperparameters,
        output_path='s3://bm-v1-model/trained_models'
    )
    
    # Define data channels
    data_channels = {
        'training': f's3://bm-v1-training-images',
        'text': f's3://bm-v1-training-text',
        'actions': f's3://bm-v1-training-actions'
    }
    
    # Start training
    estimator.fit(data_channels)

# Main function to load data, create model, and train
def main():
    start_time = time.time()
    logger.info("Starting initialization...")
    
    logger.info("Setting up Google Drive connection...")
    service = setup_google_drive()
    
    logger.info("Downloading dataset from Drive...")
    folder_id = "14rV_AfSINfFyUQgZN4wJEgGtJCyzlv0a"
    csv_path = download_from_drive(service, folder_id, 'dataset.csv')
    
    # S3 bucket names
    IMAGE_BUCKET = 'bm-v1-training-images'
    TEXT_BUCKET = 'bm-v1-training-text'
    ACTIONS_BUCKET = 'bm-v1-training-actions'
    
    # First check if preprocessed images exist in Drive
    if check_preprocessed_images(service, folder_id):
        logger.info("Loading preprocessed images from local directory...")
        images = np.load('preprocessed_images/images.npy')
        text_sequences = np.load('preprocessed_images/text_sequences.npy')
        actions = np.load('preprocessed_images/actions.npy')
        
        # Load fresh data from S3 for tokenizer and action_mapping
        _, _, _, tokenizer, action_mapping = load_and_preprocess_data_from_s3(
            IMAGE_BUCKET, TEXT_BUCKET, ACTIONS_BUCKET
        )
    else:
        logger.info("Processing images from scratch...")
        images, text_sequences, actions, tokenizer, action_mapping = load_and_preprocess_data_from_s3(
            IMAGE_BUCKET, TEXT_BUCKET, ACTIONS_BUCKET
        )
        
        # Save preprocessed data
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
    setup_sagemaker_training() 
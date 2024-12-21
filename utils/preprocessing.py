import boto3
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from io import BytesIO
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

# Load action mapping from S3
def load_action_mapping():
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket='bm-v1-model', Key='action_mapping.json')
    action_mapping = json.loads(response['Body'].read().decode('utf-8'))
    return action_mapping

# Preprocess image
def preprocess_image(image_data):
    """
    Args:
        image_data: Either a file path or BytesIO object
    """
    if isinstance(image_data, BytesIO):
        image = tf.keras.preprocessing.image.load_img(image_data, target_size=(224, 224))
    else:
        image = tf.keras.preprocessing.image.load_img(image_data, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# Load the tokenizer from S3
def load_tokenizer():
    """Load the tokenizer from S3."""
    s3_client = boto3.client('s3')
    try:
        training_id = os.getenv('LATEST_STABLE_TRAINING')
        if not training_id:
            raise ValueError("LATEST_STABLE_TRAINING not set in .env")
            
        tokenizer_path = f"trained_models/bm-v1-training-{training_id}/output/preprocessing_info.pkl"
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        
        response = s3_client.get_object(
            Bucket='bm-v1-model',
            Key=tokenizer_path
        )
        return pickle.loads(response['Body'].read())
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        raise

# Update the preprocess_text function to optionally load the tokenizer
def preprocess_text(text, tokenizer=None, max_sequence_length=50):
    """
    Preprocess text using saved tokenizer.
    Args:
        text: String to process
        tokenizer: Optional tokenizer instance. If None, loads from S3.
        max_sequence_length: Max length for padding
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences[0]

def get_matching_files(images_path, text_path, actions_path):
    """Get files that exist in all three directories"""
    image_files = set(os.listdir(images_path))
    text_files = set(os.listdir(text_path))
    action_files = set(os.listdir(actions_path))
    
    # Find common files
    common = image_files & text_files & action_files
    return sorted(list(common))

def load_data(images_path, text_path, actions_path, filenames):
    """Load data from local paths instead of S3"""
    images = []
    texts = []
    labels = []
    
    for filename in filenames:
        # Load image
        with open(os.path.join(images_path, filename), 'rb') as f:
            images.append(f.read())
            
        # Load text
        with open(os.path.join(text_path, filename), 'r') as f:
            texts.append(f.read().strip())
            
        # Load action
        with open(os.path.join(actions_path, filename), 'r') as f:
            labels.append(json.load(f))
    
    return images, texts, labels
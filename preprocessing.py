import boto3
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from io import BytesIO
import pickle

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
    # Adjust the path to where your latest model's tokenizer is stored
    response = s3_client.get_object(
        Bucket='bm-v1-model', 
        Key='trained_models/tokenizer.pkl'
    )
    return pickle.loads(response['Body'].read())

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
import boto3
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load action mapping from S3
def load_action_mapping():
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket='bm-v1-model', Key='action_mapping.json')
    action_mapping = json.loads(response['Body'].read().decode('utf-8'))
    return action_mapping

# Preprocess image
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# Preprocess text
def preprocess_text(text, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences
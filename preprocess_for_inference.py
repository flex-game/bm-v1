import pickle
import requests
from io import BytesIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import numpy as np

# Load preprocessing information
with open('preprocessing_info.pkl', 'rb') as f:
    preprocessing_info = pickle.load(f)

tokenizer = preprocessing_info['tokenizer']
max_sequence_length = preprocessing_info['max_sequence_length']

def preprocess_text(text):
    """Preprocess text for model input."""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

def preprocess_image(image_url):
    """Preprocess image for model input."""
    response = requests.get(image_url)
    img = load_img(BytesIO(response.content), target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = resnet_preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array 
import logging
import sagemaker
import boto3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.models import Model, load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from src.dataset import Dataset  # Import the Dataset class

s3 = boto3.client('s3')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def train():
    logger = setup_logging()
    logger.info("=== Starting Train ===")

    # Load the dataset
    dataset = Dataset(dataset_path='data/dataset.json', image_folder='data/images')

    # Get the embeddings and labels
    image_embeddings = dataset.image_embeddings
    text_embeddings = dataset.text_embeddings
    labels = dataset.labels
    num_classes = len(np.unique(labels))

    # One-hot encode the labels
    labels = np.eye(num_classes)[labels]

    # Define input shapes
    input_shape_image = image_embeddings.shape[1:]  # Shape of image embeddings
    input_shape_text = text_embeddings.shape[1:]  # Shape of text embeddings

    # Build the model
    model = build_model(input_shape_image, input_shape_text, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit([image_embeddings, text_embeddings], labels, epochs=10, batch_size=32)

    # Save the model
    model.save('model.keras')

    logger.info("=== Training Process Complete ===")

def predict():
    logger = setup_logging()
    logger.info("=== Starting Predict ===")

    # Load the model
    model = load_model('model.keras')

    # Example prediction
    image_data = np.random.rand(1, 224, 224, 3)  # Replace with actual image data
    text_data = np.random.rand(1, 5000)  # Replace with actual text data
    prediction = model.predict([image_data, text_data])
    
    print(prediction)       
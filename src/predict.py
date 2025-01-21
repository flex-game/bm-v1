import logging
import sagemaker
import tensorflow as tf
import boto3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

s3 = boto3.client('s3')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def predict():
    pass

def build_model(input_shape_image, input_shape_text, num_classes, image_dataset, text_dataset):
    '''
    Builds the model using ResNet50 for image feature extraction 
    and TFIDF for text feature extraction. 
    '''
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape_image)
    for layer in resnet_base.layers:
        layer.trainable = False  # Freeze ResNet50 layers

    # Add custom layers for image feature extraction
    x = resnet_base.output
    x = Dense(256, activation='relu')(x)

    # Text feature extraction using TFIDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
    # Use the Dataset class to get text data
    text_data = [text for text in text_dataset]
    text_embeddings = tfidf_vectorizer.fit_transform(text_data).toarray()
    text_input = Input(shape=(text_embeddings.shape[1],))
    y = Dense(256, activation='relu')(text_input)  # Example custom layer

    # Concatenate image and text features
    combined = Concatenate()([x, y])

    # Add fully connected layers for classification
    z = Dense(128, activation='relu')(combined)
    output = Dense(num_classes, activation='softmax')(z)  # Use 'sigmoid' for binary classification

    # Create the model
    model = Model(inputs=[resnet_base.input, text_input], outputs=output)

    return model

def train():
    logger = setup_logging()
    logger.info("=== Starting Train ===")

    # Define input shapes and number of classes
    input_shape_image = (224, 224, 3)  # Example input shape for ResNet50
    input_shape_text = (5000,)  # Example input shape for TFIDF
    num_classes = 10  # Example number of classes

    # Build the model
    model = build_model(input_shape_image, input_shape_text, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Example training data
    image_data = np.random.rand(100, 224, 224, 3)  # Replace with actual image data
    text_data = np.random.rand(100, 5000)  # Replace with actual text data
    labels = np.random.randint(0, num_classes, 100)  # Replace with actual labels
    labels = np.eye(num_classes)[labels]  # One-hot encode the labels

    # Train the model
    model.fit([image_data, text_data], labels, epochs=10, batch_size=32)

    logger.info("=== Training Process Complete ===")

def main():
    train()
    return
    
if __name__ == "__main__":
    main()
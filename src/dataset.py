import boto3
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
import os
import json
import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
import logging

s3 = boto3.client('s3')

class Dataset:
    
    def __init__(self, dataset_path: str, image_folder: str = None, embeddings_path=None) -> None:
        logging.info("Initializing Dataset with path: %s and image folder: %s", dataset_path, image_folder)
        self.dataset_path = dataset_path
        self.image_folder = image_folder
        self.raw_text = self.load_raw_text()
        self.image_embeddings = self.load_and_embed_images()
        self.text_embeddings = self.embed_text()
        self.cleaned_data = self.clean_data()
        self.labels = self.create_labels()

    def load_raw_text(self):
        '''
        Loads raw text from the dataset
        '''
        logging.info("Loading raw text...")
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def load_and_embed_images(self):
        '''
        Loads and embeds images using ResNet50
        '''
        logging.info("Loading and embedding images")
        preprocessed_images = self.preprocess_images()
        image_embeddings = self.embed_images()
        return image_embeddings

    def _preprocess_images(image_folder: str) -> np.ndarray:
        '''
        Helper function to load and preprocess images for ResNet50 embedding
        '''
        logging.info("Loading and preprocessing images from folder: %s", image_folder)
        image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.jpg')]
        preprocessed_images = self.preprocess_images(image_paths)
        
        return preprocessed_images

    def _embed_images(preprocessed_images: np.ndarray) -> np.ndarray:
        '''
        Embeds preprocessed images using ResNet50
        '''
        logging.debug("Embedding preprocessed image data")
    
        # Load ResNet50 model pre-trained on ImageNet
        base_model = ResNet50(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

        # Assumes preprocessed_images is an array of preprocessed images
        embeddings = model.predict(preprocessed_images)

        # Flatten the embeddings
        image_embeddings = embeddings.reshape(embeddings.shape[0], -1)
        logging.debug("Image embeddings generated")

        return image_embeddings

    def embed_text(self) -> np.ndarray:
        '''
        Embeds text using TF-IDF
        '''
        logging.debug("Embedding text data")
        # Initialize CountVectorizer
        vectorizer = CountVectorizer()

        # Fit and transform the text data
        text_embeddings = vectorizer.fit_transform(self.cleaned_data['game_state'].apply(lambda x: ' '.join(x)))

        self.text_embeddings = text_embeddings.toarray()
        logging.debug("Text embeddings generated")
        
        return self.text_embeddings

    def clean_data(self):
        '''
        Cleans the raw data, dropping some actions,
        protecting against null values, removing punctuation, 
        and handling missing turn data for image filenames.
        '''
        logging.debug("Cleaning data")
        df = pd.DataFrame(self.raw_text)

        # Drops all actions but the first (multi-action prediction not supported yet)
        df['actions'] = df['actions'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
        
        # Clean actions data
        df['actions'] = df['actions'].astype(str).str.lower().str.replace('[^\w\s]', '', regex=True)

        # Clean game_state data
        df['game_state'] = df['game_state'].apply(lambda x: [text.lower().replace('[^\w\s]', '') for text in x])

        # Handle null or missing values in 'actions' column
        df['actions'] = df['actions'].fillna('')

        # Handle null or missing values in 'game_state' column
        df['game_state'] = df['game_state'].apply(lambda x: x if isinstance(x, list) else [])
        df['game_state'] = df['game_state'].apply(lambda x: [text if text is not None else '' for text in x])

        # Confirm valid "turn" value
        df['turn'] = df['turn'].apply(lambda x: x if isinstance(x, str) and x.strip() else None)
        df.dropna(subset=['turn'], inplace=True)

        # Handle missing turn data for image filenames
        df['screenshot'] = df['screenshot'].apply(lambda x: x if x in self.image_data else None)
        df.dropna(subset=['screenshot'], inplace=True)

        # Encode actions as labels
        df['actions_encoded'] = self.label_encoder.fit_transform(df['actions'])

        # Store cleaned data
        self.cleaned_data = df
        logging.debug("Example cleaned data: %s", df.head())
        logging.debug("Cleaned data columns: %s", df.columns)
        logging.debug("Cleaned data stored")

        return df

    def create_labels(self):
        '''
        Creates and returns one-hot encoded labels from actions data.
        '''
        logging.debug("Creating labels")
        # Implement the logic to create one-hot encoded labels from actions data
        # This is a placeholder and should be replaced with the actual implementation
        return np.zeros((len(self.cleaned_data), len(self.label_encoder.classes_)))

    def decode_actions(self, encoded_actions) -> list:
        '''
        Decodes numerical labels back to action text.
        '''
        logging.debug("Decoding actions")
        return self.label_encoder.inverse_transform(encoded_actions)

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

s3 = boto3.client('s3')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

class Dataset:
    
    def __init__(self, dataset_path: str, image_folder: str = None) -> None:
        self.raw_data = self.load_raw_data(dataset_path)
        self.cleaned_data = None
        self.embeddings = None
        self.image_folder = image_folder
        self.image_data = self.load_images() if image_folder else None

    def load_raw_data(self, dataset_path: str) -> pd.DataFrame:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def load_images(self) -> dict:
        image_data = {}
        for filename in os.listdir(self.image_folder):
            if filename.endswith('.jpg'):
                img_path = os.path.join(self.image_folder, filename)
                image_data[filename] = preprocess_image(img_path)
        return image_data

    def clean(self, *args: Any, **kwargs: Any) -> Any:

        df = pd.DataFrame(self.raw_data)

        # Drops all actions but the first (multi-action prediction not supported yet)
        df['actions'] = df['actions'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')

        # Clean actions data
        df['actions'] = df['actions'].str.lower().str.replace('[^\w\s]', '')

        # Clean game_state data
        df['game_state'] = df['game_state'].apply(lambda x: [text.lower().replace('[^\w\s]', '') for text in x])

        # Handle null or missing values in 'actions' column
        df['actions'] = df['actions'].fillna('')

        # Handle null or missing values in 'game_state' column
        df['game_state'] = df['game_state'].apply(lambda x: x if isinstance(x, list) else [])
        df['game_state'] = df['game_state'].apply(lambda x: [text if text is not None else '' for text in x])

        # Store cleaned data
        self.cleaned_data = df

        return df
    
    def embed_text(self, *args: Any, **kwargs: Any) -> Any:
        '''
        Embed text using TF-IDF
        '''

        # Initialize CountVectorizer
        vectorizer = CountVectorizer()

        # Fit and transform the text data
        text_embeddings = vectorizer.fit_transform(self.cleaned_data['game_state'].apply(lambda x: ' '.join(x)))

        self.embeddings = text_embeddings.toarray()
        return self.embeddings

    def embed_image(self, *args: Any, **kwargs: Any) -> Any:
        '''
        Embed images using ResNet50
        '''

        # Load ResNet50 model pre-trained on ImageNet
        base_model = ResNet50(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

        image_embeddings = []
        for img_name in self.cleaned_data['screenshot']:
            if img_name in self.image_data:
                img_data = self.image_data[img_name]
                embedding = model.predict(img_data)
                image_embeddings.append(embedding.flatten())
            else:
                image_embeddings.append(np.zeros((2048,)))  # Assuming 2048 is the output size of avg_pool layer

        self.embeddings = np.array(image_embeddings)
        return self.embeddings

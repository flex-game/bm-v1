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
    
    def __init__(self, raw_data: pd.DataFrame) -> None:
        self.raw_data = raw_data
        self.cleaned_data = None
        self.embeddings = None

    def clean(self, *args: Any, **kwargs: Any) -> Any:

        df = pd.DataFrame(self.raw_data)

        # Clean actions
        df['actions'] = df['actions'].apply(lambda x: list(set(x)))  # dedupe
        df['actions'] = df['actions'].apply(lambda x: [action.replace('\n', ' ').replace('\r', ' ').strip() for action in x])  # remove special characters

        # Store cleaned data
        self.cleaned_data = df

        return df
    
    def embed_text(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def embed_image(self, *args: Any, **kwargs: Any) -> Any:
        '''
        Embed images using ResNet50
        '''

        # Load ResNet50 model pre-trained on ImageNet
        base_model = ResNet50(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

        image_embeddings = []
        for img_path in args:
            img_data = preprocess_image(img_path)
            embedding = model.predict(img_data)
            image_embeddings.append(embedding.flatten())

        self.embeddings = np.array(image_embeddings)
        return self.embeddings

        
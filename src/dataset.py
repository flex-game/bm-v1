import boto3
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
import os
import json
import numpy as np
import pandas as pd

s3 = boto3.client('s3')

def download_file(bucket, key, local_path):
    s3.download_file(bucket, key, local_path)

def preprocess_image(image_path) -> np.array:
    return image

def preprocess_text(json_path) -> np.array:
    return text_vector

def preprocess_actions(json_path) -> np.array:
    return action_vector

def download_and_preprocess_data(bucket, image_keys, text_keys, action_keys, local_dir) -> tuple[np.array, np.array, np.array]:
    return images, texts, actions

import unittest
import os
import numpy as np
from src.dataset import Dataset, preprocess_image
import pandas as pd

class TestDataset(unittest.TestCase):

    def setUp(self):

        self.dataset_path = 'data/test_dataset.json'
        self.image_folder = 'data/assets/test_images'
        
        # Create a Dataset instance
        self.dataset = Dataset(self.dataset_path, self.image_folder)

    def test_load_raw_json(self):
        
        # Test loading raw JSON data
        raw_data = self.dataset.load_raw_json(self.dataset_path)
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertGreater(len(raw_data), 0)

    def test_load_images(self):
        # Test loading images
        image_data = self.dataset.load_images()
        self.assertIsInstance(image_data, dict)
        self.assertGreater(len(image_data), 0)

    def test_clean_json(self):
        # Test cleaning JSON data
        cleaned_data = self.dataset.clean_json()
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertIn('actions_encoded', cleaned_data.columns)

    def test_embed_text(self):
        # Test text embedding
        self.dataset.clean_json()  # Ensure data is cleaned before embedding
        text_embeddings = self.dataset.embed_text()
        self.assertIsInstance(text_embeddings, np.ndarray)
        self.assertEqual(text_embeddings.shape[0], len(self.dataset.cleaned_data))

    def test_embed_image(self):
        # Test image embedding
        self.dataset.clean_json()  # Ensure data is cleaned before embedding
        image_embeddings = self.dataset.embed_image()
        self.assertIsInstance(image_embeddings, np.ndarray)
        self.assertEqual(image_embeddings.shape[0], len(self.dataset.cleaned_data))

    def test_decode_actions(self):
        # Test decoding actions
        self.dataset.clean_json()  # Ensure data is cleaned before encoding
        encoded_actions = self.dataset.cleaned_data['actions_encoded']
        decoded_actions = self.dataset.decode_actions(encoded_actions)
        self.assertEqual(len(decoded_actions), len(encoded_actions))
        self.assertTrue(all(isinstance(action, str) for action in decoded_actions))

if __name__ == '__main__':
    unittest.main() 
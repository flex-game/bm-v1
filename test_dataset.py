import unittest
import logging
from src.dataset import Dataset
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset_path = 'data/test_dataset.json'
        self.image_folder = 'data/assets/test'
        
        # Create a Dataset instance
        self.dataset = Dataset(self.dataset_path, self.image_folder)
        logging.debug("Dataset initialized.")

    def test_load_raw_data(self):
        # Test loading raw data
        raw_data = self.dataset.load_raw_data()
        logging.info("Raw data loaded: %s", raw_data.head())
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertGreater(len(raw_data), 0)

    def test_clean_data(self):
        # Test cleaning data
        cleaned_data = self.dataset.clean_data()
        logging.info("Cleaned data: %s", cleaned_data.head())
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertGreater(len(cleaned_data), 0)

    def test_load_and_embed_images(self):
        # Test loading and embedding images
        image_embeddings = self.dataset.load_and_embed_images()
        logging.info("Image embeddings generated: %s", image_embeddings[:5])
        self.assertIsInstance(image_embeddings, np.ndarray)
        self.assertEqual(image_embeddings.shape[0], len(self.dataset.cleaned_data))

    def test_embed_text(self):
        # Log the initial state of the data
        logging.debug("Original data: %s", self.dataset.cleaned_data.head())

        # Call the method to test
        text_embeddings = self.dataset.embed_text()
        logging.debug("Text embeddings generated: %s", text_embeddings[:5])

        # Add assertions as needed
        self.assertIsNotNone(text_embeddings)
        self.assertIsInstance(text_embeddings, np.ndarray)
        self.assertEqual(text_embeddings.shape[0], len(self.dataset.cleaned_data))

    def test_embed_image(self):
        # Test image embedding
        self.dataset.clean_json()  # Ensure data is cleaned before embedding
        image_embeddings = self.dataset.embed_image()
        logging.debug("Image embeddings generated: %s", image_embeddings[:5])
        self.assertIsInstance(image_embeddings, np.ndarray)
        self.assertEqual(image_embeddings.shape[0], len(self.dataset.cleaned_data))

    def test_decode_actions(self):
        # Test decoding actions
        self.dataset.clean_json()  # Ensure data is cleaned before encoding
        encoded_actions = self.dataset.cleaned_data['actions_encoded']
        decoded_actions = self.dataset.decode_actions(encoded_actions)
        logging.debug("Decoded actions: %s", decoded_actions[:5])
        self.assertEqual(len(decoded_actions), len(encoded_actions))
        self.assertTrue(all(isinstance(action, str) for action in decoded_actions))

if __name__ == '__main__':
    unittest.main()
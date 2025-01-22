import unittest
import logging
from src.dataset import Dataset
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset_path = 'data/test_dataset.json'
        self.image_folder = 'data/assets/test'
        
        # Create a Dataset instance
        self.dataset = Dataset(self.dataset_path, self.image_folder)
        logging.debug("Dataset initialized.")

    def test_load_raw_json(self):
        # Test loading raw JSON data
        raw_data = self.dataset.load_raw_json(self.dataset_path)
        logging.info("Raw data loaded: %s", raw_data.head())
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertGreater(len(raw_data), 0)

    def test_load_images(self):
        # Test loading images
        image_data = self.dataset.load_images()
        logging.info("Image data loaded: %d images", len(image_data))
        self.assertIsInstance(image_data, dict)
        self.assertGreater(len(image_data), 0)

    def test_clean_json(self):
        # Test cleaning JSON data
        cleaned_data = self.dataset.clean_json()
        logging.debug("Cleaned data: %s", cleaned_data.head())
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertIn('actions_encoded', cleaned_data.columns)
    
    '''
    def test_embed_text(self):
        # Log the initial state of the data
        logging.debug("Original data: %s", self.dataset.raw_data.head())

        # Clean the data
        self.dataset.clean_json()
        logging.debug("Cleaned data: %s", self.dataset.cleaned_data.head())

        # Log the state before embedding
        logging.debug("Data to be embedded: %s", self.dataset.cleaned_data['game_state'].head())

        # Call the method to test
        text_embeddings = self.dataset.embed_text()
        logging.debug("Text embeddings generated: %s", text_embeddings[:5])

        # Add assertions as needed
        self.assertIsNotNone(text_embeddings)

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
'''

if __name__ == '__main__':
    unittest.main() 
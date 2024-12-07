import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gdrive_utils import authenticate_gdrive
import io
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from openai_utils import generate_frame_description
import logging
import uuid
from datetime import datetime
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_and_tokenize_text(text, tokenizer, max_sequence_length):
    """Clean and tokenize the text for model input."""
    # Clean the text
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()  # Remove markdown code block markers

    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    
    return padded_sequences

def preprocess_input(raw_text, image_url, tokenizer, max_sequence_length):
    """Preprocess the input text and image for prediction."""
    response = requests.get(image_url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")
    
    # Load and process image
    img = load_img(BytesIO(response.content), target_size=(224, 224))
    
    # Clean and tokenize text
    padded_sequences = clean_and_tokenize_text(raw_text, tokenizer, max_sequence_length)
    
    # Image preprocessing
    img_array = img_to_array(img)
    img_array = resnet_preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return {'input_layer': img_array, 'input_layer_2': padded_sequences}

def predict_actions(model, input_data):
    """Predict actions from the input data using the model."""
    predictions = model.predict(input_data)
    predicted_actions = np.argmax(predictions, axis=-1)
    return predicted_actions

def upload_to_drive(service, file_path, folder_id, file_name):
    """Upload a file to Google Drive."""
    # Determine MIME type based on file extension
    mime_type = 'image/jpeg' if file_name.endswith(('.jpg', '.jpeg')) else 'text/plain'
    
    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    logger.info(f"Uploaded {file_name} to Drive with ID: {file.get('id')}")
    return file.get('id')

def get_preprocessing_info(drive_service):
    """Download and load preprocessing info from Google Drive."""
    logger.info("Downloading preprocessing info from Drive...")
    
    # Find the preprocessing info file
    query = "name='preprocessing_info.pkl' and trashed=false"
    results = drive_service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    
    files = results.get('files', [])
    if not files:
        raise FileNotFoundError("preprocessing_info.pkl not found in Google Drive")
    
    file_id = files[0]['id']
    request = drive_service.files().get_media(fileId=file_id)
    preprocessing_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(preprocessing_buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    preprocessing_buffer.seek(0)
    
    # Load the preprocessing info
    preprocessing_info = pickle.load(preprocessing_buffer)
    logger.info("Successfully loaded preprocessing info")
    
    return preprocessing_info

def main():
    logger.info("Starting initialization...")
    drive_service = authenticate_gdrive()
    
    # Get preprocessing info
    preprocessing_info = get_preprocessing_info(drive_service)
    
    # Extract components
    tokenizer = preprocessing_info['tokenizer']
    max_sequence_length = preprocessing_info['max_sequence_length']
    vocab_size = preprocessing_info['vocab_size']
    embedding_dim = preprocessing_info['embedding_dim']
    action_mapping = preprocessing_info['action_mapping']

    logger.info("Checking for local model files...")
    need_model = not os.path.exists('trained_model.h5')
    
    if need_model:
        logger.info("Model file missing locally. Downloading from Drive...")
        # Download model
        query = "name='trained_model.h5' and trashed=false"
        results = drive_service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        files = results.get('files', [])
        if not files:
            raise FileNotFoundError("trained_model.h5 not found in Google Drive")
        
        model_file_id = files[0]['id']
        request = drive_service.files().get_media(fileId=model_file_id)
        model_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(model_buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        model_buffer.seek(0)
        
        # Save model temporarily
        with open('trained_model.h5', 'wb') as f:
            f.write(model_buffer.getvalue())
    else:
        logger.info("Using existing local files...")

    # Now load the model
    model = tf.keras.models.load_model('trained_model.h5')

    # Prompt user for image URL
    image_url = input("Please enter the image URL: ")
    
    # Generate text sequence using OpenAI
    frame_prompt_path = 'system_prompts/frame_analysis_system_prompt.txt'
    raw_text = generate_frame_description(image_url, frame_prompt_path)

    # Create unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex
    text_filename = f"text_{timestamp}_{unique_id}.txt"
    image_filename = f"image_{timestamp}_{unique_id}.jpg"
    prediction_filename = f"prediction_{timestamp}_{unique_id}.txt"

    # Save the text and image locally
    with open(text_filename, 'w') as f:
        f.write(raw_text)
    logger.info(f"Saved text to {text_filename}")

    # Download and save the image locally
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(image_filename, 'wb') as f:
            f.write(response.content)
        logger.info(f"Saved image to {image_filename}")
    else:
        logger.error(f"Failed to download image. Status code: {response.status_code}")
        return

    # Upload text and image to Google Drive
    usage_logs_folder_id = '1NMIZXKWUxAX428xNfTl36fjZiryqyWDZ'
    upload_to_drive(drive_service, text_filename, usage_logs_folder_id, text_filename)
    upload_to_drive(drive_service, image_filename, usage_logs_folder_id, image_filename)

    # Preprocess the input
    input_data = preprocess_input(raw_text, image_url, tokenizer, max_sequence_length)

    # Predict actions
    predicted_actions = predict_actions(model, input_data)

    # Save and upload the prediction
    with open(prediction_filename, 'w') as f:
        for action_index in predicted_actions:
            action_name = action_mapping.get(action_index, "Unknown Action")
            f.write(f"Action Index: {action_index}, Action: {action_name}\n")
    logger.info(f"Saved prediction to {prediction_filename}")
    upload_to_drive(drive_service, prediction_filename, usage_logs_folder_id, prediction_filename)

    # Clean up local files
    os.remove(text_filename)
    os.remove(image_filename)
    os.remove(prediction_filename)
    if os.path.exists('trained_model.h5'):
        os.remove('trained_model.h5')

    logger.info("Process complete.")

if __name__ == "__main__":
    main()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import logging
import requests
from io import BytesIO
import json
import boto3
from utils.create_text_data import generate_frame_description
from utils.text_preprocessing import preprocess_texts
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_with_endpoint(endpoint_name, image_data, text_content):
    """Make predictions using the SageMaker endpoint."""
    try:
        logger.info("Making prediction with SageMaker endpoint...")
        
        runtime = boto3.client('sagemaker-runtime')
        
        max_sequence_length = int(os.getenv('MAX_SEQUENCE_LENGTH', 512))
        padded_text = preprocess_texts([text_content], max_sequence_length)
        
        payload = {
            "instances": [{
                "inputs": image_data.tolist(),
                "inputs_1": padded_text[0].tolist()
            }]
        }
        
        logger.info(f"Payload structure: {list(payload['instances'][0].keys())}")
        
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        prediction = json.loads(response['Body'].read())
        return prediction
    except Exception as e:
        logger.error(f"Full error response: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response.get('Body', ''), 'read'):
            logger.error(f"Error response body: {e.response['Body'].read()}")
        raise

def preprocess_input(image_url):
    """Preprocess the input image for prediction."""
    response = requests.get(image_url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")
    
    # Load and process image
    img = load_img(BytesIO(response.content), target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = resnet_preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def main():
    logger.info("Starting initialization...")
    
    # Get the endpoint name from environment variable
    endpoint_name = os.getenv('SAGEMAKER_ENDPOINT_NAME')
    print(f"Loaded endpoint name: {endpoint_name}")  # Debug print
    if not endpoint_name:
        raise ValueError("SAGEMAKER_ENDPOINT_NAME environment variable not set")
    
    # Get image URL from user
    image_url = input("Please enter the image URL: ")
    
    # Generate text description using OpenAI
    frame_prompt_path = 'utils/oai_system_prompts/frame_analysis_system_prompt.txt'
    text_content = generate_frame_description(image_url, frame_prompt_path)
    
    # Preprocess both inputs
    image_data = preprocess_input(image_url)
    
    # Get predictions from the endpoint
    prediction_response = predict_with_endpoint(endpoint_name, image_data, text_content)
    predicted_actions = prediction_response['predictions']
    
    # Print predictions
    for action_index in predicted_actions:
        action_name = action_mapping.get(str(action_index), "Unknown Action")
        print(f"Predicted Action: {action_name}")

    logger.info("Process complete.")

if __name__ == "__main__":
    main()

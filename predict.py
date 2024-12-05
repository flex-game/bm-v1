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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_preprocessing_info(endpoint_name):
    """Fetch preprocessing info from the SageMaker endpoint."""
    logger.info("Fetching preprocessing info from SageMaker endpoint...")
    
    runtime = boto3.client('sagemaker-runtime')
    
    # Make a request to get preprocessing info
    # We'll use a special flag in the request to indicate we want preprocessing info
    payload = {
        "request_type": "get_preprocessing_info"
    }
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    preprocessing_info = json.loads(response['Body'].read())
    return preprocessing_info

def predict_with_endpoint(endpoint_name, image_data):
    """Make predictions using the SageMaker endpoint."""
    logger.info("Making prediction with SageMaker endpoint...")
    
    runtime = boto3.client('sagemaker-runtime')
    
    # Prepare the payload
    payload = {
        "request_type": "prediction",
        "image_data": image_data.tolist()  # Convert numpy array to list for JSON serialization
    }
    
    # Make prediction request
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    # Parse response
    prediction = json.loads(response['Body'].read())
    return prediction

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
    if not endpoint_name:
        raise ValueError("SAGEMAKER_ENDPOINT_NAME environment variable not set")
    
    # Get preprocessing info from the endpoint
    preprocessing_info = get_preprocessing_info(endpoint_name)
    action_mapping = preprocessing_info['action_mapping']
    
    # Get image URL from user
    image_url = input("Please enter the image URL: ")
    
    # Generate text description using OpenAI
    frame_prompt_path = 'utils/oai_system_prompts/frame_analysis_system_prompt.txt'
    text_content = generate_frame_description(image_url, frame_prompt_path)
    
    # Preprocess both inputs
    image_data = preprocess_input(image_url)
    
    # Get predictions from the endpoint
    prediction_response = predict_with_endpoint(endpoint_name, {
        'image_data': image_data.tolist(),
        'text_data': text_content
    })
    predicted_actions = prediction_response['predictions']
    
    # Print predictions
    for action_index in predicted_actions:
        action_name = action_mapping.get(str(action_index), "Unknown Action")
        print(f"Predicted Action: {action_name}")

    logger.info("Process complete.")

if __name__ == "__main__":
    main()
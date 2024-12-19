import json
import pickle
import tensorflow as tf
import numpy as np
from utils.preprocessing import preprocess_image, preprocess_text

def model_fn(model_dir):
    """Load the model and tokenizer from the model directory."""
    model = tf.keras.models.load_model(f"{model_dir}/model")
    
    with open(f"{model_dir}/tokenizer.pkl", 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return model, tokenizer

def input_fn(request_body, request_content_type):
    """Convert the request data into model input format."""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data['instances'][0]
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_and_tokenizer):
    """Make prediction using the model."""
    model, tokenizer = model_and_tokenizer
    
    # Process inputs
    image_data = np.array(input_data['inputs'])
    text_data = input_data['inputs_1']
    
    # Ensure text is properly tokenized using saved tokenizer
    if isinstance(text_data, str):
        text_sequence = tokenizer.texts_to_sequences([text_data])
        text_data = tf.keras.preprocessing.sequence.pad_sequences(
            text_sequence, 
            maxlen=50,  # Make sure this matches training
            padding='post'
        )
    
    # Make prediction
    prediction = model.predict([
        np.expand_dims(image_data, 0),
        np.expand_dims(text_data, 0)
    ])
    
    return prediction

def output_fn(prediction, response_content_type):
    """Convert prediction to response format."""
    if response_content_type == 'application/json':
        return json.dumps({
            'predictions': prediction.tolist()
        })
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}") 
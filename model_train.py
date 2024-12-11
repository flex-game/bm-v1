import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, LSTM, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
import boto3
from utils.text import preprocess_texts

# Load action mapping from S3
def load_action_mapping():
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket='bm-v1-model', Key='action_mapping.json')
    action_mapping = json.loads(response['Body'].read().decode('utf-8'))
    return action_mapping

# Preprocess image
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# Preprocess text
def preprocess_text(text, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

# Define the model
def create_model(num_actions, max_sequence_length, num_words, embedding_dim):
    # Image model
    image_input = Input(shape=(224, 224, 3), name='image_input')
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
    x = GlobalAveragePooling2D()(base_model.output)

    # Text model
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    embedding = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_sequence_length)(text_input)
    y = LSTM(128)(embedding)

    # Combine models
    combined = Concatenate()([x, y])
    z = Dense(256, activation='relu')(combined)
    z = Dense(num_actions, activation='softmax')(z)

    model = Model(inputs=[image_input, text_input], outputs=z)
    return model

def main():
    # Load action mapping
    action_mapping = load_action_mapping()
    num_actions = len(action_mapping)

    # Hyperparameters
    max_sequence_length = 50
    num_words = 10000
    embedding_dim = 50
    learning_rate = 0.001

    # Create model
    model = create_model(num_actions, max_sequence_length, num_words, embedding_dim)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Load and preprocess data (placeholder)
    # You will need to implement data loading and preprocessing here
    # images, texts, labels = load_data()

    # Tokenizer for text data
    tokenizer = Tokenizer(num_words=num_words)
    # tokenizer.fit_on_texts(texts)  # Fit tokenizer on your text data

    # Preprocess images and texts
    # preprocessed_images = np.array([preprocess_image(img) for img in images])
    # preprocessed_texts = np.array([preprocess_text(txt, tokenizer, max_sequence_length) for txt in texts])

    # Train the model
    # model.fit([preprocessed_images, preprocessed_texts], labels, epochs=10, batch_size=32)

    # Save the model
    model.save('s3://bm-v1-model/trained_models/model.h5')

if __name__ == "__main__":
    main() 
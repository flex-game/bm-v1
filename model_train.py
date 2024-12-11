import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, LSTM, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
import numpy as np
from io import BytesIO
from preprocessing import load_action_mapping, preprocess_image, preprocess_text
from utils.s3_utils import s3_get_matching_files, s3_load_data

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
    num_words = 2500  # Reduced by 3/4 from the original 10,000
    embedding_dim = 50
    learning_rate = 0.001

    # Create model
    model = create_model(num_actions, max_sequence_length, num_words, embedding_dim)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Load common files
    common_files = s3_get_matching_files('bm-v1-training-images', 'bm-v1-training-text', 'bm-v1-training-actions')

    # Load data from S3
    raw_images, texts, labels = s3_load_data('bm-v1-training-images', 'bm-v1-training-text', 'bm-v1-training-actions', common_files)

    # Preprocess images
    images = np.array([preprocess_image(BytesIO(img)) for img in raw_images])

    # Tokenizer for text data
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)

    # Preprocess texts
    preprocessed_texts = np.array([preprocess_text(txt, tokenizer, max_sequence_length) for txt in texts])

    # Convert labels to categorical
    label_indices = [[action_mapping[action] for action in label] for label in labels]
    categorical_labels = tf.keras.utils.to_categorical(label_indices, num_classes=num_actions)

    # Train the model
    model.fit([images, preprocessed_texts], categorical_labels, epochs=10, batch_size=32)

    # Save the model
    model.save('s3://bm-v1-model/trained_models/model.h5')

if __name__ == "__main__":
    main()
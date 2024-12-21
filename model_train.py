import logging
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, LSTM, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
import numpy as np
from io import BytesIO
from utils.preprocessing import (
    load_data,           # Local file version
    get_matching_files,  # Local file version
    preprocess_image, 
    preprocess_text
)
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import boto3
import pickle
import json

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

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
    logger = setup_logging()
    logger.info("=== Starting Model Training Process ===")
    
    try:
        # Debug: List contents of data directories
        logger.info(f"Contents of training dir: {os.listdir('/opt/ml/input/data/training')}")
        logger.info(f"Contents of text dir: {os.listdir('/opt/ml/input/data/text')}")
        logger.info(f"Contents of actions dir: {os.listdir('/opt/ml/input/data/actions')}")
        
        # Load action mapping
        logger.info("Loading action mapping...")
        mapping_path = '/opt/ml/input/data/mapping/action_mapping.json'
        with open(mapping_path, 'r') as f:
            action_mapping = json.load(f)
        num_actions = len(action_mapping)
        logger.info(f"Found {num_actions} unique actions")

        # Get file lists directly
        training_files = set(os.listdir('/opt/ml/input/data/training'))
        text_files = set(os.listdir('/opt/ml/input/data/text'))
        action_files = set(os.listdir('/opt/ml/input/data/actions'))
        
        common_files = sorted(list(training_files & text_files & action_files))
        logger.info(f"Found {len(common_files)} matching files")

        # Then load the data using those common files
        logger.info("Loading data...")
        raw_images, texts, labels = load_data(training_path, 
                                            text_path, 
                                            actions_path, 
                                            common_files)
        logger.info(f"Loaded {len(raw_images)} images, {len(texts)} texts, and {len(labels)} labels")

        # Hyperparameters
        max_sequence_length = 50
        num_words = 2500
        embedding_dim = 50
        learning_rate = 0.001
        logger.info("Hyperparameters: max_seq_len=%d, vocab_size=%d, embed_dim=%d, lr=%.4f", 
                    max_sequence_length, num_words, embedding_dim, learning_rate)

        # Create model
        logger.info("Creating and compiling model...")
        model = create_model(num_actions, max_sequence_length, num_words, embedding_dim)
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Preprocess images
        logger.info("Preprocessing images...")
        images = np.array([preprocess_image(BytesIO(img)) for img in raw_images])
        logger.info("Image array shape: %s", images.shape)

        # Tokenizer for text data
        logger.info("Tokenizing text data...")
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(texts)
        logger.info("Vocabulary size: %d", len(tokenizer.word_index))

        # Preprocess texts
        logger.info("Converting texts to sequences...")
        preprocessed_texts = np.array([preprocess_text(txt, tokenizer, max_sequence_length) 
                                     for txt in texts])
        logger.info("Text array shape: %s", preprocessed_texts.shape)

        # Convert labels to categorical (taking only first action)
        logger.info("Processing labels...")
        label_indices = [action_mapping[label[0]] if label else 0 for label in labels]  # Take first action or default to 0
        categorical_labels = tf.keras.utils.to_categorical(label_indices, num_classes=num_actions)
        logger.info("Label array shape: %s", categorical_labels.shape)

        # Add callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.00001
            )
        ]

        # Train the model
        logger.info("=== Starting Training ===")
        history = model.fit(
            [images, preprocessed_texts],
            categorical_labels,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Log final metrics
        logger.info("=== Training Complete ===")
        logger.info("Final training accuracy: %.4f", history.history['accuracy'][-1])
        logger.info("Final validation accuracy: %.4f", history.history['val_accuracy'][-1])

        # Save tokenizer BEFORE model saving
        logger.info("Saving tokenizer...")
        tokenizer_path = '/opt/ml/model/tokenizer.pkl'
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Tokenizer saved to {tokenizer_path}")

        # Save model (only to /opt/ml/model/)
        logger.info("Saving model...")
        model.save('/opt/ml/model/model.keras')
        logger.info("Model saved to /opt/ml/model/model.keras")

        # Add debug logging to verify contents
        logger.info(f"Contents of /opt/ml/model/: {os.listdir('/opt/ml/model/')}")

        logger.info("=== Training Process Complete ===")

    except Exception as e:
        logger.error("Training failed: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()
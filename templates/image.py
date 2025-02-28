import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

# Load the saved model
saved_model_path = 'drive/MyDrive/archive/best_model.h5'
model = load_model(saved_model_path)

# Load the tokenizer
with open('drive/MyDrive/archive/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Maximum length of captio
max_length = 35  # Corrected to 35

# Load VGG16 model for feature extraction
vgg_model = VGG16()
feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)

# Function to map index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate caption for an image
def predict_caption(model, image_features, tokenizer, max_length):
    # Add start tag for generation process
    in_text = 'startseq'
    # Iterate over the max length of sequence
    for i in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Predict next word
        yhat = model.predict([image_features, sequence], verbose=0)
        # Get index with high probability
        yhat = np.argmax(yhat)
        # Convert index to word
        word = idx_to_word(yhat, tokenizer)
        # Stop if word not found
        if word is None:
            break
        # Append word as input for generating next word
        in_text += " " + word
        # Stop if we reach end tag
        if word == 'endseq':
            break

    return in_text

# Function to generate caption for an image path
def generate_caption(image_path):
    # Load the image
    image = load_img(image_path, target_size=(224, 224))
    # Convert image pixels to numpy array
    image = img_to_array(image)
    # Reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Preprocess image for VGG
    image = preprocess_input(image)
    # Extract image features
    image_features = feature_extractor.predict(image)
    # Predict the caption
    caption = predict_caption(model, image_features, tokenizer, max_length)
    return caption

# Test with an image
image_path = 'drive/MyDrive/pic.jpg'
caption = generate_caption(image_path)
print("Generated Caption:", caption)
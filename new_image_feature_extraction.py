import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import requests
from PIL import ImageEnhance
from io import BytesIO

# Load the CSV file
data = pd.read_csv("A2_Data_modified.csv")

# Initialize ResNet50 model with pre-trained weights
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to preprocess and extract features from an image
def extract_features(url):
    response = requests.get(url)
    img = image.load_img(BytesIO(response.content), target_size=(300, 300))  # Increase size of the image
    img = ImageEnhance.Contrast(img).enhance(1.5)  # Increase contrast
    img = ImageEnhance.Brightness(img).enhance(1.2)  # Increase brightness/exposure
    
    img_array = image.img_to_array(img)  # Convert PIL image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
    img_array = preprocess_input(img_array)  # Preprocess the image according to ResNet50 requirements
    features = model.predict(img_array)  # Extract features using ResNet50
    
    # Normalizing features using min-max scaling
    features_min = np.min(features)
    features_max = np.max(features)
    normalized_features = (features - features_min) / (features_max - features_min)
    
    return normalized_features, img

# Saving the processed images in a directory
os.makedirs("preprocessed_images", exist_ok=True)

# Applying feature extraction and normalization to all images in the database
features_list = []
for idx, row in data.iterrows():
    features, preprocessed_img = extract_features(row['Image'])
    features_list.append(features)
    
    print(f"Features for image {idx}: {features}")
    
    id_count = data[data['ID'] == row['ID']].shape[0]
    
    # Saving the processed images
    image_name = f"preprocessed_image_{idx}_{id_count}.jpg"
    preprocessed_img.save(os.path.join("preprocessed_images", image_name))

# Converting features_list to numpy array
features_array = np.array(features_list)

# Adding the new column having the extracted features
data['Feature Extracted'] = features_array.tolist()

# Save the DataFrame to a new CSV file
data.to_csv("A2_Data_with_features_norm.csv", index=False)
with open("A2_Data_with_features_norm.pkl", "wb") as f:
    pickle.dump(data, f)
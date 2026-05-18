import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="Indian Bird Classifier (Custom CNN)", page_icon="🐦")
st.title("🐦 Indian Bird Species Detection")
st.write("Classification using a Custom-Built CNN Architecture")

# Define the 25 Species (Ensure this order matches your training labels!)
bird_labels = [
    'Asian Green Bee-Eater', 'Brown-Headed Barbet', 'Cattle Egret', 
    'Common Kingfisher', 'Common Myna', 'Common Rosefinch', 
    'Common Tailorbird', 'Coppersmith Barbet', 'Forest Wagtail', 
    'Gray Wagtail', 'Hoopoe', 'House Crow', 'Indian Grey Hornbill', 
    'Indian Peacock', 'Indian Pitta', 'Indian Roller', 'Jungle Babbler', 
    'Northern Lapwing', 'Red-Wattled Lapwing', 'Ruddy Shelduck', 
    'Rufous Treepie', 'Sarus Crane', 'White Wagtail', 
    'White-Breasted Kingfisher', 'White-Breasted Waterhen'
]

@st.cache_resource
def load_cnn():
    # Standard load for custom models
    return tf.keras.models.load_model('indianbirds_cnnmodel.keras')

model = load_cnn()

uploaded_file = st.file_uploader("Upload a bird photo...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    predictions = model.predict(img_array)
    score = np.max(predictions)
    result = bird_labels[np.argmax(predictions)]
    
    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence: **{score*100:.2f}%**")
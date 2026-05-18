import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Indian Bird Classifier (Transfer Learning)", page_icon="🦅")
st.title("🦅 Indian Bird Species Detection")
st.write("Classification using MobileNetV2 Transfer Learning")

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
def load_tl_model():
    # 1. Rebuild MobileNetV2 Backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights=None
    )
    # 2. Add your custom head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(25, activation='softmax')
    ])
    # 3. Load weights only
    model.load_weights('indianbirds_transfer_model.keras')
    return model

model_tl = load_tl_model()

uploaded_file = st.file_uploader("Upload a bird photo...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # MobileNetV2 Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model_tl.predict(img_array)
    result = bird_labels[np.argmax(predictions)]
    score = np.max(predictions)
    
    st.success(f"Prediction: **{result}**")
    st.metric("Confidence Score", f"{score*100:.2f}%")
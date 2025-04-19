import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from streamlit_option_menu import option_menu
import gdown

# Menu for selecting prediction type
selected = option_menu("Multiple Disease Prediction System",
                       ["❤️ Heart Disease Prediction",
                        "🧠 Brain Disease Prediction"],
                       default_index=0)

@st.cache_resource
def load_model():
    if selected == '❤️ Heart Disease Prediction':
        url = 'https://drive.google.com/uc?export=download&id=1SbUc4mGPiuLo59DH50MVc6Ju18GiOpPF'  # Your Google Drive file link
        output = 'model_roi_net_epoch050.h5'
    elif selected == '🧠 Brain Disease Prediction':
        url = 'https://drive.google.com/uc?export=download&id=1WVkjuSSXYCg8VZppR1s6lPm35tbMvbLI'
        output = 'inceptionv3_binary_model.keras'

    gdown.download(url, output, quiet=False)
    model = tf.keras.models.load_model(output)
    return model

# Preprocessing function with dynamic target size
def preprocess_image(image, target_size):
    # Resize to target size based on the model
    image = cv2.resize(image, target_size)
    # Convert to RGB if needed
    if image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Normalize to [0, 1]
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict(image, model, class_labels, target_size):
    preprocessed_image = preprocess_image(image, target_size)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100
    return class_labels[predicted_class], confidence

# Load the model once
model = load_model()

# Define target sizes and class labels for each model
if selected == '❤️ Heart Disease Prediction':
    st.title("Heart Disease Prediction from MRI Images")
    uploaded_file = st.file_uploader("Upload an MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    class_labels = ['HCM', 'DCM', 'MINF', 'ARV']
    target_size = (256, 256)  # Adjust this if you know the correct size

elif selected == '🧠 Brain Disease Prediction':
    st.title("Brain Tumor Prediction from MRI Images")
    uploaded_file = st.file_uploader("Upload a brain MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary_tumor']  # Match notebook's class_indices
    target_size = (299, 299)  # Matches InceptionV3 input size

# Process uploaded image
if uploaded_file is not None:
    # Read the image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        st.error("Error reading the image. Please upload a valid image file.")
        st.stop()
    
    # Display the uploaded image
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    
    # Predict
    with st.spinner("Predicting..."):
        predicted_class, confidence = predict(image, model, class_labels, target_size)
    
    # Display the result
    st.success(f"**Prediction:** {predicted_class} \n**Confidence:** {confidence:.2f}%")
else:
    st.info("Please upload an MRI image to get a prediction.")

# Footer
st.markdown("---")
st.write("Built with ❤ by Prendu using Streamlit and TensorFlow")
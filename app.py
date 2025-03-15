import streamlit as st
import numpy as np
from PIL import Image

from unet_model import unet_model

input_shape = (112, 112, 3)
model_weights_path = "unet_best.weights.h5"

def load_model_and_weights(input_shape, model_weights_path):
    model = unet_model(input_shape)
    model.load_weights(model_weights_path)
    return model

def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((112,112))
    new_image_array = np.array(image) / 255
    expanded_image_array = np.expand_dims(new_image_array, axis=0)
    return expanded_image_array

def main():
    st.title('Brain Tumor Segmentation in MRI scans')

    uploaded_image = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "tif"])

    if st.button('Predict'):
        if uploaded_image is not None:
            model = load_model_and_weights(input_shape, model_weights_path)
            preprocessed_image = preprocess_image(uploaded_image)
            prediction = model.predict(preprocessed_image)

            col1, col2 = st.columns(2)
            col1.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
            col2.image(prediction, caption='Predicted Image', use_column_width=True)
        else:
            st.warning('Please upload an image.')

if __name__ == '__main__':
    main()
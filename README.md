# Brain Tumor Segmentation 🧠📈

This project provides a web application for segmenting brain tumors in MRI scans using a U-Net deep learning model. It is built with TensorFlow/Keras and Streamlit.

**✨ Live Demo:** [https://brain--tumor--segmentation.streamlit.app/](https://brain--tumor--segmentation.streamlit.app/) ✨

## 🖼️ Features

* **Upload MRI Images:** Supports JPG, JPEG, PNG, and TIF image formats [cite: uploaded:brain-tumor-segmentation-main/app.py].
* **Preprocessing:** Automatically resizes (to 112x112 pixels) and preprocesses uploaded images to fit the model's input requirements [cite: uploaded:brain-tumor-segmentation-main/app.py].
* **Tumor Segmentation:** Utilizes a pre-trained U-Net model to predict the segmentation mask for potential tumors [cite: uploaded:brain-tumor-segmentation-main/app.py].
* **Visualization:** Displays the original uploaded MRI scan alongside the predicted tumor segmentation mask for easy comparison [cite: uploaded:brain-tumor-segmentation-main/app.py].

## ⚙️ Model Architecture

The segmentation is performed using a U-Net model, a type of convolutional neural network designed for biomedical image segmentation [cite: uploaded:brain-tumor-segmentation-main/unet_model.py]. Key aspects include:
* **Encoder Path:** Captures context through convolutional and max-pooling layers [cite: uploaded:brain-tumor-segmentation-main/unet_model.py].
* **Decoder Path:** Enables precise localization using up-sampling and concatenation with high-resolution features from the encoder path [cite: uploaded:brain-tumor-segmentation-main/unet_model.py].
* **Implementation:** Built using TensorFlow's Keras API [cite: uploaded:brain-tumor-segmentation-main/unet_model.py, uploaded:brain-tumor-segmentation-main/requirements.txt].

## 📦 Dependencies

* Python
* NumPy
* TensorFlow
* Streamlit
* Pillow (PIL)

## 🛠️ Installation & Setup (for running locally)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd brain-tumor-segmentation-main
    ```
2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install numpy tensorflow streamlit
    ```

3.  **Model Weights:**
    Ensure the pre-trained model weights file (`unet_best.weights.h5`) is present in the root directory of the project.

## ▶️ Usage (Local)

1.  Navigate to the project directory in your terminal.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  Open the local URL provided by Streamlit in your web browser.
4.  Click "Browse files" to upload an MRI image
5.  Click the "Predict" button to see the segmentation result

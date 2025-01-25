import streamlit as st
import os, shutil, pathlib
from tensorflow import keras # API de tensorflow: rutas de acceso al código de tensorflow
from tensorflow.keras import layers # construir las capas de la red convolucional
# lee las imágenes, decodifica las imágenes,
# convierte en tensores, cambia el tamaño de las imágenes, las empaca en lotes
from tensorflow.keras.utils import image_dataset_from_directory # similar a la de NLP
import numpy as np # para cálculo de operaciones matemáticas
import tensorflow as tf # tensorflow
import matplotlib.pyplot as plt # graficas
from keras.applications.vgg16 import VGG16 # modelo preentrenado
from PIL import Image # para trabajar con imágenes

#Set upthe page
st.set_page_config(page_title="Money Classifier", page_icon="💰", layout="centered", initial_sidebar_state="expanded")
st.title("৳ Money Classifier 🤑")
st.header("This is a simple image classifier for Bangladesh money bills")

#Upload images
st.subheader("Upload an image of a Bangladeshi Taka money bill")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

#Save image
if uploaded_file is not None:
    # Save image
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image("temp.jpg", caption="Uploaded Image", use_container_width=True)

    # Load model
    model = keras.models.load_model("Money_Clasifier\cnn_modelo_denominaciones.keras")
    img = keras.preprocessing.image.load_img("temp.jpg", target_size=(250, 120))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Get class names
    class_names = ['5', '500', '50', '1', '100', '1000', '10', '20', '2']

    st.success("#### This image most likely belongs to a Bill of ৳ {} Banglaseshi Taka with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This is a simple image classifier for Bangladesh money bills. It is based on a Convolutional Neural Network (CNN) and was trained using the VGG16 model.")

st.sidebar.title("How to use")
st.sidebar.info("Upload an image of a Bangladesh money bill. The classifier will then predict the denomination of the bill and the confidence level of the prediction.")





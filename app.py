import requests
import tensorflow
import keras
import streamlit as st
from PIL import Image
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from data import *


def extract_features(filename):
    """ extract features from each photo in the directory
    """
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


def word_for_id(integer, tokenizer):
    """ map an integer to a word
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    """ generate a description for an image
    """
    result = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([result])[0]

        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
            # append as input for generating the next word
        result += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return result.replace('startseq', '').replace('endseq', '')


@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    # load the tokenizer
    tokenizer = load(open('model_trained/tokenizer.pkl', 'rb'))
    # load the model
    model = load_model('model_trained/model_1.h5')
    return model, tokenizer


def main():

    st.set_page_config(layout="wide")

    st.header("Group9 - Image Caption Generator")
    top_left_column, top_right_column = st.columns(2)
    top_left_column.write("Please select model and image from the left options")
    # st.write("select model and image from the left options")
    pred_button = top_right_column.button("Generate Caption")

    st.sidebar.title('Options')
    model_type = st.sidebar.radio(
        'Select Model->', options=['Model 1 - LSTM', 'Model 2- Attention'])
    #st.sidebar.header('Test Images ->')
    #image_selected = st.sidebar.selectbox("choose image:", example_image_files)

    image_selected_name = st.sidebar.radio(
        'Select Image ->', options=['image_1.jpg', 'image_2.jpg', 'image_3.jpg'])

    model, tokenizer = load_model_and_tokenizer()

    max_length = 34

    # Create a UI component to read image
    #image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if model_type == 'Model 1 - LSTM':
        # divide your interface to two parts
        left_column, right_column = st.columns(2)
        image_file = 'test_images/' + image_selected_name
        # left_column.title("Selected Image: ")
        left_column.image(image_file,  # caption="Selected image",
                          use_column_width=True)
        input_image = Image.open(image_file)

        if pred_button:
            photo = extract_features(image_file)

            caption = generate_desc(model, tokenizer, photo, max_length)
            right_column.subheader("Generated Caption: ")
            right_column.title(caption)
    else:
        st.write("model coming soon ...")


if __name__ == '__main__':
    main()

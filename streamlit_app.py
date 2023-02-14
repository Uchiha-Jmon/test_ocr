import streamlit as st 
import pytesseract
from PIL import ImageOps,Image
import pandas as pd
from fuzzywuzzy import fuzz
import tempfile
import os
import io
from google.cloud import vision
from fuzzywuzzy import fuzz
import re
import urllib.request
import requests

def detect_text(image_in):
    """Detects text in the file."""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'APi_KEY.json'
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_text_detection]
    with io.open(image_in, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    text_for_file=[]
    text_for_file.append(texts[0].description)
    new_text = ""
    for line in text_for_file:
        # line = line.replace("\n", "")
        new_text=new_text+line

        # print(new_text)


    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return new_text

def ocr_core(filename,config):
    """
    This function will handle the core OCR processing of images.
    """
    # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    text = pytesseract.image_to_string(filename,config=config)
    # data = pytesseract.image_to_data(filename,config=config,output_type='data.frame')
    # data = data[data.conf != -1]
    return text #, data


image_file=st.file_uploader(label='Upload Image File', type=['png','jpg','jpeg'])
st.image(image_file)
col1,col2,col3=st.columns(3, gap="large")

if image_file:
    file_path = tempfile.NamedTemporaryFile().name
    with open(file_path, "wb") as f:
        f.write(image_file.getvalue())
    # st.write("File path:", file_path)
    file_name=image_file.name.replace(".jpg","")
    # st.write(file_name)
    file=open(file_name+".txt","r")
    ground_truth=file.read()

with col1:
    # Free Version
    st.markdown('<div style="text-align: center;"><h3>Free Version</h3></div>', unsafe_allow_html=True)
    if image_file is not None:
        selected_file=Image.open(image_file).convert('L')
        config=('-l test --oem 1 --psm 6')
        text=ocr_core(selected_file,config)
        text_to_save = st.text_area('', text, height = 500, max_chars=None)
        hypothesis=text
        match=fuzz.ratio(re.sub(' +',' ', hypothesis.strip()), re.sub(' +',' ',ground_truth.strip()))
        st.write("**Accuracy**",match,"**%**")

with col2:
    # Paid Google Version
    st.markdown('<div style="text-align: center;"><h3>Google Version</h3></div>', unsafe_allow_html=True)
    text=detect_text(file_path)
    
    
    text_to_save = st.text_area('', text, height = 500, max_chars=None)
        # [END vision_python_migration_text_detection]
        # [END vision_text_detection]
    
    hypothesis=text
    match=fuzz.ratio(re.sub(' +',' ', hypothesis.strip()), re.sub(' +',' ',ground_truth.strip()))
    st.write("**Accuracy**",match,"**%**")

with col3:
    # In House Model
    st.markdown('<div style="text-align: center;"><h3>Local Model</h3></div>', unsafe_allow_html=True)
    if image_file is not None:
        selected_file=Image.open(image_file).convert('L')
        config=('-l 4M_DER_69 --oem 1 --psm 6')
        text=ocr_core(selected_file,config)
        text_to_save = st.text_area('', text, height = 500, max_chars=None)
    hypothesis=text
    match=fuzz.ratio(re.sub(' +',' ', hypothesis.strip()), re.sub(' +',' ',ground_truth.strip()))
    st.write("**Accuracy**",match,"**%**")

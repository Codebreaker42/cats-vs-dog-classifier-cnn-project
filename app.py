import streamlit as st
import pickle as pkl
import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
# unpickle the model
model=pkl.load(open('model.pkl','rb'))

st.title('Cat VS Dog Identifier')

# file uploader
uploaded_file=st.file_uploader('Upload Your File Here')

# saving image in uploads folder
def save_image(uploaded_file):
    image_location_path=os.path.join('uploads',uploaded_file.name)
    with open(image_location_path,'wb') as f:
        f.write(uploaded_file.read()) 

def image_preprocess(img):
    # Resize the image to the required input size of the model
    img = image.resize((256, 256))
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Expand the dimensions to create a batch with a single sample
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    # saving image 
    save_image(uploaded_file)
    st.write('File Uploaded Succesfully')
    #displaying image
    image=Image.open(uploaded_file)
    st.image(image)
    # prediction
    preprocessed_image=image_preprocess(image)
    if model.predict(preprocessed_image) == 0:
        st.title('            Cat')
    else:
        st.title('            Dog')
    

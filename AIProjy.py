from roboflow import Roboflow
import streamlit as st
import os
from PIL import Image

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

st.title("Spider Venomosity Checker")
st.markdown("checks spider venomosity")


image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    img = load_image(image_file)
    st.image(img)
    with open(os.path.join(image_file.name),"wb") as f: 
      f.write(image_file.getbuffer())         
    st.success("Saved File")

    rf = Roboflow(api_key="uc3t6tXboFLZ91mVoqwu")
    project = rf.workspace().project("arach-net")
    model = project.version(1).model

# infer on a local image
    st.markdown(model.predict(image_file.name, confidence=40, overlap=30).json())







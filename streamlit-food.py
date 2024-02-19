import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from fastai.learner import load_learner
import pathlib
from fastai.vision.all import *

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

st.markdown("""# Mongolian Food Classifier

This app can be used to identify four common mongolian foods: buuz/dumplings, khuushuur, tsuivan and niislel/olivier salad.""")

st.markdown("""### Try it out!""")

learn = load_learner("export.pkl")

image_file = st.file_uploader("Upload your image here",type=["png","jpg","jpeg"])

if image_file is not None:
    img = Image.open(image_file)
    if img is not None:
        img = PILImage.create(image_file)
        pred, pred_idx, probs = learn.predict(img)

        st.write(f"Prediction: {pred}; Probability: {probs[pred_idx].item():.4f}")
    else:
        st.write("Failed to open the image.")

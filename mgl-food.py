import streamlit as st
import pickle

st.markdown("""# Mongolian Food Classifier

This app can be used to identify four commonly found mongolian food: buuz/dumplings, khuushuur, tsuivan and niislel/olivier salad.""")

st.markdown("""### Upload your image here""")

image_file = st.file_uploader("Image Uploader", type=["png","jpg","jpeg"])

## Model Loading Section
model = pickle.load(open('export.pkl'))

# if not model_path.exists():
#     with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
#         url = 'https://drive.google.com/uc?id=1hks-274Jgo6cWFhijBwTR2WjSczUxNhR'
#         output = 'export.pkl'
#         gdown.download(url, output, quiet=False)
#     learn_inf = load_learner('export.pkl')
# else:
#     learn_inf = load_learner('export.pkl')

col1, col2 = st.columns(2)
if image_file is not None:
    img = PILImage.create(image_file)
    pred, pred_idx, probs = learn_inf.predict(img)

    with col1:
        st.markdown(f"""### Predicted food: {pred.capitalize()}""")
        st.markdown(f"""### Probability: {round(max(probs.tolist()), 3) * 100}%""")
    with col2:
        st.image(img, width=300)
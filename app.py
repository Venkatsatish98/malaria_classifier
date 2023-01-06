import streamlit as st
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH ='model_malaria.h5'

model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
   
    x = x / 255 ##Converting the image intoarray and scaling
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)

    if preds == 0:
        preds = "The Person is Infected With Malaria"
    else:
        preds = "The Person is not Infected With Malaria"

    return preds




def main():
    html_temp = """
        <div style="background-color:aqua;padding:10px;">
        <h2 style="color:black;text-align:center;">Malaria Classifier</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write('This app will identify whether the person is affected with malaria or not.')
    st.write('Please upload a "PNG" file and click on "View Result" button.')


    st.subheader("Please select a 'png' file")
    filename = st.file_uploader("Upload", type="png")
    if filename is not None:
        try:
            if st.button('Predict'):
                preds = model_predict(filename, model)
                result = preds
                return st.write(result)

        except Exception as e:
            return st.write(e)


if __name__ == '__main__':
    main()

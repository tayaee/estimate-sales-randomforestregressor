import os

import pandas as pd
import streamlit as st
from joblib import load

from ut_model import ModelInfo, load_model_info_from_json

MODEL_VERSION = "1.0.0"
METADATA_PATH = f"models/{MODEL_VERSION}.json"
MODEL_PATH = f"models/{MODEL_VERSION}.joblib"
FEATURE_NAMES: list[str] = []
TARGET_NAME: str = "Volume"
model_info: ModelInfo | None = None
try:
    if os.path.exists(METADATA_PATH):
        model_info = load_model_info_from_json(METADATA_PATH)
        FEATURE_NAMES = model_info.data_schema.features
        TARGET_NAME = model_info.data_schema.target
    else:
        st.error(f"Metadata file not found: {METADATA_PATH}. Cannot initialize application.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model metadata: {e}. Cannot initialize application.")
    st.stop()


@st.cache_resource
def load_ml_model():
    if os.path.exists(MODEL_PATH):
        try:
            return load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    st.error(f"Model file not found: {MODEL_PATH}. Cannot perform prediction.")
    st.stop()


model = load_ml_model()
st.set_page_config(layout="wide", page_title=f"Estimate {TARGET_NAME} using ML Model")
st.title(f"Estimate {TARGET_NAME} using Machine Learning Model")
st.markdown(f"Adjust the input features below to predict the **{TARGET_NAME}** volume.")
input_data = {}
st.subheader("Input Features")
for feature in FEATURE_NAMES:
    input_data[feature] = st.slider(
        f"{feature}",
        min_value=0.0,
        max_value=1000.0,
        value=500.0,
        step=0.1,
        format="%.2f",
    )
if st.button("Predict"):
    features_df = pd.DataFrame([input_data])
    try:
        prediction = model.predict(features_df)[0]
        st.subheader("Prediction Result")
        st.metric(label=f"Predicted {TARGET_NAME}", value=f"{prediction:.2f} units")
    except Exception as e:
        st.error(f"Prediction Error: {e}")

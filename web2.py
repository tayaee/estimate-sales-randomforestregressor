import os

import gradio as gr
import pandas as pd
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
        print(f"Loaded metadata. Features: {FEATURE_NAMES}, Target: {TARGET_NAME}")
    else:
        print(f"Model metadata file not found at: {METADATA_PATH}. Gradio will not start with proper features.")
except Exception as e:
    print(f"Error loading model metadata: {e}. Gradio will use empty feature list.")


model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load(MODEL_PATH)
        print(f"Model successfully loaded from: {MODEL_PATH}")
    else:
        print(f"Model file not found: {MODEL_PATH}. Prediction function will fail.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


def predict_sales(*args):
    if not FEATURE_NAMES:
        return "ERROR: Feature names were not loaded from metadata. Cannot perform prediction."

    if model is None:
        return "ERROR: Model not loaded. Please ensure the model file exists."

    if len(args) != len(FEATURE_NAMES):
        return f"Prediction Error: Expected {len(FEATURE_NAMES)} inputs, but received {len(args)}."

    input_values = dict(zip(FEATURE_NAMES, args))

    features_df = pd.DataFrame([input_values])
    try:
        prediction = model.predict(features_df)[0]
        return f"""
        ### Prediction Result
        ---
        **Predicted {TARGET_NAME}**: **{prediction:.2f}** units
        """
    except Exception as e:
        return f"Prediction Error: {e}"


if not FEATURE_NAMES:
    input_components = [
        gr.Markdown(f"## Model Initialization Failed\nMetadata file not found or invalid: `{METADATA_PATH}`")
    ]
    description = "Model metadata could not be loaded. Check console for errors."
    model_name = "N/A"
else:
    input_components = [gr.Slider(minimum=0, maximum=1000, value=500, label=f"{name}") for name in FEATURE_NAMES]

    model_name = "Not Loaded"
    if model and "rf" in model.named_steps:
        model_name = model.named_steps["rf"].__class__.__name__

    description = f"Adjust the input features to predict the **{TARGET_NAME}** volume."
    description += f"\n\nThis interface uses a `{model_name}` model."

iface = gr.Interface(
    fn=predict_sales,
    inputs=input_components,
    outputs=gr.Markdown(label="Prediction Result"),
    title=f"Estimate {TARGET_NAME} using Machine Learning Model",
    description=description,
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch(share=True)

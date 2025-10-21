import os
import gradio as gr
import pandas as pd
from joblib import load
from ut_model import ModelInfo, load_model_info_from_json

MODEL_VERSION = "sales-1.0.0"
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
        print(f"Meta loaded. F:{len(FEATURE_NAMES)}, T:{TARGET_NAME}")
    else:
        print(f"Meta not found: {METADATA_PATH}. Gradio starting without features.")
except Exception as e:
    print(f"Meta load err: {e}. Empty features used.")


model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print(f"Model not found: {MODEL_PATH}. Prediction will fail.")
except Exception as e:
    model = None
    print(f"Model load error: {e}")


def predict_sales(*args):
    if not FEATURE_NAMES:
        return "ERROR: Features missing. Cannot predict."

    if model is None:
        return "ERROR: Model not loaded."

    if len(args) != len(FEATURE_NAMES):
        return f"Error: Input count mismatch. Expected {len(FEATURE_NAMES)}, got {len(args)}."

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
        return f"Predict Error: {e}"


if not FEATURE_NAMES:
    input_components = [gr.Markdown(f"## Init Fail\nMeta error: `{METADATA_PATH}`")]
    description = "Metadata load error. Check console."
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

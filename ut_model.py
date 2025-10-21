import json
import os
from datetime import datetime
from typing import Any, Literal

import joblib
import sklearn
from joblib import dump
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

joblib_version = joblib.__version__


class ModelDetails(BaseModel):
    type: Literal["ml", "dl", "genai"]
    algorithm: str
    library: str
    library_version: str
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
    joblib_version: str


class DataSchema(BaseModel):
    features: list[str]
    target: str


class ModelFiles(BaseModel):
    model_path: str
    metadata_path: str


class ModelInfo(BaseModel):
    model_details: ModelDetails
    data_schema: DataSchema
    equation: str
    hyperparameters: dict[str, Any]
    model_files: ModelFiles


def save_model_and_metadata(model, params, output_path: str, feature_names: list, target_name: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dump(model, output_path)
    print(f"Model saved: {output_path}")

    json_path = os.path.splitext(output_path)[0] + ".json"
    regressor_step = model.named_steps["rf"]

    if isinstance(regressor_step, LinearRegression):
        intercept = regressor_step.intercept_
        coefficients = regressor_step.coef_

        equation_parts = [f"{intercept:.4f}"]
        for feature, coef in zip(feature_names, coefficients):
            sign = "+" if coef >= 0 else "-"
            equation_parts.append(f" {sign} {abs(coef):.4f} * `{feature}`")
        equation = f"{target_name} = {''.join(equation_parts)}"
    elif isinstance(regressor_step, RandomForestRegressor):
        equation = f"{target_name} = RandomForestRegressor(votes from multiple trees)"
    else:
        equation = f"{target_name} = f({', '.join(feature_names)})"

    model_info = ModelInfo(
        model_details=ModelDetails(
            type="ml",
            algorithm=regressor_step.__class__.__name__,
            library="scikit-learn",
            library_version=sklearn.__version__,
            joblib_version=joblib_version,
        ),
        data_schema=DataSchema(features=feature_names, target=target_name),
        equation=equation,
        hyperparameters=params,
        model_files=ModelFiles(model_path=os.path.basename(output_path), metadata_path=os.path.basename(json_path)),
    )

    with open(json_path, "w") as f:
        f.write(model_info.model_dump_json(indent=4))
    print(f"Meta saved: {json_path}")


def load_model_info_from_json(json_path: str) -> ModelInfo:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Meta not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ModelInfo.model_validate(data)

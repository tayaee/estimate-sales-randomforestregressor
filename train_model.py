import os

import click
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ut_model import save_model_and_metadata

FEATURE_NAMES = [
    "Advertising Expenditure",
    "Campaign Engagement Score",
    "Discount Percentage",
    "Product Price",
]
TARGET_NAME = "Sales"


def load_and_prepare_data(input_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    df = pd.read_csv(input_path)
    required_cols = FEATURE_NAMES + [TARGET_NAME]
    if not all(col in df.columns for col in required_cols):
        missing = set(required_cols) - set(df.columns)
        raise ValueError(f"Required columns missing in data: {missing}")
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data loaded. Train samples: {len(X_train)}")

    return X_train, X_test, y_train, y_test


def build_and_train_model(X_train, y_train):
    pipeline = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(random_state=42))])
    param_grid = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [None, 5, 10],
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best CV Score (MSE): {-grid_search.best_score_:.2f}")
    return best_model, best_params


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="CSV data path.")
@click.option("--output-model", "output_path", required=True, type=click.Path(), help="Output model path (joblib)")
def main(input_path, output_path):
    try:
        X_train, _, y_train, _ = load_and_prepare_data(input_path)
        best_model, best_params = build_and_train_model(X_train, y_train)
        save_model_and_metadata(best_model, best_params, output_path, FEATURE_NAMES, TARGET_NAME)
    except Exception as e:
        print(f"Training failed: {e}")


if __name__ == "__main__":
    main()

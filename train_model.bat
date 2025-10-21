@echo off
if .%VIRTUAL_ENV%. == .. (
    if not exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate
    )
)
python train_model.py --input=data/sales.csv --output-model=models/sales-1.0.0.joblib

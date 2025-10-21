@echo off
if .%VIRTUAL_ENV%. == .. (
    if not exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate
    )
)
streamlit run web1.py

# Sales Prediction ML Project

This project demonstrates a complete workflow for training a Machine Learning (ML) model using scikit-learn and serving that model using two modern Python web frameworks: Streamlit and Gradio.

## Model Objective

The primary goal of the model is to predict the **Sales** volume based on key independent variables related to marketing efforts and product characteristics.

**Features (Independent Variables) Used for Prediction:**
* `Advertising Expenditure`
* `Campaign Engagement Score`
* `Discount Percentage`
* `Product Price`

**Response Variable (Target):**
* `Sales` (Predicted Sales Volume)

---

## Quickstart

Follow these steps to set up the environment, train the model, and run the web applications.

### 1. One time per machine

1. Install Python 3.11+

2. Install uv

### 2. Clone repo
    ```
    git clone <URL>
    ```
### 3. One time per clone

1. Create .venv, install dependencies and activate .venv
    ```
    make-venv-uv.bat
    ```
2. (Optional) Train a model (optional, a pre-trained model is available at models/1.0.0.joblib)
    ```
    train_model.bat
    ```
### 4. Test

1. Run the demo app (streamlit version). Use browser to test with http://localhost:7860
    ```
    web1.bat
    ```
2. OR, run another demo app (gradio version). Use browser to test with http://localhost:8501

    ```
    web2.bat
    ```

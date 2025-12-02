# Aluminum Cold Roll Durability Prediction

This project aims to predict the durability of aluminum cold roll using a CatBoost machine learning model.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KingSam1124/Stats-101C-Final-Project-Group-1/tree/main
    cd Stats-101C-Final-Project-Group-1
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    ./venv/Scripts/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This project contains two main CatBoost models:

1.  **`catboost_model.py`**: A baseline CatBoost model. This model achieved a private score of **0.421349** on Kaggle.
    To train this model and generate predictions, run:
    ```bash
    python scripts/catboost_model.py
    ```

2.  **`catboost_model_eng.py`**: A CatBoost model with additional feature engineering. This model achieved a private score of **0.421372** on Kaggle.
    To train this model and generate predictions, run:
    ```bash
    python scripts/catboost_model_eng.py
    ```

## Project Structure

-   `scripts/`: Contains the final Python scripts for the models used in the project.
-   `unused_model_scripts/`: Contains scripts for models that were explored but not selected as final models.
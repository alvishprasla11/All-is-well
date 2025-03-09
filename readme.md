# Depression Prediction Model

This repository contains a machine learning model to predict the likelihood of depression based on various factors such as age, marital status, education level, smoking status, physical activity, employment status, income, alcohol consumption, dietary habits, sleep patterns, history of mental illness, substance abuse history, family history of depression, and chronic medical conditions.

## 📥 Cloning the Repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/alvishprasla11/All-is-well.git
cd All-is-wel
```

## 🚀 Getting Started with Django

### 1. Install Required Dependencies

Ensure you have Python installed (preferably version 3.8 or later). Then, install the required dependencies:

```bash
pip install -r requirements.txt
```
To use a virtual environment (recommended):
```bash
# Create a virtual environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### 2. Running Django

Start the Django server:

```bash
cd stress_pridiction
python manage.py runserver
```

Your application should now be running at `http://127.0.0.1:8000/`.

## 📦 Dependencies

All required dependencies are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
```

## 📂 Directory Structure

- **requirements.txt** → Contains all required dependencies.
- **Data/** → Stores original and transformed datasets.
  - `stress_detection_data.csv` → Raw dataset.
  - `transformed_data.csv` → Cleaned and processed dataset.
- **Model/** → Contains model training files.
  - `Model_Training/stress.py` → Training script for the depression prediction model.
  - `stress_prediction_model.pkl` → Saved trained model.
- **PowerBI/** → Contains Power BI visualizations and reports.
  - `stress_analysis.pbix` → Power BI report file.
- **stress_pridiction/** → Contains The Dajngo website and api for the model
  - `manage.py` → file to start the server


## 🧠 What the Model Does

This machine learning model predicts the likelihood of depression based on input features. It assigns a probability score indicating how likely a person is to be depressed based on their lifestyle, medical history, and socioeconomic factors.

## 🏗️ How to Train and Test the Model

1. Navigate to the `Model/Model_Training/` directory.

```bash
cd Model/Model_Training/
```

2. Run the training script:

```bash
python stress.py
```
or
```bash
python Stress.ipynb
```

3. Once trained, the model will be saved as `stress_prediction_model.pkl` in the `Model/` directory.

## 📊 Power BI Analysis

The Power BI report can be found in the `PowerBI/` directory:

```bash
PowerBI/stress_analysis.pbix
```

Open this file using Power BI Desktop to explore insights and trends in the dataset.

---

This repository provides a complete pipeline for predicting depression and analyzing results using machine learning and Power BI. 🚀


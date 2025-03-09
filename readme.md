# Depression Prediction Model

## Overview
This project analyzes depression likelihood based on various factors such as age, marital status, education level, smoking status, physical activity, and more. It uses machine learning to assign a likelihood score indicating how likely an individual is to be depressed.

---

## **Getting Started**

### **1. Clone the Repository**
```bash
git clone https://github.com/alvishprasla11/All-is-well.git
cd All-is-well
```

### **2. Install Required Dependencies**
#### **For Windows/macOS/Linux**
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

This will install all necessary libraries.

---

## **Project Structure**
```
├── data/
│   ├── original_data.csv    # The raw dataset
│   ├── transformed_data.csv # Preprocessed dataset used for modeling
│
├── model/
│   ├── trained_model.pkl    # The saved machine learning model
│
├── powerbi/
│   ├── depression_analysis.pbix # Power BI file for data visualization
│
├── requirements.txt         # List of required libraries
│
├── app/
│   ├── manage.py            # Django entry point
│   ├── myapp/               # Main Django application
│
├── README.md                # Project documentation (this file)
```

---

## **Running the Django Application**
To start the Django web application:

### **1. Navigate to the Django Project Folder**
```bash
cd app
```

### **2. Apply Migrations**
```bash
python manage.py migrate
```

### **3. Run the Server**
```bash
python manage.py runserver
```
Now, the app should be accessible at `http://127.0.0.1:8000/`.

---

## **Model Information**
### **What the Model Does**
- Predicts depression likelihood based on provided input features.
- Uses machine learning algorithms to assign a probability score.
- The higher the score, the more likely an individual is to be depressed.

### **Where the Model is Stored**
The trained machine learning model is located at:
```
model/trained_model.pkl
```
It can be loaded using Python's `joblib` library:
```python
import joblib
model = joblib.load("model/trained_model.pkl")
prediction = model.predict([[25, 1, 2, 6.5, 7, 2, 3, 1, 0, 8, 30, 5, 1, 120, 200, 90]])
print("Predicted Depression Likelihood:", prediction)
```

---

## **Libraries & Dependencies**
All required libraries are listed in `requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
django
joblib
```
To install them, run:
```bash
pip install -r requirements.txt
```

---

## **Data Files**
- **Original Data**: `data/original_data.csv`
- **Transformed Data** (Preprocessed): `data/transformed_data.csv`
- **Power BI File**: `powerbi/depression_analysis.pbix`

---

## **Next Steps**
- Improve accuracy by fine-tuning hyperparameters.
- Implement a better frontend for user interaction.
- Add more data to increase model robustness.

---

## **Contributors**
- Your Name
- Team Members (if applicable)

For questions, reach out via email or open an issue in the repository.

Happy coding! 🚀


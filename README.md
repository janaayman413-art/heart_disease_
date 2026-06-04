# Heart Disease Prediction 

## Project Overview

This project aims to predict the risk of heart disease using Machine Learning techniques.
The workflow includes:

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Training and Evaluation
* Hyperparameter Tuning
* Model Deployment using Gradio

The project was developed as part of a Data Science course project.

---

# Objectives

* Perform data cleaning and preprocessing
* Analyze the dataset using visualizations
* Create new engineered features
* Train and compare multiple machine learning models
* Evaluate model performance using classification metrics
* Deploy the final model as a web application

---

# Machine Learning Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

---

# Dataset Information

Dataset Name: Heart Disease Prediction Dataset

Original Source: Kaggle

Dataset Link:
[Add your dataset link here]



---

# Project Structure

```bash
├── Heart_Disease_Project_jaa5_f.ipynb
├── app.py
├── heart_attack_model.pkl
├── requirements.txt
├── README.md
```

* `Heart_Disease_Project_jaa5_f.ipynb` → Main notebook containing the full ML workflow
* `app.py` → Gradio deployment application
* `heart_attack_model.pkl` → Saved trained model
* `requirements.txt` → Required Python libraries

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

# Required Libraries

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* gradio
* joblib

---

# Exploratory Data Analysis (EDA)

The project includes several visualizations such as:

* Heart Disease Distribution
* BMI Distribution
* Age Category Analysis
* Correlation Heatmap
* Sleep Hours vs Heart Disease
* Gender-based Heart Disease Analysis
* High Risk Lifestyle Analysis

---

# Feature Engineering

A new feature called `High_Risk_Lifestyle` was created to identify unhealthy lifestyle patterns associated with higher heart disease risk.

---

# Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

Hyperparameter tuning was performed using GridSearchCV to improve performance.

---

# Gradio Deployment

The final model was deployed using Gradio for real-time predictions.

APP URL:
  https://2bfa3f16eab5a09d95.gradio.live

Application Preview
<img width="1365" height="605" alt="image" src="https://github.com/user-attachments/assets/090d428c-3497-4a63-9f44-d03f107e7487" />


# Demo Video

Project Demo Video:
[Add your video link here]



# Results

The Random Forest model achieved the best overall performance among the tested models.

Example outputs and visualizations are included in the notebook.

---

# Team Members

* Jana Ayman

---

# How to Use

1. Open the Gradio application
2. Enter patient health information:

   * BMI
   * Physical Health Days
   * Mental Health Days
   * Sleep Hours
3. Click Submit
4. The model predicts the heart disease risk instantly

---

# Additional Notes

* The project was developed using Python and Google Colab.
* The notebook contains the complete workflow from preprocessing to deployment.
* The application predicts heart disease risk based on patient health indicators.




 

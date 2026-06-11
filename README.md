Overview

Heart Disease Prediction is a machine learning project developed to predict the likelihood of heart disease based on patient medical information. The system analyzes clinical attributes such as age, blood pressure, cholesterol level, heart rate, and other health indicators to classify whether a patient is at risk of developing heart disease.
The project applies multiple supervised machine learning algorithms and evaluates their performance to identify the most effective predictive model. The solution demonstrates how machine learning can assist healthcare professionals in supporting early diagnosis and informed clinical decision making.

Objectives

The primary objective of this project is to develop an intelligent disease prediction system capable of accurately identifying the presence of heart disease from structured medical data. The project focuses on data preprocessing, model training, performance evaluation, and comparative analysis of multiple classification algorithms.

Dataset
This project uses the Heart Disease Dataset containing patient health information such as age, sex, chest pain type, blood pressure, cholesterol level, maximum heart rate, and other clinical attributes. The target variable indicates the presence or absence of heart disease.
Dataset Source:
Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/45/heart+disease
The dataset is not included in this repository due to size limitations. Download the dataset from the links above and place heart.csv inside:
data/heart_kaggle/

Technologies Used

Python
Pandas
NumPy
Scikit Learn
XGBoost
Matplotlib
Joblib

Machine Learning Algorithms

Logistic Regression
A statistical classification algorithm used as a baseline model for disease prediction.

Support Vector Machine
A powerful classification technique that identifies optimal decision boundaries between classes.

Random Forest
An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

XGBoost
A gradient boosting algorithm known for its efficiency, scalability, and strong predictive performance on structured datasets.

Project Workflow

Data Collection
Data Preprocessing
Feature Scaling
Training and Testing Split
Model Training
Performance Evaluation
Accuracy Comparison
Confusion Matrix Visualization
Model Serialization

Performance Evaluation
The models are evaluated using standard classification metrics including:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Comparative analysis is performed to determine the best performing model for heart disease prediction.

Results

The trained models achieved high classification performance on the dataset. Random Forest and XGBoost produced the highest prediction accuracy, demonstrating their effectiveness for medical diagnosis tasks involving structured healthcare data.

Conclusion

This project successfully demonstrates the application of machine learning techniques for heart disease prediction. By leveraging clinical patient information and advanced classification algorithms, the system provides accurate predictions that can support healthcare professionals in early disease detection and risk assessment. The project highlights the potential of machine learning in improving healthcare analytics and decision support systems.

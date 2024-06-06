# Heart Disease Prediction

This repository contains a dataset for heart disease prediction and a Jupyter notebook that demonstrates how to build a predictive model using machine learning techniques.

## Files in the Repository

- `data.csv`: This file contains the dataset used for training and evaluating the heart disease prediction model. The dataset includes various features related to patients' health and medical history.
- `Heart_Disease_prediction.ipynb`: This Jupyter notebook contains the code for loading the dataset, preprocessing the data, building and evaluating a machine learning model to predict heart disease.

## Dataset (`data.csv`)

The dataset contains the following columns:

- `age`: Age of the patient
- `sex`: Sex of the patient (1 = male; 0 = female)
- `cp`: Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
- `trestbps`: Resting blood pressure (in mm Hg)
- `chol`: Serum cholesterol in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- `restecg`: Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (1 = yes; 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
- `ca`: Number of major vessels (0-3) colored by fluoroscopy
- `thal`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
- `target`: Presence of heart disease (1 = yes; 0 = no)

## Jupyter Notebook (`Heart_Disease_prediction.ipynb`)

### Overview

The notebook includes the following steps:

1. **Loading the Dataset**: Importing the data from `data.csv`.
2. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
3. **Exploratory Data Analysis (EDA)**: Visualizing the dataset to understand the distribution of features and their relationship with the target variable.
4. **Model Building**: Training various machine learning models (e.g., Logistic Regression, Decision Trees, Random Forest, etc.) to predict the presence of heart disease.
5. **Model Evaluation**: Evaluating the performance of the models using metrics such as accuracy, precision, recall, and F1-score.
6. **Hyperparameter Tuning**: Optimizing the model parameters to improve performance.
7. **Conclusion**: Summarizing the findings and suggesting potential improvements.

## Getting Started

### Prerequisites

To run the notebook, you need to have the following installed:

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

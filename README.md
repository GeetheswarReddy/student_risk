
# Student Mental Health Risk Detection API

This project builds a machine learning system that predicts the potential mental health risk of students using lifestyle and academic features such as sleep, stress, study hours, and physical activity.

The trained model is exposed through a REST API using FastAPI, containerized with Docker, and deployed on AWS EC2. The project demonstrates how a machine learning model can move from experimentation to a deployable production style service.

---

# Project Motivation

Students often face high stress levels, lack of sleep, and unhealthy routines that can negatively affect mental health.

This project explores how behavioral and academic data can be used to build a predictive system that identifies students who may be at higher risk.

The focus of the project is not only model performance but also system design, reproducibility, and deployment.

---

# Features

• Predicts potential student mental health risk using machine learning
• Implements Logistic Regression as a baseline model
• Uses XGBoost as the primary model for improved performance
• Handles class imbalance using scale_pos_weight
• Performs feature engineering and preprocessing
• Exposes predictions through a FastAPI REST API
• Provides interactive API documentation using Swagger
• Containerized with Docker for reproducible environments
• Deployed on AWS EC2

---

# Dataset Features

The model uses the following student attributes.

| Feature            | Description                          |
| ------------------ | ------------------------------------ |
| Age                | Age of the student                   |
| Gender             | Gender of the student                |
| Department         | Field of study                       |
| CGPA               | Academic performance                 |
| Sleep_Duration     | Average hours of sleep per night     |
| Study_Hours        | Average hours spent studying per day |
| Social_Media_Hours | Time spent on social media           |
| Physical_Activity  | Weekly physical activity in minutes  |
| Stress_Level       | Self reported stress level           |

Categorical features are encoded using one hot encoding.

Example code used:

```python
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
```

---

# Model Overview

Two models were trained and evaluated.

1. Logistic Regression
2. XGBoost Classifier

Logistic Regression was used as a baseline model. XGBoost was chosen as the final model because it handles structured tabular data and feature interactions more effectively.

Class imbalance was handled using the `scale_pos_weight` parameter so the minority class receives higher importance during training.

Threshold tuning was applied to prioritize recall for the risk class.

---

# Final Model Performance

XGBoost Test Results

```
precision    recall  f1-score   support

0       0.95      0.72      0.82     13491
1       0.21      0.67      0.32      1509

accuracy                           0.71     15000
macro avg       0.58      0.69      0.57     15000
weighted avg    0.88      0.71      0.77     15000
```

Confusion Matrix

```
[[9719 3772]
 [ 504 1005]]
```

Interpretation

The model successfully identifies approximately 67 percent of students in the risk category. Precision is lower for the risk class but this tradeoff is acceptable since the system prioritizes identifying potentially at risk students.

---

# Project Structure

```
risk_1
│
├── api
│   └── main.py                FastAPI application
│
├── src
│   ├── data
│   │   ├── load_data.py
│   │   └── split_data.py
│   │
│   ├── preprocessing
│   │
│   └── training
│       └── train_model.py
│
├── models
│   └── xgboost_model.pkl
│
├── data
│   └── raw_data
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# System Architecture

The system follows a simple machine learning service architecture.

High Level Flow

```
Student Input
      │
      ▼
FastAPI REST API
      │
      ▼
Preprocessing Layer
(one hot encoding and feature alignment)
      │
      ▼
Trained Model
(XGBoost / Logistic Regression)
      │
      ▼
Prediction Output
(risk probability and risk flag)
```

---

# Deployment Architecture

The API is containerized with Docker and deployed on an AWS EC2 instance.

```
User Request
     │
     ▼
Internet
     │
     ▼
AWS EC2 Instance
     │
     ▼
Docker Container
     │
     ▼
FastAPI Application
     │
     ▼
Loaded ML Model
     │
     ▼
Prediction Response
```

---

# API Endpoints

### Health Check

```
GET /health
```

Returns the status of the service.

---

### Predict Risk

```
POST /predict
```

Example input

```json
{
  "Age": 21,
  "CGPA": 3.2,
  "Sleep_Duration": 6,
  "Study_Hours": 4,
  "Social_Media_Hours": 3,
  "Physical_Activity": 120,
  "Stress_Level": 7,
  "Gender": "Male",
  "Department": "Engineering"
}
```

Example response

```json
{
  "risk_probability": 0.67,
  "risk_flag": 1
}
```

---

# Running the Project Locally

Install dependencies

```
pip install -r requirements.txt
```

Run the API

```
uvicorn api.main:app --reload
```

Open Swagger documentation

```
http://127.0.0.1:8000/docs
```

---

# Docker Deployment

Build Docker image

```
docker build -t risk_1 .
```

Run container

```
docker run -d -p 8000:8000 risk_1
```

Access API

```
http://localhost:8000/docs
```

---

# AWS Deployment

The API is deployed on AWS EC2.

Deployment steps

1. Launch EC2 instance
2. Install Docker
3. Upload project files
4. Build Docker image
5. Run container

Example command

```
docker run -d --restart always -p 8000:8000 risk_1
```

The API becomes accessible through

```
http://<EC2_PUBLIC_IP>:8000/docs
```

---

# Reproducibility

To retrain the model from scratch

1. Install dependencies

```
pip install -r requirements.txt
```

2. Run training script

```
python src/training/train_model.py
```

3. The trained model will be saved to

```
models/xgboost_model.pkl
```

---

# Technologies Used

Python
FastAPI
XGBoost
Scikit learn
Pandas
Docker
AWS EC2
Uvicorn

---

# Future Improvements

• Add authentication to API endpoints
• Add monitoring and logging
• Implement CI CD pipeline for automated deployment
• Add model drift detection
• Build a simple frontend dashboard

---

# Author

Geetheswar Pogula
Computer Science Undergraduate


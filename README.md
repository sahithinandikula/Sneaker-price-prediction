# 👟 The Price Prediction of Sneakers Based on Machine Learning

A full-stack Django web application that predicts sneaker prices using machine learning techniques. This platform allows users and admins to upload sneaker data, visualize pricing trends, and make intelligent price predictions based on historical data.

## 🔍 Project Overview

**Sneaker Price Prediction** is a Django-based web app that enables users to forecast sneaker prices using machine learning. With a focus on user-friendly data interaction, it provides visual insights into sneaker trends and pricing, while also allowing admins to manage uploaded datasets.

## ✨ Features

- 🔐 User registration & authentication (user/admin roles)
- 📂 Upload and manage sneaker datasets (CSV format)
- 📈 Visualize sneaker price trends by region, model, and date
- 🤖 Predict sneaker prices using trained ML models
- 📊 Interactive and responsive data dashboards
- 💻 Mobile-friendly interface with modern UI (Bootstrap-based)

## 🚀 Recent Improvements (ML Enhancements)

We've significantly upgraded the machine learning capabilities of the application:

### 🔧 ML Model Improvements
- Implemented **Random Forest Regressor** as the primary prediction model
- Added **feature engineering** with new temporal features:
  - Days Since Release
  - Release Year/Month
  - Price Premium ratio
- Enhanced **data preprocessing** pipeline:
  - Automated label encoding for categorical variables
  - Standard scaling for numerical features
  - Robust handling of missing/unseen data

### 🛠️ New ML Functionality
- Created dedicated **ML training endpoint** for admin users
- Built **REST API** for price predictions
- Added **model persistence** with joblib serialization
- Implemented **automatic model reloading** when updated
- Added **training metrics** display (MAE, RMSE)

### 📈 Prediction Features
- Interactive **prediction form** with real-time results
- Support for predicting based on:
  - Brand and model
  - Retail price
  - Release date
  - Buyer region
  - Days since release
- Detailed **error handling** for prediction inputs

## 🛠️ Tech Stack

| Layer         | Tools Used                           |
|---------------|---------------------------------------|
| Frontend      | HTML, CSS, Bootstrap 5                |
| Backend       | Python, Django                        |
| ML & Data     | Pandas, NumPy, Scikit-learn, joblib   |
| Database      | SQLite                                |
| Visualization | Matplotlib, Seaborn                   |
| Templates     | Django Templating Engine              |

## ⚙️ Installation

[... rest of the existing installation instructions remain unchanged ...]

## 📊 Machine Learning Model

* **Dataset**: [StockX Sneaker Data 2019 (Kaggle)](https://www.kaggle.com/datasets/stockx/stockx-sneaker-data-2019)
* **Primary Model**: Random Forest Regressor (n_estimators=100)
* **Secondary Model**: Linear Regression (baseline)
* **Libraries**: Scikit-learn, Pandas, NumPy, joblib
* **Target Variable**: `Sale Price`
* **Key Features**:
  - `Brand` (encoded)
  - `Sneaker Name` (encoded)  
  - `Retail Price` (scaled)
  - `Release Year/Month` (temporal)
  - `Buyer Region` (encoded)
  - `Days Since Release` (engineered)

📁 The complete ML implementation is now located in:
- `ml_model/utils.py` (core prediction logic)
- `ml_model/views.py` (API endpoints)
- `templates/users/ml.html` (prediction interface)

[... rest of the README remains unchanged ...]

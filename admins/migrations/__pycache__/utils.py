import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
import os
from django.conf import settings

class SneakerPricePredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.load_model()

    def load_model(self):
        try:
            model_path = os.path.join(settings.MEDIA_ROOT, 'ml_model/model.pkl')
            scaler_path = os.path.join(settings.MEDIA_ROOT, 'ml_model/scaler.pkl')
            encoders_path = os.path.join(settings.MEDIA_ROOT, 'ml_model/encoders.pkl')
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoders = joblib.load(encoders_path)
        except Exception as e:
            print(f"Error loading model: {e}")

    def train_model(self, data_path):
        try:
            df = pd.read_csv(data_path)
            
            # Preprocessing
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            df['Release Date'] = pd.to_datetime(df['Release Date'])
            df['Days Since Release'] = (df['Order Date'] - df['Release Date']).dt.days
            df['Release Year'] = df['Release Date'].dt.year
            df['Release Month'] = df['Release Date'].dt.month
            
            categorical_cols = ['Brand', 'Sneaker Name', 'Buyer Region']
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                
            features = ['Brand', 'Sneaker Name', 'Retail Price', 'Release Year', 
                       'Release Month', 'Buyer Region', 'Days Since Release']
            X = df[features]
            y = df['Sale Price']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.scaler = StandardScaler()
            num_cols = ['Retail Price', 'Release Year', 'Release Month', 'Days Since Release']
            X_train[num_cols] = self.scaler.fit_transform(X_train[num_cols])
            
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            X_test[num_cols] = self.scaler.transform(X_test[num_cols])
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Save artifacts
            os.makedirs(os.path.join(settings.MEDIA_ROOT, 'ml_model'), exist_ok=True)
            joblib.dump(self.model, os.path.join(settings.MEDIA_ROOT, 'ml_model/model.pkl'))
            joblib.dump(self.scaler, os.path.join(settings.MEDIA_ROOT, 'ml_model/scaler.pkl'))
            joblib.dump(self.label_encoders, os.path.join(settings.MEDIA_ROOT, 'ml_model/encoders.pkl'))
            
            return True, mae
        except Exception as e:
            return False, str(e)

    def predict_price(self, input_data):
        try:
            # Convert to DataFrame
            df = pd.DataFrame([input_data])
            
            # Convert dates
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            df['Release Date'] = pd.to_datetime(df['Release Date'])
            df['Days Since Release'] = (df['Order Date'] - df['Release Date']).dt.days
            df['Release Year'] = df['Release Date'].dt.year
            df['Release Month'] = df['Release Date'].dt.month
            
            # Encode categoricals
            for col, le in self.label_encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col])
            
            # Select features and scale
            features = ['Brand', 'Sneaker Name', 'Retail Price', 'Release Year', 
                       'Release Month', 'Buyer Region', 'Days Since Release']
            num_cols = ['Retail Price', 'Release Year', 'Release Month', 'Days Since Release']
            
            if not all(col in df.columns for col in features):
                raise ValueError("Missing required features in input data")
                
            df = df[features]
            df[num_cols] = self.scaler.transform(df[num_cols])
            
            prediction = self.model.predict(df)
            return float(prediction[0])
        except Exception as e:
            raise ValueError(f"Prediction error: {e}")

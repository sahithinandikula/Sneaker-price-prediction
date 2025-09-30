# SNEAKER-PRICE-PR/users/views.py
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from django.conf import settings
from django.contrib.auth.decorators import login_required

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn import metrics

# ----------------------------
# Base ML Models
# ----------------------------
class BaseModel:
    def __init__(self):
        self.model = None
        self.is_fitted = False

    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return {
            'r2_score': metrics.r2_score(y_test, y_pred),
            'mae': metrics.mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        }

class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

class SVRModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SVR(kernel='rbf', C=1.0)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

class ModelComparison:
    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name, model):
        self.models[name] = model

    def train_all_models(self, X_train, y_train):
        for name, model in self.models.items():
            try:
                model.train(X_train, y_train)
            except Exception as e:
                print(f"Error training {name}: {e}")

    def evaluate_all_models(self, X_test, y_test):
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    self.results[name] = model.evaluate(X_test, y_test)
                except Exception as e:
                    print(f"Error evaluating {name}: {e}")

# ----------------------------
# User Registration & Login
# ----------------------------
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.error(request, 'Email or Mobile Already Existed')
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if check.status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                return render(request, 'users/UserHomePage.html')
            else:
                messages.error(request, 'Your Account Not activated')
        except Exception:
            messages.error(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html')

def UserHome(request):
    return render(request, 'users/UserHomePage.html')


# ----------------------------
# Dataset Viewing
# ----------------------------
def DatasetView(request):
    path = settings.MEDIA_ROOT + "/Clean_Shoe_Data.csv"
    try:
        df = pd.read_csv(path, nrows=100)
        df_html = df.to_html()
        return render(request, 'users/viewdataset.html', {'data': df_html})
    except Exception as e:
        return render(request, 'users/viewdataset.html', {'data': f'Error loading data: {str(e)}'})


# ----------------------------
# Machine Learning Training
# ----------------------------
def machinelearning(request):
    path = settings.MEDIA_ROOT + "/Clean_Shoe_Data.csv"
    try:
        df = pd.read_csv(path, parse_dates=True)
        df = df.rename(columns={
            "Order Date": "Order_date",
            "Sneaker Name": "Sneaker_Name",
            "Sale Price": "Sale_Price",
            "Retail Price": "Retail_Price",
            "Release Date": "Release_Date",
            "Shoe Size": "Shoe_Size",
            "Buyer Region": "Buyer"
        })
        df['Order_date'] = pd.to_datetime(df['Order_date'], errors='coerce').map(dt.datetime.toordinal)
        df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce').map(dt.datetime.toordinal)

        X = df.drop(['Sale_Price'], axis=1)
        y = df['Sale_Price']

        object_cols = ['Sneaker_Name', 'Buyer', 'Brand']
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]), index=X_train.index)
        OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]), index=X_valid.index)
        OH_cols_train.columns = OH_encoder.get_feature_names_out(object_cols)
        OH_cols_valid.columns = OH_encoder.get_feature_names_out(object_cols)

        OH_X_train = pd.concat([X_train.drop(object_cols, axis=1), OH_cols_train], axis=1)
        OH_X_valid = pd.concat([X_valid.drop(object_cols, axis=1), OH_cols_valid], axis=1)

        lm = RandomForestRegressor(n_estimators=100, random_state=42)
        lm.fit(OH_X_train, y_train)
        predictions = lm.predict(OH_X_valid)

        MAE = metrics.mean_absolute_error(y_valid, predictions)
        MSE = metrics.mean_squared_error(y_valid, predictions)
        RMSE = np.sqrt(MSE)

        return render(request, "users/ml.html", {"MAE": MAE, "MSE": MSE, "RMSE": RMSE})
    except Exception as e:
        return render(request, "users/ml.html", {"error": str(e)})


# ----------------------------
# Price Prediction
# ----------------------------
def prediction(request):
    if request.method == "POST":
        try:
            path = settings.MEDIA_ROOT + "/Clean_Shoe_Data.csv"
            df = pd.read_csv(path, parse_dates=True)
            df = df.rename(columns={
                "Order Date": "Order_date",
                "Sneaker Name": "Sneaker_Name",
                "Sale Price": "Sale_Price",
                "Retail Price": "Retail_Price",
                "Release Date": "Release_Date",
                "Shoe Size": "Shoe_Size",
                "Buyer Region": "Buyer"
            })

            def safe_date_convert(date_str):
                try:
                    return pd.to_datetime(date_str, errors='coerce').toordinal()
                except:
                    return dt.datetime(2023, 1, 1).toordinal()

            Order_date = safe_date_convert(request.POST.get("Order_date"))
            Release_Date = safe_date_convert(request.POST.get("Release_Date"))
            Retail_Price = float(request.POST.get("Retail_Price") or 0)
            Shoe_Size = float(request.POST.get("Shoe_Size") or 0)
            Brand = request.POST.get("Brand", "Unknown")
            Sneaker_Name = request.POST.get("Sneaker_Name", "Unknown")
            Buyer = request.POST.get("Buyer", "Unknown")

            X = df.drop(['Sale_Price'], axis=1)
            y = df['Sale_Price']

            object_cols = ['Sneaker_Name', 'Buyer', 'Brand']
            OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]), index=X_train.index)
            OH_cols_train.columns = OH_encoder.get_feature_names_out(object_cols)
            OH_X_train = pd.concat([X_train.drop(object_cols, axis=1), OH_cols_train], axis=1)

            lm = RandomForestRegressor(n_estimators=100, random_state=42)
            lm.fit(OH_X_train, y_train)

            new_data = pd.DataFrame({
                'Order_date': [Order_date],
                'Brand': [Brand],
                'Sneaker_Name': [Sneaker_Name],
                'Retail_Price': [Retail_Price],
                'Release_Date': [Release_Date],
                'Shoe_Size': [Shoe_Size],
                'Buyer': [Buyer]
            })

            OH_cols_new = pd.DataFrame(OH_encoder.transform(new_data[object_cols]), index=new_data.index)
            OH_cols_new.columns = OH_encoder.get_feature_names_out(object_cols)
            OH_X_new = pd.concat([new_data.drop(object_cols, axis=1), OH_cols_new], axis=1)

            # Align columns
            for col in OH_X_train.columns:
                if col not in OH_X_new.columns:
                    OH_X_new[col] = 0
            OH_X_new = OH_X_new[OH_X_train.columns]

            y_pred = lm.predict(OH_X_new)
            predicted_price = max(round(float(y_pred[0]), 2), 0)

            return render(request, 'users/prediction.html', {
                'y_pred': [predicted_price],
                'input_data': {
                    'Order_date': request.POST.get("Order_date"),
                    'Brand': Brand,
                    'Sneaker_Name': Sneaker_Name,
                    'Retail_Price': Retail_Price,
                    'Release_Date': request.POST.get("Release_Date"),
                    'Shoe_Size': Shoe_Size,
                    'Buyer': Buyer
                }
            })
        except Exception as e:
            return render(request, 'users/prediction.html', {'error': str(e)})

    return render(request, 'users/prediction.html')

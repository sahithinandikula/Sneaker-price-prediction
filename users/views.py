from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import ThePricePredictionOfsneakersBasedOnMachineLearning

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import classification_report


# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):

    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'Clean_Shoe_Data.csv'
    df = pd.read_csv(path, nrows=100)
    df = df.to_html()
    return render(request, 'users/viewdataset.html', {'data': df})

def machinelearning(request):
    # Reading in the data
    path = settings.MEDIA_ROOT + "\\" + "Clean_Shoe_Data.csv"

    shoe_data = pd.read_csv(path, parse_dates = True)
    df = shoe_data.copy()
    df
    # Checking for missing values in the dataset
    nulls = pd.concat([df.isnull().sum()], axis=1)
    nulls[nulls.sum(axis=1) > 0]
    
    # Renaming columns to get rid of spaces 
    df = df.rename(columns={
    "Order Date": "Order_date",
    "Sneaker Name": "Sneaker_Name",
    "Sale Price": "Sale_Price",
    "Retail Price": "Retail_Price",
    "Release Date": "Release_Date",
    "Shoe Size": "Shoe_Size",
    "Buyer Region": "Buyer"
    })
    df['Order_date'] = pd.to_datetime(df['Order_date'])
    df['Order_date']=df['Order_date'].map(dt.datetime.toordinal)

    df['Release_Date'] = pd.to_datetime(df['Release_Date'])
    df['Release_Date']=df['Release_Date'].map(dt.datetime.toordinal)
    # Starting the linear regression


    X = df.drop(['Sale_Price'], axis=1)
    y = df.Sale_Price
    #X = X.columns.astype(str)
    print(X.columns)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    object_cols = ['Sneaker_Name', 'Buyer', 'Brand']
    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Adding the column names after one hot encoding
    OH_cols_train.columns = OH_encoder.get_feature_names_out(object_cols)
    OH_cols_valid.columns = OH_encoder.get_feature_names_out(object_cols)

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    lm = RandomForestRegressor()
    lm.fit(OH_X_train,y_train)
    predictions = lm.predict(OH_X_valid)
    MAE = metrics.mean_absolute_error(y_valid, predictions)
    MSE =  metrics.mean_squared_error(y_valid, predictions)
    RMSE =  np.sqrt(metrics.mean_squared_error(y_valid, predictions))
   
    
    return render(request,"users/ml.html",{"MAE":MAE,"MSE":MSE,"RMSE":RMSE})



def prediction(request):
    if request.method == "POST":
        import pandas as pd
        from django.conf import settings

        Order_date = request.POST.get("Order_date")
        Brand = request.POST.get("Brand")
        Sneaker_Name = request.POST.get("Sneaker_Name")
        Retail_Price = request.POST.get("Retail_Price")
        Release_Date = request.POST.get("Release_Date")
        Shoe_Size = request.POST.get("Shoe_Size")
        Buyer = request.POST.get("Buyer")
        print(Buyer)

        path = settings.MEDIA_ROOT + "\\" + "Clean_Shoe_Data.csv"

        shoe_data = pd.read_csv(path, parse_dates = True)
        df = shoe_data.copy()
        df
        # Checking for missing values in the dataset
        nulls = pd.concat([df.isnull().sum()], axis=1)
        nulls[nulls.sum(axis=1) > 0]
        
        # Renaming columns to get rid of spaces 
        df = df.rename(columns={
        "Order Date": "Order_date",
        "Sneaker Name": "Sneaker_Name",
        "Sale Price": "Sale_Price",
        "Retail Price": "Retail_Price",
        "Release Date": "Release_Date",
        "Shoe Size": "Shoe_Size",
        "Buyer Region": "Buyer"
        })
        df['Order_date'] = pd.to_datetime(df['Order_date'])
        df['Order_date']=df['Order_date'].map(dt.datetime.toordinal)

        df['Release_Date'] = pd.to_datetime(df['Release_Date'])
        df['Release_Date']=df['Release_Date'].map(dt.datetime.toordinal)
        # Starting the linear regression


        X = df.drop(['Sale_Price'], axis=1)
        y = df.Sale_Price
        #X = X.columns.astype(str)
        print(X.columns)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        object_cols = ['Sneaker_Name', 'Buyer', 'Brand']
        # Apply one-hot encoder to each column with categorical data
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
        OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

        # One-hot encoding removed index; put it back
        OH_cols_train.index = X_train.index
        OH_cols_valid.index = X_valid.index

        # Adding the column names after one hot encoding
        OH_cols_train.columns = OH_encoder.get_feature_names_out(object_cols)
        OH_cols_valid.columns = OH_encoder.get_feature_names_out(object_cols)

        # Remove categorical columns (will replace with one-hot encoding)
        num_X_train = X_train.drop(object_cols, axis=1)
        num_X_valid = X_valid.drop(object_cols, axis=1)

        # Add one-hot encoded columns to numerical features
        OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
        OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

        lm = RandomForestRegressor()
        lm.fit(OH_X_train,y_train)
        test_set = [Order_date, Brand, Sneaker_Name, Retail_Price, Release_Date, Shoe_Size, Buyer]
        print('test_set',test_set)
        new_data = pd.DataFrame({
            'Order_date': [Order_date],
            'Brand': [Brand],
            'Sneaker_Name': [Sneaker_Name],
            'Retail_Price': [Retail_Price],
            'Release_Date': [Release_Date],
            'Shoe_Size': [Shoe_Size],
            'Buyer': [Buyer],
        })


        new_data['Order_date'] = pd.to_datetime(new_data['Order_date'])
        new_data['Order_date']=new_data['Order_date'].map(dt.datetime.toordinal)

        new_data['Release_Date'] = pd.to_datetime(new_data['Release_Date'])
        new_data['Release_Date']=new_data['Release_Date'].map(dt.datetime.toordinal)
        print('type:',new_data['Order_date'].dtype)


        
        new_data_object_cols = new_data[object_cols]





        # Apply one-hot encoding to the new input data using the already fitted encoder
        OH_cols_new = pd.DataFrame(OH_encoder.transform(new_data_object_cols))

        # Set appropriate column names for the one-hot encoded features
        OH_cols_new.columns = OH_encoder.get_feature_names_out(object_cols)

        # Drop the original categorical columns from the new input data
        num_X_new = new_data.drop(object_cols, axis=1)

        # Concatenate the numerical features with the one-hot encoded features for the new input data
        OH_X_new = pd.concat([num_X_new, OH_cols_new], axis=1)

        # Now, OH_X_new contains the transformed input data with one-hot encoded features
        print(OH_X_new)
        print(OH_cols_new.shape)
        
        OH_X_new = OH_X_new.values
        y_pred = lm.predict(OH_X_new)
        print(y_pred)
        return render(request, 'users/prediction.html', {'y_pred': y_pred})

    return render(request, 'users/prediction.html')

        

   
        


    



import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load and prepare the data
file_path = '/content/Book2.csv'
data = pd.read_csv(file_path)
X = data.drop('Status', axis=1)
y = data['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models with scaled data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

lr_model = LogisticRegression(max_iter=1000)  # Increase the number of iterations
lr_model.fit(X_train_scaled, y_train)


# Train models
#rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#rf_model.fit(X_train_imputed, y_train)

#xgb_model = XGBClassifier(n_estimators=100, random_state=42)
#xgb_model.fit(X_train_imputed, y_train)

#lr_model = LogisticRegression()
#lr_model.fit(X_train_imputed, y_train)

# Streamlit app
st.title('Mortality Prediction App')

st.write("Please input the values for the following features:")

# Collecting input from users
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f'{feature}', value=float(X[feature].mean()))

input_df = pd.DataFrame([input_data])

# Handle missing values in user input
input_imputed = imputer.transform(input_df)

# Making predictions
rf_prediction = rf_model.predict(input_imputed)[0]
xgb_prediction = xgb_model.predict(input_imputed)[0]
lr_prediction = lr_model.predict(input_imputed)[0]

st.write(f"Random Forest Prediction: {'Deceased' if rf_prediction == 1 else 'Alive'}")
st.write(f"XGBoost Prediction: {'Deceased' if xgb_prediction == 1 else 'Alive'}")
st.write(f"Logistic Regression Prediction: {'Deceased' if lr_prediction == 1 else 'Alive'}")
    
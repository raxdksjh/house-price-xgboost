import streamlit as st
import pandas as pd
import joblib

# Load model

model = joblib.load("house_price_xgboost_model.pkl")

# Page Config

st.set_page_config(
page_title="House Price Prediction",
page_icon="🏠",
layout="wide"
)

# Header

st.title("🏠 House Price Prediction using XGBoost")

st.markdown("""
This application uses an XGBoost Regressor model trained on the Ames Housing Dataset
to estimate house sale prices based on selected house characteristics.
""")

st.caption("Model: Tuned XGBoost Regressor | Dataset: Ames Housing Dataset")

st.divider()

# Input Layout

col1, col2 = st.columns(2)

with col1:
overall_qual = st.slider(
"Overall Quality (1 = Poor, 10 = Excellent)",
min_value=1,
max_value=10,
value=5
)

```
gr_liv_area = st.number_input(
    "Above Ground Living Area (GrLivArea)",
    min_value=0,
    value=1500
)

total_bsmt_sf = st.number_input(
    "Total Basement Area (TotalBsmtSF)",
    min_value=0,
    value=1000
)

first_flr_sf = st.number_input(
    "First Floor Area (1stFlrSF)",
    min_value=0,
    value=1200
)
```

with col2:
garage_cars = st.number_input(
"Garage Cars",
min_value=0,
max_value=6,
value=2
)

```
garage_area = st.number_input(
    "Garage Area",
    min_value=0,
    value=500
)

full_bath = st.number_input(
    "Full Bathrooms",
    min_value=0,
    max_value=5,
    value=2
)

year_built = st.slider(
    "Year Built",
    min_value=1900,
    max_value=2025,
    value=2000
)
```

st.divider()

# Prediction

if st.button("🔮 Predict House Price", use_container_width=True):

```
input_data = pd.DataFrame({
    'OverallQual': [overall_qual],
    'GrLivArea': [gr_liv_area],
    'GarageCars': [garage_cars],
    'GarageArea': [garage_area],
    'TotalBsmtSF': [total_bsmt_sf],
    '1stFlrSF': [first_flr_sf],
    'FullBath': [full_bath],
    'YearBuilt': [year_built]
})

prediction = model.predict(input_data)

st.success(
    f"🏡 Estimated House Price: ${prediction[0]:,.2f}"
)
```

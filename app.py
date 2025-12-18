import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Chargement du dataset original pour valeurs par défaut ---
df_train = pd.read_csv("data/df_encoded.csv")

# Créer dictionnaire de valeurs par défaut
default_values = {}
for col in df_train.columns:
    if col not in ["SalePrice", "Id"]:
        if df_train[col].dtype in ["int64", "float64"]:
            default_values[col] = df_train[col].median()
        else:
            default_values[col] = df_train[col].mode()[0]

# Charger aussi le dataset brut (avant encodage) pour remplir les menus
raw_data = pd.read_csv("data/train.csv")

# Charger encodeur et scaler (si Ridge utilisé)
try:
    encoder = joblib.load("encoder.pkl")
except:
    encoder = None

# Fonction pour remplir les features manquantes avec valeurs par défaut
def fill_defaults(user_inputs):
    full_inputs = user_inputs.copy()
    for col, val in default_values.items():
        if col not in full_inputs:
            full_inputs[col] = val
    return full_inputs

# Fonction pour transformer inputs → DataFrame utilisable
def create_input_data(user_inputs, scaler=None):
    # Construire un DataFrame avec toutes les colonnes attendues
    expected_cols = df_train.drop(columns=["SalePrice", "Id"]).columns
    df_final = pd.DataFrame([user_inputs]).reindex(columns=expected_cols, fill_value=0)

    # Appliquer le scaler si Ridge
    if scaler is not None:
        df_final[expected_cols] = scaler.transform(df_final[expected_cols])

    return df_final



# --- Interface Streamlit ---
st.sidebar.title("🏡 House Price Prediction")
model_choice = st.sidebar.radio("Choose the page", ["Info", "Predict house price"])

st.title("🏡 House Price Prediction")

if model_choice == "Info":
    st.markdown("""
    This project aims to predict house prices in Ames, Iowa, using advanced regression techniques.  
    Models used:
    - Ridge Regression (with scaling + encoding)  
    - XGBoost (with built-in tree boosting)  
    """)

if model_choice == "Predict house price":
    regression_model = st.radio("Choose the regression model you want to use", ["Ridge", "XGBoost"])

    if regression_model == "Ridge":
        model = joblib.load("ridge_model.pkl")
        scaler = joblib.load("scaler.pkl")
        st.info("**Ridge model loaded**.")
    else:
        model = joblib.load("xgboost_model.pkl")
        scaler = None
        st.info("**XGBoost model loaded**.")

    st.write("### 📝 Please fill in the following fields:")

    # --- Formulaire simplifié ---
    neighborhood = st.selectbox("Neighborhood", sorted(raw_data["Neighborhood"].dropna().unique()),help="Physical locations within Ames city limits")
    grlivarea = st.number_input("Above ground living area (sq ft)", min_value=300, max_value=6000, value=1500,help="Above grade (ground) living area square feet")
    overallqual = st.slider("Overall Quality (1-10)", 1, 10, 5,help="Rates the overall material and finish of the house")
    yearbuilt = st.slider("Year Built", 1880, 2020, 1980,help="Original construction date")
    fullbath = st.slider("Full Bathrooms", 0, 4, 2,help="Full bathrooms above grade")
    totrmsabvgrd = st.slider("Total Rooms Above Grade", 2, 14, 6,help="Total rooms above grade (does not include bathrooms)")
    garagecars = st.selectbox("Garage Capacity (in cars)", [0, 1, 2, 3, 4],help="Size of garage in car capacity")
    garagetype = st.selectbox("Garage Type", list(raw_data["GarageType"].dropna().unique()) + ["No Garage"],help=
                            """
    Garage type:
    - Attchd = Attached to the home  
    - Detchd = Detached from the home  
    - BuiltIn = Built-in, part of house structure  
    - Basment = Basement garage  
    - CarPort = Carport (open-sided shelter)  
    - 2Types = More than one type of garage  
    - No Garage = No garage
    """)
    centralair = st.radio("Has Central Air?", ["Y", "N"],help="Central air conditioning")
    kitchenqual = st.selectbox("Kitchen Quality", ["TA", "Gd", "Ex", "Fa", "Po"],help="""
    Kitchen quality:
    - Ex = Excellent  
    - Gd = Good  
    - TA = Typical/Average  
    - Fa = Fair  
    - Po = Poor
    """)

    if st.button("📊 Predict Price"):
        # Construire inputs utilisateur
        user_inputs = {
            "Neighborhood": neighborhood,
            "GrLivArea": grlivarea,
            "OverallQual": overallqual,
            "YearBuilt": yearbuilt,
            "FullBath": fullbath,
            "TotRmsAbvGrd": totrmsabvgrd,
            "GarageCars": garagecars,
            "GarageType": garagetype,
            "CentralAir": centralair,
            "KitchenQual": kitchenqual
        }

        # Ajouter valeurs par défaut
        full_inputs = fill_defaults(user_inputs)

        # Transformation pour prédiction
        input_data = create_input_data(full_inputs, scaler if regression_model == "Ridge" else None)


        # Prédiction
        log_pred = model.predict(input_data)
        price = np.expm1(log_pred)[0]

        st.success(f"💰 **Predicted Sale Price: ${price:,.0f}**")
        st.write(f"Using the **{regression_model}** model.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built by Fiher Youssef with Streamlit • Dataset: House Prices - Advanced Regression Techniques")
st.write("Source code available on [GitHub](https://github.com/yousseffiher)")

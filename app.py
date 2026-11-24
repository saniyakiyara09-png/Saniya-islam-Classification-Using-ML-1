from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_soil_fertility_model.pkl"
DATA_PATH = ROOT / "data" / "soil_fertility_classification_1000.csv"

st.set_page_config(page_title="Soil Fertility Classifier", page_icon="🌱", layout="centered")
st.title("🌱 Soil Fertility Classification")
st.caption("Machine Learning project | Developed by Saniya Islam")
st.warning("Educational demonstration only. The training dataset is source-calibrated synthetic teaching data, and the fertility labels are derived rather than independently laboratory-measured.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. First run: python -m src.train")
    st.stop()

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

states = sorted(data["State"].dropna().unique().tolist())
districts = sorted(data["District"].dropna().unique().tolist())
soil_types = sorted(data["Soil_Type"].dropna().unique().tolist())
crops = sorted(data["Crop"].dropna().unique().tolist())

st.subheader("Enter soil sample details")
with st.form("soil_form"):
    c1, c2 = st.columns(2)
    with c1:
        state = st.selectbox("State", states)
        district = st.selectbox("District", districts)
        soil_type = st.selectbox("Soil Type", soil_types)
        crop = st.selectbox("Crop", crops)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        ec = st.number_input("Electrical Conductivity (dS/m)", min_value=0.0, value=0.5, step=0.1)
        oc = st.number_input("Organic Carbon (g/kg)", min_value=0.0, value=5.0, step=0.1)
    with c2:
        n = st.number_input("Available Nitrogen (kg/ha)", min_value=0.0, value=250.0, step=1.0)
        p = st.number_input("Available Phosphorus (kg/ha)", min_value=0.0, value=20.0, step=1.0)
        k = st.number_input("Available Potassium (kg/ha)", min_value=0.0, value=150.0, step=1.0)
        sulphur = st.number_input("Sulphur (mg/kg)", min_value=0.0, value=15.0, step=0.5)
        zinc = st.number_input("Zinc (mg/kg)", min_value=0.0, value=1.0, step=0.1)
        boron = st.number_input("Boron (mg/kg)", min_value=0.0, value=0.5, step=0.1)
        iron = st.number_input("Iron (mg/kg)", min_value=0.0, value=5.0, step=0.1)
        manganese = st.number_input("Manganese (mg/kg)", min_value=0.0, value=2.0, step=0.1)
        copper = st.number_input("Copper (mg/kg)", min_value=0.0, value=0.5, step=0.1)
    submitted = st.form_submit_button("Predict Fertility Class", type="primary")

if submitted:
    sample = pd.DataFrame([{
        "pH": ph, "Electrical_Conductivity_dS_m": ec, "Organic_Carbon_g_kg": oc,
        "Available_N_kg_ha": n, "Available_P_kg_ha": p, "Available_K_kg_ha": k,
        "Sulphur_mg_kg": sulphur, "Zinc_mg_kg": zinc, "Boron_mg_kg": boron,
        "Iron_mg_kg": iron, "Manganese_mg_kg": manganese, "Copper_mg_kg": copper,
        "State": state, "District": district, "Soil_Type": soil_type, "Crop": crop
    }])
    prediction = model.predict(sample)[0]
    st.success(f"Predicted class: **{prediction}**")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        classes = model.named_steps["classifier"].classes_
        st.write("Model class probabilities")
        st.bar_chart(pd.DataFrame({"Probability": probabilities}, index=classes))
    st.info("This is a model prediction from a synthetic educational dataset, not a substitute for a certified soil laboratory test.")

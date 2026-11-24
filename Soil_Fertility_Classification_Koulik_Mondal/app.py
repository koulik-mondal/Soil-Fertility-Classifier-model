from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_soil_fertility_model.pkl"
DATA_PATH = ROOT / "data" / "soil_fertility_classification_1000.csv"
RESULTS_PATH = ROOT / "outputs" / "model_comparison_results.csv"
IMPORTANCE_PATH = ROOT / "outputs" / "feature_importance_results.csv"

st.set_page_config(page_title="Soil Fertility Classifier model", layout="centered")
st.title(" Soil Fertility Classification by Koulik Mondal using ml")
st.caption("Machine Learning project | Developed by Koulik Mondal (BRAINWARE UNIVERSITY)")
# st.warning("Educational demonstration only. The training dataset is source-calibrated synthetic teaching data, "
#            "and the fertility labels are derived rather than independently laboratory-measured.")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the saved pipeline. If it is missing or was saved with another
    scikit-learn version, retrain automatically (takes about a minute)."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from src import train
        train.main()
        return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
data = load_data()

state_district = data.groupby("State")["District"].unique().apply(sorted).to_dict()
states = sorted(state_district)
soil_types = sorted(data["Soil_Type"].dropna().unique().tolist())
crops = sorted(data["Crop"].dropna().unique().tolist())

st.subheader("Enter soil sample details")

# State/district sit outside the form so the district list updates when the state changes.
c0, c00 = st.columns(2)
with c0:
    state = st.selectbox("State", states, index=states.index("West Bengal") if "West Bengal" in states else 0)
with c00:
    district = st.selectbox("District", state_district[state])

with st.form("soil_form"):
    c1, c2 = st.columns(2)
    with c1:
        soil_type = st.selectbox("Soil Type", soil_types)
        crop = st.selectbox("Crop", crops)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        ec = st.number_input("Electrical Conductivity (dS/m)", min_value=0.0, value=0.35, step=0.05)
        oc = st.number_input("Organic Carbon (g/kg)", min_value=0.0, value=8.0, step=0.1)
        n = st.number_input("Available Nitrogen (kg/ha)", min_value=0.0, value=180.0, step=1.0)
    with c2:
        p = st.number_input("Available Phosphorus (kg/ha)", min_value=0.0, value=35.0, step=1.0)
        k = st.number_input("Available Potassium (kg/ha)", min_value=0.0, value=160.0, step=1.0)
        sulphur = st.number_input("Sulphur (mg/kg)", min_value=0.0, value=12.0, step=0.5)
        zinc = st.number_input("Zinc (mg/kg)", min_value=0.0, value=0.8, step=0.1)
        boron = st.number_input("Boron (mg/kg)", min_value=0.0, value=0.7, step=0.1)
        iron = st.number_input("Iron (mg/kg)", min_value=0.0, value=20.0, step=0.5)
        manganese = st.number_input("Manganese (mg/kg)", min_value=0.0, value=8.0, step=0.5)
        copper = st.number_input("Copper (mg/kg)", min_value=0.0, value=0.7, step=0.1)
    submitted = st.form_submit_button("Predict Fertility Class", type="primary")

if submitted:
    sample = pd.DataFrame([{
        "pH": ph, "Electrical_Conductivity_dS_m": ec, "Organic_Carbon_g_kg": oc,
        "Available_N_kg_ha": n, "Available_P_kg_ha": p, "Available_K_kg_ha": k,
        "Sulphur_mg_kg": sulphur, "Zinc_mg_kg": zinc, "Boron_mg_kg": boron,
        "Iron_mg_kg": iron, "Manganese_mg_kg": manganese, "Copper_mg_kg": copper,
        "State": state, "District": district, "Soil_Type": soil_type, "Crop": crop,
    }])
    prediction = model.predict(sample)[0]
    st.success(f"Predicted class: **{prediction}**")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        classes = model.named_steps["classifier"].classes_
        st.write("Model class probabilities")
        st.bar_chart(pd.DataFrame({"Probability": probabilities}, index=classes))
    st.info("This is a model prediction from a synthetic educational dataset, "
            "not a substitute for a certified soil laboratory test.")

with st.expander("Model results from training (your dataset)"):
    if RESULTS_PATH.exists():
        st.write("Model comparison (stratified 80/20 hold-out, 200 test rows)")
        st.dataframe(pd.read_csv(RESULTS_PATH).round(4), hide_index=True)
    if IMPORTANCE_PATH.exists():
        st.write("Top features by permutation importance")
        imp = pd.read_csv(IMPORTANCE_PATH).head(8).set_index("Feature")
        st.bar_chart(imp["Importance_Mean"])

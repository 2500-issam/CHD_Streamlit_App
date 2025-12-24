import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import joblib
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import FunctionTransformer

# -------------------------------
# 🔧 Configuration de la page
# -------------------------------
st.set_page_config(
    page_title="Prédiction CHD",
    page_icon="🫀",
    layout="centered"
)

st.title("🩺 Prédiction du risque de maladie cardiaque (CHD)")
st.write("""
Cette application utilise un modèle **Machine Learning** entraîné sur le dataset **CHD.csv**  
(pipeline : prétraitement + PCA + régression logistique).  
""")

# -------------------------------
# 📦 Chargement du modèle
# -------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("Model.pkl")
    except Exception as e:
        st.error("❌ Erreur lors du chargement du modèle Model.pkl")
        st.write(e)
        return None

model = load_model()

if model is None:
    st.stop()

# -------------------------------
# 🧼 Nettoyage de famhist
# -------------------------------
def clean_famhist(df):
    df = df.copy()
    df["famhist"] = df["famhist"].astype(str).str.strip().str.lower()
    return df

# -------------------------------
# 🧾 Formulaire de saisie
# -------------------------------
st.subheader("🧾 Informations du patient")

with st.form("inputs_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", min_value=10, max_value=100, value=50)
        sbp = st.number_input("Pression systolique (sbp)", value=140.0)
        ldl = st.number_input("LDL (mauvais cholestérol)", value=4.0)

    with col2:
        adiposity = st.number_input("Adiposity", value=25.0)
        obesity = st.number_input("Obesity", value=30.0)
        famhist = st.selectbox("Antécédents familiaux", ["Present", "Absent"])

    submitted = st.form_submit_button("Prédire le risque")

# -------------------------------
# 🔮 Prédiction
# -------------------------------
if submitted:

    input_df = pd.DataFrame([{
        "sbp": sbp,
        "ldl": ldl,
        "adiposity": adiposity,
        "obesity": obesity,
        "age": age,
        "famhist": famhist
    }])

    # Nettoyage du champ catégoriel
    input_df = clean_famhist(input_df)

    st.write("### 📄 Données saisies")
    st.dataframe(input_df)

    # Prédiction
    proba_chd = model.predict_proba(input_df)[0, 1]
    pred_chd = model.predict(input_df)[0]

    st.subheader("🔎 Résultat")

    st.write(f"**Probabilité estimée de CHD : `{proba_chd:.2f}`**")

    if pred_chd == 1:
        st.error("🔴 Risque élevé de maladie cardiaque (CHD = 1).")
    else:
        st.success("🟢 Risque faible de maladie cardiaque (CHD = 0).")

    st.info("⚠️ Cette application ne remplace pas un avis médical.")


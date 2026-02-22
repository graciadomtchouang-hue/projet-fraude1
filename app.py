import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Entraînement du modèle au démarrage
@st.cache_resource
def train_model():
    df = pd.read_csv('creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time'] = scaler.fit_transform(X[['Time']])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Rééquilibrage
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]
    np.random.seed(42)
    oversampled = np.random.choice(fraud_idx, size=len(legit_idx), replace=True)
    idx = np.concatenate([legit_idx, oversampled])
    np.random.shuffle(idx)

    X_bal = X_train.iloc[idx]
    y_bal = y_train.iloc[idx]

    # Modèle allégé
    rf = RandomForestClassifier(n_estimators=50, random_state=42,
                                 class_weight='balanced', n_jobs=-1)
    rf.fit(X_bal, y_bal)
    return rf, scaler

# Interface
st.title("🔍 Détection de Fraude à la Carte de Crédit")
st.markdown("Cette application prédit si une transaction est **frauduleuse ou légitime**.")
st.markdown("---")

with st.spinner("⏳ Chargement du modèle en cours... (1-2 minutes)"):
    model, scaler = train_model()

st.success("✅ Modèle prêt !")
st.markdown("---")

st.sidebar.header("📋 Paramètres de la transaction")
amount = st.sidebar.number_input("Montant (€)", min_value=0.0, value=100.0, step=1.0)
time = st.sidebar.number_input("Temps (secondes)", min_value=0.0, value=50000.0)

v_values = []
for i in range(1, 29):
    v = st.sidebar.slider(f"V{i}", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    v_values.append(v)

# Normalisation
amount_scaled = scaler.transform([[amount]])[0][0]
time_scaled = (time - 94813.0) / 47488.0

features = np.array([[time_scaled] + v_values + [amount_scaled]])

if st.button("🔎 Analyser la transaction"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    st.markdown("---")
    st.subheader("📊 Résultat")

    col1, col2 = st.columns(2)
    col1.metric("Probabilité Légitime", f"{proba[0]*100:.2f}%")
    col2.metric("Probabilité Fraude", f"{proba[1]*100:.2f}%")

    if prediction == 1:
        st.error("⚠️ TRANSACTION FRAUDULEUSE DÉTECTÉE !")
    else:
        st.success("✅ Transaction légitime")

st.markdown("---")
st.markdown("*Modèle : Random Forest | Dataset : Credit Card Fraud Detection (Kaggle)*")
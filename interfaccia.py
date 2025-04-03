import pandas as pd
import pickle
import streamlit as st

# Carica i modelli salvati
with open('best_model_lat_long.pickle', 'rb') as model_file:
    best_model_lat_long = pickle.load(model_file)

with open('best_model_other_features.pickle', 'rb') as model_file:
    best_model_other_features = pickle.load(model_file)

# Crea un'interfaccia con Streamlit
st.set_page_config(page_title="Prezzo al Metro Quadro di Immobili", layout="centered")

# Titolo principale
st.title("Stima del Prezzo al Metro Quadro di Immobili")
st.markdown("""
Selezionare il metodo di previsione da utilizzare:
""")

# Finestra di scelta per la modalità di previsione
prediction_mode = st.selectbox("Seleziona il metodo di previsione:", ("Latitudine e Longitudine", "Altre Variabili"))

if prediction_mode == "Latitudine e Longitudine":
    # Sezione per l'input dell'utente
    st.subheader("Inserisci i Dati di Latitudine e Longitudine:")
    lat = st.number_input("Latitudine (25.0 - 25.1)", min_value=25.0, max_value=25.1, step=0.01)
    long = st.number_input("Longitudine (121.4 - 121.5)", min_value=121.4, max_value=121.5, step=0.01)

    # Pulsante per la previsione
    if st.button("Predici Prezzo", key="predict_lat_long"):
        if (25.0 <= lat <= 25.1) and (121.4 <= long <= 121.5):
            prediction = best_model_lat_long.predict([[lat, long]])[0]
            st.success(f"Il prezzo stimato al metro quadro è: **{prediction:.2f} NT$**")
        else:
            st.error("❌ Valori di latitudine e longitudine non validi.")

elif prediction_mode == "Altre Variabili":
    # Sezione per l'input dell'utente
    st.subheader("Inserisci i Dati Altri Variabili:")
    age = st.number_input("Età dell'immobile (in anni)", min_value=0)
    distance_mrt = st.number_input("Distanza dalla stazione MRT (in metri)", min_value=0)
    minimarkets = st.number_input("Numero di minimarket nelle vicinanze", min_value=0)

    # Pulsante per la previsione
    if st.button("Predici Prezzo", key="predict_other"):
        prediction = best_model_other_features.predict([[age, distance_mrt, minimarkets]])[0]
        st.success(f"Il prezzo stimato al metro quadro è: **{prediction:.2f} NT$**")

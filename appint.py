import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("Prédicteur de Survie - Titanic")
st.write("Entrez les informations du passager pour tester les capacités de prédiction du modèle.")

# Chargement du modèle sauvegardé
try:
    model = joblib.load("modele.pkl")
except:
    st.error("Erreur : Le fichier 'modele.pkl' est introuvable. Vérifiez l'emplacement du fichier.")

# Création de l'interface avec deux colonnes
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Classe du voyageur (1: Haute, 3: Basse)", [1, 2, 3])
    sex = st.selectbox("Sexe", ["male", "female"])
    age = st.slider("Âge", 0, 80, 25)

with col2:
    sibsp = st.number_input("Nombre de frères/sœurs & époux à bord", 0, 10, 0)
    parch = st.number_input("Nombre de parents & enfants à bord", 0, 10, 0)
    fare = st.number_input("Prix du billet (Fare)", 0.0, 512.0, 32.0)
    embarked = st.selectbox("Port d'embarquement", ["S", "C", "Q"])

# Bouton de prédiction
if st.button("Lancer la prédiction"):
    # Création du DataFrame pour le modèle (doit avoir les mêmes colonnes que X_train)
    input_df = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    input_df['family'] = input_df['SibSp'] + input_df['Parch']
    
    # Exécution de la prédiction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # Affichage du résultat
    st.divider()
    if prediction[0] == 1:
        st.success(f"🎉 Le passager aurait probablement SURVÉCU (Probabilité : {probability[0][1]:.2%})")
    else:
        st.error(f"💀 Le passager n'aurait probablement PAS survécu (Probabilité de survie : {probability[0][1]:.2%})")
import streamlit as st
import players_app
import clubs_app

st.title("Comparateur Rugby 🏉")

menu = st.sidebar.radio(
    "Que voulez-vous comparer ?",
    ("Accueil", "Joueurs", "Clubs")
)

if menu == "Accueil":
    st.write("Bienvenue dans l'application de comparaison rugby.")
    st.write("Choisissez une option dans le menu de gauche.")
elif menu == "Joueurs":
    players_app.run()
elif menu == "Clubs":
    clubs_app.run()

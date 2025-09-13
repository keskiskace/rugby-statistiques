import streamlit as st
import players_app
import clubs_app

st.set_page_config(page_title="Rugby Top14/Prod2", layout="wide")
st.title("🏉 Rugby Top14/Prod2 Dashboard")

menu = st.sidebar.radio("Navigation", ["Accueil", "Joueurs", "Clubs"])

if menu == "Accueil":
    st.write("Bienvenue sur le comparateur Rugby 🏉")
    st.write("Choisissez **Joueurs** ou **Clubs** dans le menu.")
elif menu == "Joueurs":
    players_app.run()
elif menu == "Clubs":
    clubs_app.run()

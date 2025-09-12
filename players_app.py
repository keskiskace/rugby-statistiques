import streamlit as st
import db

def run():
    st.header("Comparaison de joueurs")
    # Exemple simple
    players = db.get_players()
    choice = st.selectbox("Choisir un joueur", players)
    st.write(f"Statistiques de {choice}")

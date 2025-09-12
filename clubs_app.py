import streamlit as st
import db

def run():
    st.header("Comparaison de clubs")
    clubs = db.get_clubs()
    choice = st.selectbox("Choisir un club", clubs)
    st.write(f"Statistiques du club : {choice}")

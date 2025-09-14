# topflop_players_app.py
import streamlit as st
import pandas as pd
import numpy as np
from utils import load_players, dataframe_to_image

st.set_page_config(page_title="Top/Flop Joueurs", layout="wide")
st.title("🏉 Top / Flop 10 — Joueurs")

df = load_players()
if df is None or df.empty:
    st.warning("Aucune donnée joueurs trouvée dans la DB.")
    st.stop()

# Saison
saisons_disponibles = sorted(df['saison'].dropna().unique(), reverse=True)
default_saison = [saisons_disponibles[0]] if saisons_disponibles else []
selected_saison = st.selectbox("Choisir une saison", saisons_disponibles, index=0)

df_filtered = df[df['saison'] == selected_saison].copy()

# Division
divisions_dispo = sorted(df_filtered['division'].dropna().unique())
choice_division = st.selectbox("Choisir une division", ["Toutes"] + divisions_dispo)

if choice_division != "Toutes":
    df_filtered = df_filtered[df_filtered['division'] == choice_division]

# Club (optionnel)
clubs_dispo = sorted(df_filtered['club'].dropna().unique())
choice_club = st.selectbox("Choisir un club (optionnel)", ["Aucun"] + clubs_dispo)

if choice_club != "Aucun":
    df_filtered = df_filtered[df_filtered['club'] == choice_club]

# Statistique
numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ["player_id", "id"]
stat_cols = [c for c in numeric_cols if c not in exclude_cols]

if not stat_cols:
    st.warning("Aucune statistique disponible pour ces filtres.")
    st.stop()

choice_stat = st.selectbox("Choisir une statistique", stat_cols)

# Type (Top ou Flop)
choice_type = st.radio("Choisir le type", ["Top 10", "Flop 10"])

if not df_filtered.empty and choice_stat:
    if choice_type == "Top 10":
        result = df_filtered.nlargest(10, choice_stat)[["nom", "club", "division", choice_stat]]
    else:
        result = df_filtered.nsmallest(10, choice_stat)[["nom", "club", "division", choice_stat]]

    st.subheader(f"{choice_type} sur {choice_stat}")
    st.dataframe(result)

    st.download_button("⬇️ Télécharger en CSV",
                       result.to_csv(index=False).encode("utf-8"),
                       file_name=f"{choice_type.lower().replace(' ', '_')}_joueurs_{choice_stat}.csv",
                       mime="text/csv")

    img_file = dataframe_to_image(result, f"{choice_type.lower()}_joueurs_{choice_stat}.png")
    with open(img_file, "rb") as f:
        st.download_button("⬇️ Télécharger en PNG", f,
                           file_name=f"{choice_type.lower()}_joueurs_{choice_stat}.png",
                           mime="image/png")
else:
    st.info("Aucune donnée pour ce filtre.")

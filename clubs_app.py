import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from common import make_scatter_radar, dataframe_to_image

DB_FILE = "top14_prod2_25_26_players_clubs.db"

@st.cache_data
def load_clubs():
    with sqlite3.connect(DB_FILE) as con:
        clubs_df = pd.read_sql("SELECT * FROM clubs", con)
    for col in clubs_df.columns:
        try:
            clubs_df[col] = pd.to_numeric(clubs_df[col], errors="ignore")
        except Exception:
            pass
    return clubs_df

def run():
    st.header("📊 Comparaison des Clubs")

    clubs_df = load_clubs()
    if clubs_df.empty:
        st.warning("La table 'clubs' est vide ou introuvable.")
        return

    saisons_disponibles = sorted(clubs_df['saison'].dropna().unique(), reverse=True)
    selected_saisons = st.multiselect("Choisir une ou plusieurs saisons (clubs)",
                                      saisons_disponibles,
                                      default=[saisons_disponibles[0]] if saisons_disponibles else [])
    clubs_filtered = clubs_df[clubs_df['saison'].isin(selected_saisons)].copy()
    if len(selected_saisons) > 1:
        clubs_filtered['display_name'] = clubs_filtered['club'].astype(str) + " (" + clubs_filtered['saison'].astype(str) + ")"
    else:
        clubs_filtered['display_name'] = clubs_filtered['club'].astype(str)

    club_options = clubs_filtered['display_name'].sort_values().unique().tolist()
    selected_clubs = st.multiselect("Choisir un ou plusieurs clubs",
                                    club_options,
                                    default=[club_options[0]] if club_options else [])
    selected_clubs_df = clubs_filtered[clubs_filtered['display_name'].isin(selected_clubs)].copy()

    if not selected_clubs_df.empty:
        numeric_cols = clubs_filtered.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["classement"]
        stat_cols = [c for c in numeric_cols if c not in exclude_cols]

        selected_stats = st.multiselect("Choisir les statistiques à afficher dans le radar (clubs)",
                                        options=stat_cols,
                                        default=stat_cols[:5] if len(stat_cols) > 5 else stat_cols)

        if selected_stats:
            radar_data = []
            for _, club in selected_clubs_df.iterrows():
                entry = {"Joueur": club.get("club", "")}
                for stat in selected_stats:
                    entry[stat] = club.get(stat, np.nan)
                radar_data.append(entry)

            radar_df = pd.DataFrame(radar_data)
            fig = make_scatter_radar(radar_df, selected_stats)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Tableau comparatif des clubs")
            table_df = radar_df.set_index("Joueur").T
            st.dataframe(table_df)

            st.download_button("⬇️ Télécharger en CSV", table_df.to_csv().encode("utf-8"),
                               file_name="comparatif_clubs.csv", mime="text/csv")

            img_file = dataframe_to_image(table_df, "comparatif_clubs.png")
            with open(img_file, "rb") as f:
                st.download_button("⬇️ Télécharger en PNG", f, file_name="comparatif_clubs.png", mime="image/png")

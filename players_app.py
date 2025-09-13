import os
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from common import make_scatter_radar, dataframe_to_image, get_image_safe, download_missing_photos

DB_FILE = "top14_prod2_25_26_players_clubs.db"
IMAGES_DIR = "images"
FALLBACK_IMAGE = os.path.join(IMAGES_DIR, "no_player.webp")

@st.cache_data
def load_players():
    with sqlite3.connect(DB_FILE) as con:
        df = pd.read_sql("SELECT * FROM players", con)

    # conversions numériques
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

    # ratios utiles
    if "poids_kg" in df.columns and "taille_cm" in df.columns:
        df['ratio_poids_taille'] = (
            pd.to_numeric(df['poids_kg'], errors='coerce') / pd.to_numeric(df['taille_cm'], errors='coerce')
        ).replace([np.inf, -np.inf], np.nan).round(2)

    if "metres_parcourus" in df.columns and "courses" in df.columns:
        df['ratio_metres_courses'] = (
            pd.to_numeric(df['metres_parcourus'], errors='coerce') / pd.to_numeric(df['courses'], errors='coerce')
        ).replace([np.inf, -np.inf], np.nan).round(2)

    if "temps_jeu_min" in df.columns and "nombre_matchs_joues" in df.columns:
        df['ratio_min_matchs'] = (
            pd.to_numeric(df['temps_jeu_min'], errors='coerce') / pd.to_numeric(df['nombre_matchs_joues'], errors='coerce')
        ).replace([np.inf, -np.inf], np.nan).round(2)

    return df


def infer_league(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "division" in df.columns:
        df["__league__"] = df["division"].astype(str)
    else:
        df["__league__"] = None
    return df


def run():
    st.header("🔎 Joueurs — Comparateur")

    df = load_players()
    if df.empty:
        st.warning("La table 'players' est vide ou introuvable dans la DB.")
        return

    # Saisons disponibles
    saisons_disponibles = sorted(df['saison'].dropna().unique(), reverse=True)
    selected_saisons = st.multiselect(
        "Choisir une ou plusieurs saisons (joueurs)",
        saisons_disponibles,
        default=[saisons_disponibles[0]] if saisons_disponibles else []
    )
    df_filtered = df[df['saison'].isin(selected_saisons)].copy()
    if len(selected_saisons) > 1:
        df_filtered['display_name'] = df_filtered['nom'].astype(str) + " (" + df_filtered['saison'].astype(str) + ")"
    else:
        df_filtered['display_name'] = df_filtered['nom'].astype(str)

    if 'nombre_matchs_joues' in df_filtered.columns:
        df_nonzero = df_filtered[pd.to_numeric(df_filtered['nombre_matchs_joues'], errors='coerce').fillna(0) > 0].copy()
    else:
        df_nonzero = df_filtered.copy()

    df_labeled = infer_league(df_nonzero)

    # Sélection joueurs
    player_options = df_filtered['display_name'].sort_values().unique().tolist()
    selected_names = st.multiselect(
        "Choisir un ou plusieurs joueurs",
        player_options,
        default=[player_options[0]] if player_options else []
    )
    selected_players = df_filtered[df_filtered['display_name'].isin(selected_names)].copy()

    # Affichage infos joueurs
    for _, joueur in selected_players.iterrows():
        st.subheader(joueur.get('nom', ''))
        photo_to_show = get_image_safe(joueur, IMAGES_DIR, FALLBACK_IMAGE)
        st.image(photo_to_show, caption=joueur.get('club', ''), width=150)
        st.json({
            "Club": joueur.get('club', 'N/A'),
            "Poste": joueur.get('poste', 'N/A'),
            "Âge": joueur.get('age', 'N/A'),
            "Taille (cm)": joueur.get('taille_cm', 'N/A'),
            "Poids (kg)": joueur.get('poids_kg', 'N/A'),
            "Ratio poids/taille": joueur.get('ratio_poids_taille', 'N/A')
        })

    # Radar
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["player_id", "id"]
    stat_cols = [c for c in numeric_cols if c not in exclude_cols]

    selected_stats = st.multiselect(
        "Choisir les statistiques à afficher dans le radar",
        options=stat_cols,
        default=stat_cols[:5] if len(stat_cols) > 5 else stat_cols
    )

    if selected_stats and not selected_players.empty:
        radar_data = []
        for _, joueur in selected_players.iterrows():
            entry = {"Joueur": joueur.get('nom', '')}
            for stat in selected_stats:
                entry[stat] = joueur.get(stat, np.nan)
            radar_data.append(entry)

        radar_df = pd.DataFrame(radar_data)
        fig = make_scatter_radar(radar_df, selected_stats)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Tableau comparatif des joueurs")
        table_df = radar_df.set_index("Joueur").T
        st.dataframe(table_df)

        st.download_button("⬇️ Télécharger en CSV", table_df.to_csv().encode("utf-8"),
                           file_name="comparatif_joueurs.csv", mime="text/csv")

        img_file = dataframe_to_image(table_df, "comparatif_joueurs.png")
        with open(img_file, "rb") as f:
            st.download_button("⬇️ Télécharger en PNG", f, file_name="comparatif_joueurs.png", mime="image/png")

    # Top / Flop
    st.header("🏆 Top / Flop 10 Joueurs")
    choice_type = st.radio("Choisir le type", ["Top 10", "Flop 10"])
    choice_division = st.selectbox("Choisir une division", ["Toutes", "Top14", "ProD2"])
    club_choices = ["Aucun"] + sorted(df_filtered['club'].dropna().unique().tolist())
    choice_club = st.selectbox("Choisir un club (optionnel)", club_choices)
    choice_stat = st.selectbox("Choisir une statistique", stat_cols) if stat_cols else None

    filtered_for_top = df_nonzero.copy()
    if choice_division != "Toutes" and "division" in filtered_for_top.columns:
        filtered_for_top = filtered_for_top[filtered_for_top['division'].str.contains(choice_division, case=False, na=False)]
    if choice_club != "Aucun":
        filtered_for_top = filtered_for_top[filtered_for_top['club'] == choice_club]

    if not filtered_for_top.empty and choice_stat:
        if choice_type == "Top 10":
            result = filtered_for_top.nlargest(10, choice_stat)[["nom", "club", "division", choice_stat]]
        else:
            result = filtered_for_top.nsmallest(10, choice_stat)[["nom", "club", "division", choice_stat]]

        st.subheader(f"{choice_type} sur {choice_stat}")
        st.dataframe(result)

        st.download_button("⬇️ Télécharger en CSV",
                           result.to_csv(index=False).encode("utf-8"),
                           file_name=f"{choice_type.lower()}_{choice_stat}.csv",
                           mime="text/csv")

        img_file = dataframe_to_image(result, f"{choice_type.lower()}_{choice_stat}.png")
        with open(img_file, "rb") as f:
            st.download_button("⬇️ Télécharger en PNG", f, file_name=f"{choice_type.lower()}_{choice_stat}.png", mime="image/png")

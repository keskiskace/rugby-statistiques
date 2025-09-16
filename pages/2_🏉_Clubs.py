# clubs_app.py
import streamlit as st
import pandas as pd
import numpy as np
from utils import load_clubs, dataframe_to_image, make_scatter_radar

st.set_page_config(page_title="Comparateur Clubs", layout="wide")
st.title("🏉 Comparateur — Clubs (Top14 / ProD2)")

# Charger les données
clubs_df = load_clubs()
if clubs_df is None or clubs_df.empty:
    st.warning("Aucune donnée clubs trouvée dans la DB.")
    st.stop()

# ====== fonction reset ======
def reset_clubs_filters():
    keys = ['clubs_saisons', 'clubs_divisions', 'clubs_names', 'clubs_stats']
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

st.sidebar.button("Réinitialiser les filtres Clubs", on_click=reset_clubs_filters)

# ====== FILTRES ======
st.subheader("🎛️ Filtres")

# 1) Saison
saisons_disponibles = sorted(clubs_df['saison'].dropna().unique(), reverse=True)
default_saison = [saisons_disponibles[0]] if saisons_disponibles else []
selected_saisons = st.multiselect(
    "1) Choisir saison(s)",
    saisons_disponibles,
    default=st.session_state.get('clubs_saisons', default_saison),
    key='clubs_saisons'
)

if selected_saisons:
    df_clubs = clubs_df[clubs_df['saison'].isin(selected_saisons)].copy()
else:
    df_clubs = clubs_df.copy()

# 2) Filtre journée (optionnel)
if "journée" in clubs_df.columns:
    journees_dispo = sorted(clubs_df.loc[clubs_df['saison'].isin(selected_saisons), "journée"].dropna().unique().tolist())
    choice_journee = st.selectbox("3) Choisir une journée", ["Dernière disponible"] + [f"J{j}" for j in journees_dispo])

    if choice_journee == "Dernière disponible":
        df_clubs = df_clubs.loc[
            df_clubs.groupby(["saison", "club"])["journée"].transform("max") == df_clubs["journée"]
        ]
    else:
        journee_num = int(choice_journee[1:])
        df_clubs = df_clubs[
            ((df_clubs["saison"].isin(selected_saisons)) & (df_clubs["journée"] == journee_num))
            | ((~df_clubs["saison"].isin(selected_saisons)) & (
                df_clubs.groupby(["saison", "club"])["journée"].transform("max") == df_clubs["journée"]
            ))
        ]

# 3) Division
if "division" in df_clubs.columns:
    divisions_dispo = sorted(df_clubs['division'].dropna().unique())
    selected_div = st.multiselect(
        "2) Choisir division(s) (optionnel)",
        divisions_dispo,
        default=st.session_state.get('clubs_divisions', []),
        key='clubs_divisions'
    )
    if selected_div:
        df_clubs = df_clubs[df_clubs['division'].isin(selected_div)]

# 3) Clubs
club_options = df_clubs['club'].sort_values().unique().tolist()
if club_options:
    selected_names = st.multiselect(
        "3) Choisir club(s)",
        club_options,
        default=st.session_state.get('clubs_names', []),
        key='clubs_names'
    )
    selected_clubs_df = df_clubs[df_clubs['club'].isin(selected_names)].copy()
else:
    selected_clubs_df = pd.DataFrame()
    st.warning("Aucun club disponible avec ces filtres.")

# ====== Clubs types (moyennes par division) ======
extra_clubs = []
if not df_clubs.empty:
    if (df_clubs['division'].str.contains("Top14", case=False, na=False)).any():
        avg_top14 = df_clubs[df_clubs['division'].str.contains("Top14", case=False, na=False)].mean(numeric_only=True).round(1)
        extra_clubs.append({"club": "Club type Top14", **avg_top14.to_dict()})
    if (df_clubs['division'].str.contains("ProD2", case=False, na=False)).any():
        avg_prod2 = df_clubs[df_clubs['division'].str.contains("ProD2", case=False, na=False)].mean(numeric_only=True).round(1)
        extra_clubs.append({"club": "Club type ProD2", **avg_prod2.to_dict()})

extra_clubs_df = pd.DataFrame(extra_clubs) if extra_clubs else pd.DataFrame()

# Fusion si clubs types sélectionnés
if not extra_clubs_df.empty:
    clubs_extended = pd.concat([df_clubs, extra_clubs_df], ignore_index=True)
else:
    clubs_extended = df_clubs.copy()

# ====== Affichage comparatif ======
if not selected_clubs_df.empty:
    st.subheader("📊 Comparaison des Clubs")

    numeric_cols = clubs_extended.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["classement"]
    stat_cols = [c for c in numeric_cols if c not in exclude_cols]

    selected_stats = st.multiselect(
        "Choisir statistiques pour le radar",
        options=stat_cols,
        default=st.session_state.get('clubs_stats', stat_cols[:5] if len(stat_cols) > 5 else stat_cols),
        key='clubs_stats'
    )

    if selected_stats:
        radar_data = []
        for _, club in selected_clubs_df.iterrows():
            entry = {"Joueur": club.get("club", "")}
            for stat in selected_stats:
                entry[stat] = club.get(stat, np.nan)
            radar_data.append(entry)

        radar_df = pd.DataFrame(radar_data)
        fig = make_scatter_radar(radar_df, selected_stats)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

        st.subheader("📊 Tableau comparatif des clubs")
        table_df = radar_df.set_index("Joueur").T
        st.dataframe(table_df)

        st.download_button("⬇️ Télécharger en CSV", table_df.to_csv().encode("utf-8"),
                           file_name="comparatif_clubs.csv", mime="text/csv")

        img_file = dataframe_to_image(table_df, "comparatif_clubs.png")
        with open(img_file, "rb") as f:
            st.download_button("⬇️ Télécharger en PNG", f, file_name="comparatif_clubs.png", mime="image/png")
    else:
        st.info("Sélectionne au moins une statistique pour afficher le radar.")
else:
    st.info("Aucun club sélectionné.")

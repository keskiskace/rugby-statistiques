# 4_🏉_Top10_Flop10_Clubs.py
import streamlit as st
import pandas as pd
import numpy as np
from utils import load_clubs, dataframe_to_image, make_scatter_radar
from utils import compute_composite_ranking
from utils import load_clubs, load_players, compute_club_defense

st.set_page_config(page_title="Top10/Flop10 Clubs", layout="wide")
st.title("🏉 Top10 / Flop10 — Clubs")

# Charger les données clubs
df = load_clubs()
if df is None or df.empty:
    st.warning("Aucune donnée clubs trouvée dans la DB.")
    st.stop()

# Filtres
saisons_disponibles = sorted(df['saison'].dropna().unique(), reverse=True)
default_saison = [saisons_disponibles[0]] if saisons_disponibles else []
selected_saison = st.selectbox("Choisir une saison", saisons_disponibles, index=0)



df_filtered = df[df['saison'] == selected_saison].copy()



# Division
divisions_dispo = sorted(df_filtered['division'].dropna().unique())
choice_division = st.selectbox("Choisir une division", ["Toutes"] + divisions_dispo)
													
if choice_division != "Toutes":
    df_filtered = df_filtered[df_filtered['division'] == choice_division]

# --- garder uniquement la dernière journée PAR DIVISION ---
if "journée" in df_filtered.columns:
    # garantir une valeur pour 'division' pour le groupby
    if 'division' not in df_filtered.columns:
        df_filtered['division'] = 'Unknown'
    else:
        df_filtered['division'] = df_filtered['division'].fillna('Unknown')

    # tenter d'extraire un numéro de journée (ex: "J4", "j5", "4")
    jour_str = df_filtered['journée'].astype(str)
    jour_num = jour_str.str.extract(r'(\d+)')[0]

    if jour_num.notna().any():
        # si on a des numéros, on filtre en utilisant la valeur numérique (plus robuste)
        df_filtered['journee_num'] = pd.to_numeric(jour_num, errors='coerce')
        df_filtered = df_filtered[
            df_filtered.groupby('division')['journee_num'].transform('max') == df_filtered['journee_num']
        ]
        df_filtered = df_filtered.drop(columns=['journee_num'])
    else:
        # sinon fallback : on compare la valeur brute (lexicographique)
        df_filtered = df_filtered[
            df_filtered.groupby('division')['journée'].transform('max') == df_filtered['journée']
        ]
# --- fin ---

# Charger les joueurs et ajouter la stat de plaquage
players_df = load_players()
if players_df is not None and not players_df.empty:
    clubs_def = compute_club_defense(players_df)
    df_filtered = df_filtered.merge(clubs_def, on=["club", "saison"], how="left")

st.subheader(f"📊 Saison {selected_saison}")

# Colonnes numériques
numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ["journée"]
stat_cols = [c for c in numeric_cols if c not in exclude_cols]                       

# Statistique
numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ["journée"]
stat_cols = [c for c in numeric_cols if c not in exclude_cols]

if not stat_cols:
    st.warning("Aucune statistique disponible pour ces filtres.")
    st.stop()

choice_stat = st.selectbox("Choisir une statistique", stat_cols)
# Type (Top ou Flop)
choice_type = st.radio("Choisir le type", ["Top 10", "Flop 10"])    

if not df_filtered.empty and choice_stat:
    if choice_type == "Top 10":
        result = df_filtered.nlargest(10, choice_stat)[["club", "division", choice_stat]]
    else:
        result = df_filtered.nsmallest(10, choice_stat)[["club", "division", choice_stat]]

    st.subheader(f"{choice_type} sur {choice_stat}")
    st.dataframe(result)

    st.download_button("⬇️ Télécharger en CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name=f"{choice_type.lower().replace(' ', '_')}_clubs_{choice_stat}.csv",
                    mime="text/csv")

    img_file = dataframe_to_image(result, f"{choice_type.lower()}_clubs_{choice_stat}.png")
    with open(img_file, "rb") as f:
        st.download_button("⬇️ Télécharger en PNG", f,
                        file_name=f"{choice_type.lower()}_clubs_{choice_stat}.png",
                        mime="image/png")
else:
    st.info("Aucune donnée pour ce filtre.")



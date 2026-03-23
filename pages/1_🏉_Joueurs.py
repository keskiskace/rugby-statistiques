# players_app.py
import streamlit as st
import pandas as pd
import numpy as np
from utils import load_players, compute_extra_players, get_image_safe, download_missing_photos, make_scatter_radar, dataframe_to_image
from utils import compute_composite_ranking
from utils import format_short_name

st.set_page_config(page_title="Comparateur Joueurs", layout="wide")
st.title("🏉 Comparateur — Joueurs (Top14 / ProD2)")

# Chargement
df = load_players()
if df is None or df.empty:
    st.warning("Aucune donnée joueurs trouvée dans la DB.")
    st.stop()

# option pour ne garder que joueurs ayant joué au moins 1 match (comme dans ton app original)
if 'nombre_matchs_joues' in df.columns:
    df_nonzero = df[pd.to_numeric(df['nombre_matchs_joues'], errors='coerce').fillna(0) > 0].copy()
else:
    df_nonzero = df.copy()

# extra players (joueurs types)
extra_df = compute_extra_players(df_nonzero)

# ===== Widgets de reset =====
def reset_players_filters():
    keys = [
        'players_saisons', 'players_divisions', 'players_clubs',
        'players_postes', 'players_names', 'players_types', 'players_stats', "temps_jeu_min"
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


st.sidebar.button("Réinitialiser les filtres Joueurs", on_click=reset_players_filters)

# ===== FILTRES HIERARCHIQUES =====
st.subheader("🎛️ Filtres")

# 1) saison
saisons_disponibles = sorted(df['saison'].dropna().unique(), reverse=True)
default_saisons = [saisons_disponibles[0]] if saisons_disponibles else []
selected_saisons = st.multiselect("1) Choisir saison(s)", saisons_disponibles, default=st.session_state.get('players_saisons', default_saisons), key='players_saisons')

# construire df_players local
if selected_saisons:
    df_players = df[df['saison'].isin(selected_saisons)].copy()
else:
    df_players = df.copy()

# === Appliquer la version abrégée du nom pour l'affichage ===
if "nom" in df_players.columns:
    df_players["short_name"] = df_players["nom"].apply(format_short_name)
else:
    df_players["short_name"] = df_players.get("nom", "")

# prepare display_name
if 'saison' in df_players.columns and len(selected_saisons) > 1:
    df_players['display_name'] = df_players['short_name'] + " (" + df_players['saison'].astype(str) + ")"
else:
    df_players['display_name'] = df_players['short_name']

# 2) Filtre journée (supprimé)
# On applique directement la logique "dernière journée"
if "journée" in df_players.columns:
    df_players = df_players.loc[
        df_players.groupby(["saison", "nom"])["journée"].transform("max") == df_players["journée"]
    ]


# 3) division
if "division" in df_players.columns:
    divisions_dispo = sorted(df_players['division'].dropna().unique().tolist())
    selected_div = st.multiselect("2) Choisir division(s) (optionnel)", divisions_dispo, default=st.session_state.get('players_divisions', []), key='players_divisions')
    if selected_div:
        df_players = df_players[df_players['division'].isin(selected_div)]

# 4) club
if "club" in df_players.columns:
    clubs_dispo = sorted(df_players['club'].dropna().unique().tolist())
    selected_clubs = st.multiselect("3) Choisir club(s) (optionnel)", clubs_dispo, default=st.session_state.get('players_clubs', []), key='players_clubs')
    if selected_clubs:
        df_players = df_players[df_players['club'].isin(selected_clubs)]

# 5) poste
if "poste" in df_players.columns:
    postes_dispo = sorted(df_players['poste'].dropna().unique().tolist())
    selected_postes = st.multiselect("4) Choisir poste(s) (optionnel)", postes_dispo, default=st.session_state.get('players_postes', []), key='players_postes')
    if selected_postes:
        df_players = df_players[df_players['poste'].isin(selected_postes)]

# Temps de jeu minimum (nouveau filtre)
min_temps_options = [0, 10, 20, 30, 40, 50, 60, 70, 80]
min_temps = st.selectbox("Filtrer par temps de jeu minimum (minutes)", min_temps_options, index=0)

if "temps_jeu_min" in df_players.columns:
    df_players = df_players[df_players["temps_jeu_min"] >= min_temps]

# 6) joueurs disponibles (aucune sélection par défaut)
player_options = df_players['display_name'].sort_values().unique().tolist()
if player_options:
    # Si tu as une ancienne valeur dans le session_state et que tu veux l'effacer au chargement,
    # décommente la ligne suivante (optionnel)
    # if 'players_names' in st.session_state: del st.session_state['players_names']

    # Par défaut aucune sélection: default=[]
    selected_names = st.multiselect(
        "Choisir joueur(s)",
        player_options,
        default=[],
        key='players_names'
    )
    selected_players = df_players[df_players['display_name'].isin(selected_names)].copy()
else:
    selected_players = pd.DataFrame()
    st.warning("Aucun joueur ne correspond aux filtres choisis.")


# types (joueurs types)
if not extra_df.empty:
    types_opts = extra_df['nom'].sort_values().unique().tolist()
    selected_types = st.multiselect("Choisir joueur(s) type (moyenne)", types_opts, default=st.session_state.get('players_types', []), key='players_types')
    selected_type_players = extra_df[extra_df['nom'].isin(selected_types)].copy()
else:
    selected_type_players = pd.DataFrame()

# fusion
if not selected_type_players.empty and not selected_players.empty:
    selected_players = pd.concat([selected_players, selected_type_players], ignore_index=True)
elif not selected_type_players.empty and selected_players.empty:
    selected_players = selected_type_players.copy()

# Affichage joueurs sélectionnés
if not selected_players.empty:
    for _, joueur in selected_players.iterrows():
        nom_aff = joueur.get('nom', '')
        st.subheader(nom_aff)
        if "Joueur type" not in str(nom_aff):
            photo_to_show = get_image_safe(joueur)
            st.image(photo_to_show, caption=joueur.get('club', ''), width=150)

            # Affichage des infos sans ratio poids/taille
            st.json({
                "Club": joueur.get('club', 'N/A'),
                "Poste": joueur.get('poste', 'N/A'),
                "Âge": joueur.get('age', 'N/A'),
                "Taille (cm)": joueur.get('taille_cm', 'N/A'),
                "Poids (kg)": joueur.get('poids_kg', 'N/A'),
            })

            # Bouton vers la fiche LNR
            url = joueur.get("url", None)
            if url and isinstance(url, str) and url.startswith("http"):
                st.link_button("🔗 Voir fiche LNR", url)
            url2 = joueur.get("url_allrugby", None)
            if url2 and isinstance(url2, str) and url2.startswith("http"):
                st.link_button("🔗 Voir fiche ALLRUGBY", url2)

        else:
            st.info("📊 Joueur type (moyenne des stats).")
else:
    st.info("Aucun joueur sélectionné.")


# STATISTIQUES / RADAR
df_extended = pd.concat([df_players, extra_df], ignore_index=True) if not extra_df.empty else df_players.copy()

# Colonnes numériques disponibles
numeric_cols = df_extended.select_dtypes(include=[np.number]).columns.tolist()

# Colonnes à exclure (tu peux les remplir toi-même si besoin)
exclude_cols = ["player_id", "id","journée"]

# Statistiques proposées = toutes les numériques sauf exclusions
stat_cols = [c for c in numeric_cols if c not in exclude_cols]

selected_stats = st.multiselect(
    "Choisir les statistiques pour le radar",
    options=stat_cols,
    default=st.session_state.get('players_stats', stat_cols[:5] if len(stat_cols) > 5 else stat_cols),
    key='players_stats'
)

if selected_stats and not selected_players.empty:
    radar_data = []
    for _, joueur in selected_players.iterrows():
        # Utilise le nom abrégé si disponible, sinon le nom complet
        short_nom = joueur.get("short_name") or joueur.get("nom", "")
        entry = {"Joueur": short_nom}
        for stat in selected_stats:
            entry[stat] = joueur.get(stat, np.nan)
        radar_data.append(entry)

    radar_df = pd.DataFrame(radar_data)
    fig = make_scatter_radar(radar_df, selected_stats)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

    st.subheader("📊 Tableau comparatif")
    table_df = radar_df.set_index("Joueur").T
    st.dataframe(table_df)

    st.download_button("⬇️ Télécharger CSV", table_df.to_csv().encode("utf-8"), file_name="comparatif_joueurs.csv", mime="text/csv")

    img_file = dataframe_to_image(table_df, "comparatif_joueurs.png")
    with open(img_file, "rb") as f:
        st.download_button("⬇️ Télécharger PNG", f, file_name="comparatif_joueurs.png", mime="image/png")

    # ====== Classement composite ======
    try:
        classement_df = compute_composite_ranking(radar_df, entity_col="Joueur", stats=selected_stats)
        st.subheader("🏆 Classement composite des joueurs")
        # afficher la colonne entité, les stats choisies, la moyenne et le classement final
        cols_show = ["Joueur"] + selected_stats + ["Moyenne_rang", "Classement_final"]
        # sécurité : ne garder que les colonnes réellement présentes
        cols_show = [c for c in cols_show if c in classement_df.columns]
        st.dataframe(classement_df[cols_show])
# --- AJOUT DES BOUTONS ICI ---
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.download_button(
                "⬇️ CSV Classement", 
                classement_df[cols_show].to_csv(index=False).encode("utf-8"), 
                file_name="classement_composite.csv", 
                mime="text/csv"
            )
        with col_c2:
            img_class_file = dataframe_to_image(classement_df[cols_show], "classement_joueurs.png")
            with open(img_class_file, "rb") as f:
                st.download_button("⬇️ PNG Classement", f, file_name="classement_joueurs.png", mime="image/png")
        # -----------------------------
    except Exception as e:
        st.error(f"Erreur lors du calcul du classement composite : {e}")

else:
    st.info("Sélectionne au moins une statistique ET un joueur pour afficher le radar.")

import streamlit as st
import pandas as pd
import sqlite3
from PIL import Image
from utils_compo import render_compo, compute_compo_stats

st.set_page_config(page_title="Composition", layout="wide")
st.title("🏉 Composition d'équipe")

DB_PATH = "rugbystat.db"

def _escape_sql(val: str) -> str:
    return val.replace("'", "''") if isinstance(val, str) else val

@st.cache_data
def load_players(saison=None, division=None, club=None, poste=None):
    """
    Charge les joueurs (dernière journée dispo par joueur) et applique les filtres optionnels.
    Retourne toutes les colonnes de players + club_nom.
    """
    conn = sqlite3.connect(DB_PATH)
    base_query = """
    SELECT p.*, c.club AS club_nom
    FROM players p
    LEFT JOIN clubs c ON p.club = c.club
    WHERE p.journée = (
        SELECT MAX(journée)
        FROM players p2
        WHERE p2.player_id = p.player_id AND p2.saison = p.saison
    )
    """

    filters = []
    if saison:
        filters.append(f"p.saison = '{_escape_sql(saison)}'")
    if division:
        filters.append(f"p.division = '{_escape_sql(division)}'")
    if club:
        # club filtre sur la valeur dans la table clubs (c.club)
        filters.append(f"c.club = '{_escape_sql(club)}'")
    if poste:
        filters.append(f"p.poste_allrugby = '{_escape_sql(poste)}'")

    if filters:
        query = base_query + " AND " + " AND ".join(filters)
    else:
        query = base_query

    df = pd.read_sql(query, conn)
    conn.close()

    if "player_id" in df.columns:
        df = df.drop_duplicates(subset=["player_id"])
    return df

# --- Filtres en cascade (saison -> division -> club -> poste_allrugby) ---
conn = sqlite3.connect(DB_PATH)

# 1) Saison
saisons = pd.read_sql("SELECT DISTINCT saison FROM players;", conn)["saison"].dropna().tolist()
saison = st.selectbox("📅 Saison", ["Toutes"] + saisons)
if saison == "Toutes":
    saison = None

# 2) Division (dernière journée par joueur)
query_div = """
SELECT DISTINCT division
FROM players p
WHERE p.journée = (
    SELECT MAX(journée) FROM players p2 WHERE p2.player_id = p.player_id AND p2.saison = p.saison
)
"""
if saison:
    query_div += f" AND p.saison = '{_escape_sql(saison)}'"
divisions = pd.read_sql(query_div, conn)["division"].dropna().tolist()
division = st.selectbox("🏆 Division", ["Toutes"] + divisions)
if division == "Toutes":
    division = None

# 3) Club
query_club = """
SELECT DISTINCT c.club
FROM clubs c
JOIN players p ON p.club = c.club
WHERE p.journée = (
    SELECT MAX(journée) FROM players p2 WHERE p2.player_id = p.player_id AND p2.saison = p.saison
)
"""
conds = []
if saison:
    conds.append(f"p.saison = '{_escape_sql(saison)}'")
if division:
    conds.append(f"p.division = '{_escape_sql(division)}'")
if conds:
    query_club += " AND " + " AND ".join(conds)

clubs = pd.read_sql(query_club, conn)["club"].dropna().tolist()
club = st.selectbox("🏉 Club", ["Tous"] + clubs)
if club == "Tous":
    club = None

# 4) Poste (poste_allrugby)
query_poste = """
SELECT DISTINCT p.poste_allrugby
FROM players p
WHERE p.journée = (
    SELECT MAX(journée) FROM players p2 WHERE p2.player_id = p.player_id AND p2.saison = p.saison
)
"""
conds = []
if saison:
    conds.append(f"p.saison = '{_escape_sql(saison)}'")
if division:
    conds.append(f"p.division = '{_escape_sql(division)}'")
if club:
    # club dans players (colonne p.club)
    conds.append(f"p.club = '{_escape_sql(club)}'")
if conds:
    query_poste += " AND " + " AND ".join(conds)

postes = pd.read_sql(query_poste, conn)["poste_allrugby"].dropna().tolist()
poste = st.selectbox("🧢 Poste (filtre global)", ["Tous"] + postes)
if poste == "Tous":
    poste = None

conn.close()

# --- Chargement des joueurs filtrés (pour alimenter les listes) ---
players_df = load_players(saison, division, club, poste)
if players_df.empty:
    st.warning("Aucun joueur trouvé avec ces filtres.")
    st.stop()

# --- Initialisation session_state pour les sélections/verrouillages ---
if "comp_selected" not in st.session_state:
    # mapping poste_str -> {"player_id": ..., "display": ...}
    st.session_state["comp_selected"] = {}
if "comp_locked" not in st.session_state:
    # mapping poste_str -> bool
    st.session_state["comp_locked"] = {}

# Construire affichage "Nom — Club" et mappings à partir du players_df courant
players_display_df = players_df.copy()
club_col = "club_nom" if "club_nom" in players_display_df.columns else ("club" if "club" in players_display_df.columns else None)
if club_col:
    players_display_df["display"] = players_display_df["nom"].astype(str) + " — " + players_display_df[club_col].astype(str)
else:
    players_display_df["display"] = players_display_df["nom"].astype(str)

display_to_id = players_display_df.drop_duplicates(subset=["player_id"]).set_index("display")["player_id"].to_dict()
options_sorted = sorted(display_to_id.keys())

# Image de fond (ton compovierge)
try:
    background = Image.open("data/compovierge.webp")
except Exception:
    background = Image.open("data/terrain.png")

# --- Sidebar : sélection par poste (1..15) avec lock ---
st.sidebar.header("Sélection des joueurs (verrouillable)")

for p in range(1, 24):
    # récupérer sélection actuelle (si existante)
    key_str = str(p)
    cur_entry = st.session_state["comp_selected"].get(key_str)  # ex: {"player_id":"123", "display":"Dupont — FCG"}
    cur_display = cur_entry["display"] if cur_entry else None
    locked = st.session_state["comp_locked"].get(key_str, False)

    # options actuelles : si la display courante n'est plus dans la liste, on l'ajoute en tête
    opts = list(options_sorted)  # copy
    if cur_display and cur_display not in opts:
        opts = [cur_display] + opts
    opts_with_empty = [""] + opts

    # déterminer index par défaut
    try:
        index = opts_with_empty.index(cur_display) if cur_display else 0
    except ValueError:
        index = 0

    # afficher selectbox (désactivé si verrouillé)
    choix_display = st.sidebar.selectbox(f"Poste {p}", opts_with_empty, index=index, key=f"select_{p}", disabled=locked)

    # checkbox lock (séparée) — quand cochée, on garde la sélection et désactive le selectbox
    lock = st.sidebar.checkbox("🔒 Verrouiller", value=locked, key=f"lock_{p}")

    # Mettre à jour état en session
    if choix_display and choix_display != "":
        # si la selection est dans le mapping courant -> récupérer player_id
        if choix_display in display_to_id:
            pid = display_to_id[choix_display]
        else:
            # sinon si c'est la display déjà stockée -> reprendre son id (ex: joueur verrouillé non présent dans la vue actuelle)
            if cur_entry and choix_display == cur_display:
                pid = cur_entry["player_id"]
            else:
                pid = None

        if pid:
            st.session_state["comp_selected"][key_str] = {"player_id": str(pid), "display": choix_display}
    else:
        # choix vide : supprimer si non verrouillé
        if not lock:
            st.session_state["comp_selected"].pop(key_str, None)

    # enregistrer l'état du verrouillage
    st.session_state["comp_locked"][key_str] = lock

# --- Bouton Générer : on récupère toutes les données nécessaires (même pour joueurs verrouillés exclus par filtres) ---
if st.button("📸 Générer la compo"):
    # vérifier qu'il y a 15 sélections
    selected_map = st.session_state["comp_selected"]
    if len(selected_map) < 15:
        st.warning("⚠️ Merci de sélectionner 15 joueurs (1 par poste).")
    else:
        # mapping poste->player_id (str)
        selections = {int(k): v["player_id"] for k, v in selected_map.items()}

        # récupérer les player_id manquants (si certains ne sont pas dans players_df actuel)
        selected_ids = set(selections.values())
        present_ids = set(players_df["player_id"].astype(str).tolist())
        missing_ids = selected_ids - present_ids

        # pour les missing_ids, interroger la DB pour récupérer leur dernière ligne dispo
        extra_rows = []
        if missing_ids:
            conn = sqlite3.connect(DB_PATH)
            for pid in missing_ids:
                q = """
                SELECT p.*, c.club AS club_nom
                FROM players p
                LEFT JOIN clubs c ON p.club = c.club
                WHERE p.player_id = ?
                AND p.journée = (
                    SELECT MAX(journée) FROM players p2 WHERE p2.player_id = p.player_id AND p2.saison = p.saison
                )
                LIMIT 1
                """
                df_single = pd.read_sql(q, conn, params=(pid,))
                if not df_single.empty:
                    extra_rows.append(df_single.iloc[0])
            conn.close()

        # construire players_for_render en concaténant players_df + extra_rows
        if extra_rows:
            extra_df = pd.DataFrame(extra_rows)
            players_for_render = pd.concat([players_df, extra_df], ignore_index=True)
            players_for_render = players_for_render.drop_duplicates(subset=["player_id"])
        else:
            players_for_render = players_df.copy()

        # rendre la compo et calculer stats
        img = render_compo(background, selections, players_for_render)
        st.image(img, caption="Composition générée", use_container_width=True)

        st.subheader("📊 Statistiques moyennes (dernière journée dispo par joueur)")
        stats = compute_compo_stats(selections, players_for_render)
        st.dataframe(stats)

# utils.py
import os
import sqlite3
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go

DB_FILE = "rugbystat.db"
IMAGES_DIR = "images"
FALLBACK_IMAGE = os.path.join(IMAGES_DIR, "no_player.webp")

def load_players(db_file: str = DB_FILE) -> pd.DataFrame:
    with sqlite3.connect(db_file) as con:
        df = pd.read_sql("SELECT * FROM players", con)
    # tentative conversion numérique intelligente
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass
    # ratios pratiques
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
    # === Ratios normalisés pour mise à égalité (80 min) ===
    if "temps_jeu_min" in df.columns:
        stats_80min = [
            "points",
            "essais",
            "joués_au_pied",
            "mêtres_au_pied",
            "courses",
            "mêtres_parcourus",
            "passes",
            "franchissements",
            "offloads",
            "plaquages_cassés",
            "plaquages_réussis",
            "ballons_grattés",
            "interceptions",
            "pénalités_concédées",
            "cartons_jaunes",
            "cartons_oranges",
            "cartons_rouges",
        ]

        for stat in stats_80min:
            if stat in df.columns:
                df[f"ratio_{stat}_80min"] = (
                    pd.to_numeric(df[stat], errors="coerce") /
                    pd.to_numeric(df["temps_jeu_min"], errors="coerce") * 80
                ).replace([np.inf, -np.inf], np.nan).round(2)
    return df

def load_clubs(db_file: str = DB_FILE) -> pd.DataFrame:
    with sqlite3.connect(db_file) as con:
        clubs_df = pd.read_sql("SELECT * FROM clubs", con)
    for col in clubs_df.columns:
        try:
            clubs_df[col] = pd.to_numeric(clubs_df[col], errors='ignore')
        except Exception:
            pass
    return clubs_df

def get_image_safe(player: dict, images_dir: str = IMAGES_DIR, fallback: str = FALLBACK_IMAGE) -> str:
    player_id = player.get("player_id") or player.get("id") or ""
    img_path = os.path.join(images_dir, f"photo_{player_id}.jpg")
    if os.path.exists(img_path):
        return img_path
    photo_field = player.get("photo", "")
    if isinstance(photo_field, str) and photo_field.startswith("http"):
        return photo_field
    return fallback

def download_missing_photos(df: pd.DataFrame, img_dir: str = IMAGES_DIR, timeout: int = 5) -> int:
    os.makedirs(img_dir, exist_ok=True)
    missing_count = 0
    for _, row in df.iterrows():
        player_id = row.get("player_id") or row.get("id")
        url = row.get("photo")
        if not url or pd.isna(url):
            continue
        img_path = os.path.join(img_dir, f"photo_{player_id}.jpg")
        if os.path.exists(img_path):
            continue
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            with open(img_path, "wb") as f:
                f.write(r.content)
            missing_count += 1
        except Exception as e:
            print(f"[ERREUR] {row.get('nom','?')} ({url}) : {e}")
    return missing_count

def dataframe_to_image(df: pd.DataFrame, filename: str = "table.png") -> str:
    n_rows, n_cols = max(1, len(df)), max(1, len(df.columns))
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.2), max(2, n_rows * 0.5)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return filename

def infer_league(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "division" in df.columns:
        df["__league__"] = df["division"].astype(str)
    else:
        df["__league__"] = None
    return df

def compute_extra_players(df_nonzero: pd.DataFrame) -> pd.DataFrame:
    extra_players = []
    df_labeled = infer_league(df_nonzero)
    if not df_labeled.empty and '__league__' in df_labeled.columns:
        top14_mask = df_labeled['__league__'].str.contains("Top14", case=False, na=False)
        prod2_mask = df_labeled['__league__'].str.contains("ProD2", case=False, na=False)
        if top14_mask.any():
            top14_avg = df_labeled[top14_mask].mean(numeric_only=True).round(1)
            extra_players.append({"nom": "Joueur type Top14", "club": "Top14", **top14_avg.to_dict()})
        if prod2_mask.any():
            prod2_avg = df_labeled[prod2_mask].mean(numeric_only=True).round(1)
            extra_players.append({"nom": "Joueur type ProD2", "club": "ProD2", **prod2_avg.to_dict()})
    # postes moyens
    postes_groupes = {
        "Joueur type Avant": ["Pilier gauche", "Pilier droit", "Talonneur", "Talonner", "1ère ligne"],
        "Joueur type 2eme ligne": ["2eme ligne gauche", "2eme ligne droit", "2ème ligne gauche", "2ème ligne droit", "Deuxième ligne"],
        "Joueur type 3eme ligne": ["3eme ligne", "3ème ligne", "3eme ligne centre", "3ème ligne centre"],
        "Joueur type demi de melee": ["Demi de mêlée", "Demi de melee"],
        "Joueur type demi d'ouverture": ["Demi d'ouverture", "Ouverture"],
        "Joueur type ailier": ["Ailier", "Ailiers"],
        "Joueur type centre": ["Centre", "Centres"],
        "Joueur type arriere": ["Arrière", "Arriere"]
    }
    if 'poste' in df_nonzero.columns:
        for nom_type, postes in postes_groupes.items():
            subset = df_nonzero[df_nonzero['poste'].isin(postes)]
            if not subset.empty:
                avg_stats = subset.mean(numeric_only=True).round(1)
                extra_players.append({"nom": nom_type, "club": "Poste moyen", **avg_stats.to_dict()})
    return pd.DataFrame(extra_players) if extra_players else pd.DataFrame()

def make_scatter_radar(radar_df: pd.DataFrame, selected_stats: list):
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd

    fig = go.Figure()
    n_stats = len(selected_stats)
    if n_stats == 0:
        return fig

    angles = np.linspace(0, 2 * np.pi, n_stats, endpoint=False)
    max_val = radar_df[selected_stats].apply(pd.to_numeric, errors="coerce").max().max()
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = 1.0

    n_circles = 5
    step = max_val / n_circles
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#393b79", "#637939"
    ]

    # cercles
    for i in range(1, n_circles + 1):
        r = step * i
        circle_x = [r * np.cos(t) for t in np.linspace(0, 2 * np.pi, 200)]
        circle_y = [r * np.sin(t) for t in np.linspace(0, 2 * np.pi, 200)]
        fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode="lines",
                                 line=dict(color="lightgrey", dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=r, y=0, text=str(round(r, 1)), showarrow=False,
                           font=dict(size=10, color="grey"))

    # axes & labels
    for angle, stat in zip(angles, selected_stats):
        x_axis = [0, max_val * np.cos(angle)]
        y_axis = [0, max_val * np.sin(angle)]
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines",
                                 line=dict(color="lightgrey", dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=max_val * 1.05 * np.cos(angle),
                           y=max_val * 1.05 * np.sin(angle),
                           text=stat, showarrow=False,
                           font=dict(size=12, color="black"))

    # polygones par joueur/club
    for idx, (_, row) in enumerate(radar_df.iterrows()):
        r_values = [row.get(stat, np.nan) for stat in selected_stats]
        r_values = [np.nan if (not pd.notna(v)) else float(v) for v in r_values]
        r_values += [r_values[0]]
        theta = np.append(angles, angles[0])

        x = [0 if (v is np.nan or not np.isfinite(v)) else v * np.cos(t)
             for v, t in zip(r_values, theta)]
        y = [0 if (v is np.nan or not np.isfinite(v)) else v * np.sin(t)
             for v, t in zip(r_values, theta)]

        color = color_palette[idx % len(color_palette)]
        label = row.get("Club") or row.get("Joueur") or f"Série {idx+1}"

        # Seulement les lignes (pas de remplissage)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=color),
            showlegend=False, hoverinfo="skip"
        ))


        # Courbe avec légende + hover
        hover_texts = [
            f"{label}<br>{stat}: "
            f"{'' if (val is np.nan or not np.isfinite(val)) else round(val,1)}"
            for stat, val in zip(selected_stats, r_values[:-1])
        ]
        hover_texts.append(hover_texts[0])

        fig.add_trace(go.Scatter(x=x, y=y, mode="markers+lines",
                                 name=label,  # ✅ affiché dans la légende
                                 line=dict(color=color),
                                 marker=dict(color=color, size=6),
                                 text=hover_texts,
                                 hovertemplate="%{text}<extra></extra>"))

    fig.update_layout(width=800, height=600, hovermode="closest",
                      dragmode="pan",
                      xaxis=dict(showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False),
                      showlegend=True)
    return fig

def compute_composite_ranking(df: pd.DataFrame, entity_col: str, stats: list, lower_is_better: list = None) -> pd.DataFrame:
    """
    Calcule un classement composite basé sur plusieurs statistiques.
    - df : DataFrame avec une colonne identifiant les entités (ex: 'Joueur' ou 'Club')
    - entity_col : nom de la colonne (par ex. 'Joueur' ou 'Club')
    - stats : liste des colonnes statistiques à utiliser
    - lower_is_better : liste des stats où une valeur plus petite est meilleure (ex: 'cartons_rouges')

    Retourne un DataFrame trié avec :
    - rangs individuels par stat
    - Moyenne_rang
    - Classement_final
    """
    if lower_is_better is None:
        lower_is_better = ["british", "classement", "pts_encaissés", "essais_encaissés", "pénalités_encaissées", "pénalités_concédées", "cartons_jaunes", "cartons_oranges", "cartons_rouges"]

    ranking_df = df.set_index(entity_col)[stats].copy()

    # Créer DataFrame pour les rangs
    ranks = pd.DataFrame(index=ranking_df.index)
    for stat in stats:
        asc = False if stat in lower_is_better else True
        ranks[stat] = ranking_df[stat].rank(ascending=not asc, method="min")

    # Moyenne des rangs
    ranks["Moyenne_rang"] = ranks.mean(axis=1)

    # Classement final
    ranks["Classement_final"] = ranks["Moyenne_rang"].rank(ascending=True, method="min")

    # Tri final et reset index
    classement = ranks.sort_values("Classement_final").reset_index()

    return classement

def compute_club_defense(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le % moyen de plaquages réussis par club à partir de la table players.
    - Identifie les joueurs par 'player_id' (ou 'nom' si absent).
    - Ne garde que la ligne correspondant à la dernière 'journée' par joueur+saison.
    - Filtre les joueurs avec nb_match > 0.
    - Retourne la moyenne de 'pct_plaquages' par club+saison.
    """

    if players_df is None or players_df.empty:
        return pd.DataFrame()

    df = players_df.copy()

    # --- identifiant joueur
    id_col = "player_id" if "player_id" in df.columns else "nom"

    # --- conversions utiles
    df["journée"] = pd.to_numeric(df["journée"], errors="coerce")
    df["nb_match"] = pd.to_numeric(df["nb_match"], errors="coerce").fillna(0)
    df["pct_plaquages"] = pd.to_numeric(df["pct_plaquages"], errors="coerce")

    # --- garder uniquement les joueurs ayant joué au moins 1 match
    df = df[df["nb_match"] > 0]

    # --- récupérer la dernière journée par joueur+saison
    df_sorted = df.sort_values(by=["saison", id_col, "journée"])
    last_per_player = df_sorted.groupby(["saison", id_col], as_index=False).last()

    # --- calcul moyenne par club+saison
    club_avg = (
        last_per_player.groupby(["club", "saison"], as_index=False)["pct_plaquages"]
        .mean()
        .rename(columns={"pct_plaquages": "pct_plaquage_moyen"})
    )

    return club_avg

def format_short_name(full_name: str) -> str:
    """
    Abrège les noms pour affichage dans les tableaux/radars :
    - "Romain Ntamack" -> "R Ntamack"
    - "Antoine Dupont" -> "A Dupont"
    - "Brandon Julio Tiute NANSEN" -> "BJT NANSEN"
    """
    if not isinstance(full_name, str) or not full_name.strip():
        return full_name

    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0]  # ex: juste "Ntamack"

    # Détecte si un nom de famille est écrit en majuscules (ex: "NANSEN")
    last_part = parts[-1]
    if last_part.isupper():
        # On considère tout ce qui précède comme prénoms
        initials = "".join([p[0].upper() for p in parts[:-1] if p])
        return f"{initials} {last_part}"

    # Cas normal : "Prénom Nom" ou "Prénom composé Nom"
    first = parts[0]
    last = parts[-1]
    return f"{first[0].upper()} {last.capitalize()}"






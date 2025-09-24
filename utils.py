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
        lower_is_better = []

    ranking_df = df.set_index(entity_col)[stats].copy()

    # Créer DataFrame pour les rangs
    ranks = pd.DataFrame(index=ranking_df.index)
    for stat in stats:
        asc = True if stat in lower_is_better else True
        ranks[stat] = ranking_df[stat].rank(ascending=not asc, method="min")

    # Moyenne des rangs
    ranks["Moyenne_rang"] = ranks.mean(axis=1)

    # Classement final
    ranks["Classement_final"] = ranks["Moyenne_rang"].rank(ascending=True, method="min")

    # Tri final et reset index
    classement = ranks.sort_values("Classement_final").reset_index()

    return classement

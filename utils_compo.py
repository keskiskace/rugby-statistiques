from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import requests
from io import BytesIO
import textwrap
import io
import streamlit as st
import matplotlib.pyplot as plt


def get_positions(image_size=(1200, 1800)):
    width, height = image_size
    positions = {
        1: (width*0.279, height*0.23),  # Pilier gauche
        2: (width*0.509, height*0.23),  # Talonneur
        3: (width*0.74, height*0.23),  # Pilier droit
        4: (width*0.40, height*0.361),  # 2e ligne gauche
        5: (width*0.63, height*0.361),  # 2e ligne droit
        6: (width*0.284, height*0.492),  # 3e ligne aile g.
        7: (width*0.746, height*0.492),  # 3e ligne aile d.
        8: (width*0.515, height*0.492),  # 3e ligne centre
        9: (width*0.4, height*0.623),  # Demi de mêlée
        10: (width*0.63, height*0.623), # Ouverture
        11: (width*0.22, height*0.702), # Ailier gauche
        12: (width*0.401, height*0.754), # Centre gauche
        13: (width*0.63, height*0.754), # Centre droit
        14: (width*0.816, height*0.702), # Ailier droit
        15: (width*0.516, height*0.835), # Arrière
        16: (width*0.061, height*0.925), # finisseur 1
        17: (width*0.174, height*0.925), # finisseur 2
        18: (width*0.286, height*0.925), # finisseur 3
        19: (width*0.397, height*0.925), # finisseur 4
        20: (width*0.618, height*0.925), # finisseur 5
        21: (width*0.731, height*0.925), # finisseur 6
        22: (width*0.843, height*0.925), # finisseur 7
        23: (width*0.949, height*0.925), # finisseur 8
    }
    return positions


def _load_image_from_path_or_url(path_or_url):
    """Retourne PIL.Image ou None si échec."""
    if not path_or_url:
        return None
    try:
        if isinstance(path_or_url, str) and path_or_url.startswith("http"):
            r = requests.get(path_or_url, timeout=6)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGBA")
        else:
            img = Image.open(path_or_url).convert("RGBA")
        return img
    except Exception:
        return None


def render_compo(background_img, selections, players_df, output="compo.png",
                 photo_size=(135, 160), max_name_len=22, name_font_size=15):
    """
    Dessine la compo sur le terrain avec photos + nom sous chaque vignette.
    """
    img = background_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    positions = get_positions(img.size)

    try:
        font = ImageFont.truetype("arial.ttf", name_font_size)
    except Exception:
        font = ImageFont.load_default()

    if "player_id" in players_df.columns:
        players_df = players_df.copy()
        players_df["player_id_str"] = players_df["player_id"].astype(str)

    for poste, player_id in selections.items():
        px, py = positions.get(poste, (None, None))
        if px is None:
            continue

        pid_str = str(player_id)
        player_rows = players_df[players_df["player_id_str"] == pid_str] if "player_id_str" in players_df.columns else players_df[players_df["player_id"] == player_id]

        if not player_rows.empty:
            player = player_rows.iloc[0]
            name = str(player.get("nom", "") or "")
            photo_path = player.get("photo", None)
        else:
            name = f"Inconnu ({pid_str})"
            photo_path = None

        face = _load_image_from_path_or_url(photo_path) if photo_path else None
        if face is not None:
            face = face.resize(photo_size, Image.LANCZOS)
            x0 = int(px - face.width / 2)
            y0 = int(py - face.height / 2)
            try:
                img.paste(face, (x0, y0), face)
            except Exception:
                img.paste(face.convert("RGB"), (x0, y0))
        else:
            w, h = photo_size
            r = max(w, h) // 2
            x0 = int(px - r)
            y0 = int(py - r)
            draw.ellipse((x0, y0, x0 + 2*r, y0 + 2*r), fill="#DDDDDD", outline="black")

        short_name = textwrap.shorten(name, width=max_name_len, placeholder="…")
        bbox = draw.textbbox((0, 0), short_name, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        try:
            photo_h = face.height if face is not None else (r * 2)
        except Exception:
            photo_h = photo_size[1]

        tx = int(px - tw / 2)
        ty = int(y0 + photo_h + 6)
        if ty + th > img.height:
            ty = img.height - th - 4
        draw.text((tx, ty), short_name, fill="black", font=font)

    out_img = img.convert("RGB")
    out_img.save(output)
    return out_img


def compute_compo_stats(selections, players_df):
    ids = list(selections.values())
    sub_df = players_df[players_df["player_id"].isin(ids)].copy()

    if sub_df.empty:
        return pd.DataFrame()

    stats = {}
    stats["Nb JIFF"] = (sub_df["jiff"].str.lower() == "jiff").sum()
    stats["Moy âge"] = sub_df["age"].mean()
    stats["Moy poids"] = sub_df["poids_kg"].mean()
    pack_ids = [selections.get(i) for i in range(1, 9)]
    pack_df = players_df[players_df["player_id"].isin(pack_ids)]
    stats["Poids pack titulaire"] = pack_df["poids_kg"].sum()
    stats["Max taille"] = sub_df["taille_cm"].max()
    stats["Tot nb titulaires"] = sub_df["nb_titulaire"].sum()
    stats["Tot minutes de jeu"] = sub_df["temps_jeu_min"].sum()
    stats["Tot points"] = sub_df["points"].sum()
    stats["Tot essais"] = sub_df["essais"].sum()
    stats["Moy % pénalités"] = sub_df.loc[sub_df["pct_pénalités"] > 0, "pct_pénalités"].mean()
    stats["Moy % transformations"] = sub_df.loc[sub_df["pct_transformations"] > 0, "pct_transformations"].mean()
    stats["Tot drops réussis"] = sub_df["drop_réussis"].sum()
    stats["Moy % plaquages"] = sub_df["pct_plaquages"].mean()
    stats["Tot franchissements"] = sub_df["franchissements"].sum()
    stats["Tot offloads"] = sub_df["offloads"].sum()
    stats["Tot plaquages cassés"] = sub_df["plaquages_cassés"].sum()
    stats["Tot ballons grattés"] = sub_df["ballons_grattés"].sum()
    stats["Tot interceptions"] = sub_df["interceptions"].sum()
    stats["Tot pénalités concédées"] = sub_df["pénalités_concédées"].sum()
    stats["Nb cart jaunes"] = sub_df["cartons_jaunes"].sum()
    stats["Nb cart oranges"] = sub_df["cartons_oranges"].sum()
    stats["Nb cart rouges"] = sub_df["cartons_rouges"].sum()

    return pd.DataFrame([stats]).round(2).reset_index(drop=True)


# 🔽 AJOUT : fonction pour transformer un DataFrame en PNG
def df_to_image(df):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


# 🔽 AJOUT : exemples d’utilisation dans ton app Streamlit
def show_compo_and_stats(background_img, selections, players_df):
    img = render_compo(background_img, selections, players_df)

    st.image(img, caption="Composition", use_column_width=True)

    buf_img = io.BytesIO()
    img.save(buf_img, format="PNG")
    buf_img.seek(0)
    st.download_button("📥 Télécharger la compo en PNG", data=buf_img, file_name="composition.png", mime="image/png")

    stats_df = compute_compo_stats(selections, players_df)
    if not stats_df.empty:
        st.dataframe(stats_df)
        stats_png = df_to_image(stats_df)
        st.download_button("📥 Télécharger les stats en PNG", data=stats_png, file_name="stats_compo.png", mime="image/png")

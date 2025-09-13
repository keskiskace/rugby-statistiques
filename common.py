import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def get_image_safe(player, images_dir="images", fallback="images/no_player.webp"):
    player_id = player.get("player_id") or player.get("id") or ""
    img_path = os.path.join(images_dir, f"photo_{player_id}.jpg")
    if os.path.exists(img_path):
        return img_path
    photo_field = player.get("photo", "")
    if isinstance(photo_field, str) and photo_field.startswith("http"):
        return photo_field
    return fallback

def download_missing_photos(df, img_dir="images", timeout=5):
    os.makedirs(img_dir, exist_ok=True)
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
        except Exception as e:
            print(f"[ERREUR] {row.get('nom','?')} ({url}) : {e}")

def dataframe_to_image(df, filename="table.png"):
    n_rows, n_cols = max(1, len(df)), max(1, len(df.columns))
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.2), max(2, n_rows * 0.5)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return filename

def make_scatter_radar(radar_df, selected_stats):
    fig = go.Figure()
    n_stats = len(selected_stats)
    angles = np.linspace(0, 2 * np.pi, n_stats, endpoint=False)
    max_val = radar_df[selected_stats].apply(pd.to_numeric, errors="coerce").max().max()
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = 1.0
    n_circles = 5
    step = max_val / n_circles
    for i in range(1, n_circles + 1):
        r = step * i
        circle_x = [r * np.cos(t) for t in np.linspace(0, 2 * np.pi, 200)]
        circle_y = [r * np.sin(t) for t in np.linspace(0, 2 * np.pi, 200)]
        fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode="lines",
                                 line=dict(color="lightgrey", dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
    for angle, stat in zip(angles, selected_stats):
        fig.add_annotation(x=max_val * 1.05 * np.cos(angle),
                           y=max_val * 1.05 * np.sin(angle),
                           text=stat, showarrow=False, font=dict(size=12, color="black"))
    for idx, (_, row) in enumerate(radar_df.iterrows()):
        r_values = [row.get(stat, np.nan) for stat in selected_stats]
        r_values = [np.nan if (not pd.notna(v)) else float(v) for v in r_values]
        r_values += [r_values[0]]
        theta = np.append(angles, angles[0])
        x = [0 if (v is np.nan or not np.isfinite(v)) else v * np.cos(t) for v, t in zip(r_values, theta)]
        y = [0 if (v is np.nan or not np.isfinite(v)) else v * np.sin(t) for v, t in zip(r_values, theta)]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=row.get("Joueur", "")))
    fig.update_layout(width=800, height=600, hovermode="closest")
    return fig

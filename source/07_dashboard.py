# -*- coding: utf-8 -*-

"""
07_dashboard.py

Dashboard Streamlit de prévision de consommation électrique.

Deux onglets :
    - Courbes nationales
    - Carte régionale interactive

Deux scénarios météo nationaux :
    - OPÉRATIONNEL : météo de J-1 (déployable à 14h J)
    - MÉTÉO PARFAITE : météo de J+1 (borne supérieure)

Lancement :
    streamlit run 07_dashboard.py
"""

from pathlib import Path
from datetime import date, timedelta
from urllib.request import urlopen
import io
import json
import unicodedata

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Prévision consommation électrique",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# Deux fichiers de prédictions nationales (scénarios météo)
FICHIER_PRED_OPERATIONNEL = RESULTS_DIR / "predictions_operationnel.csv"
FICHIER_PRED_METEO_PARFAITE = RESULTS_DIR / "predictions_meteo_parfaite.csv"

# Fallback : ancien fichier unique (compat ascendante)
FICHIER_PRED_LEGACY = RESULTS_DIR / "predictions.csv"

FICHIER_PREVISIONS_REG = RESULTS_DIR / "previsions_regionales.csv"

URL_ECO2MIX_CONSOLIDE = (
    "https://odre.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/eco2mix-national-cons-def/exports/csv"
)
URL_ECO2MIX_TEMPS_REEL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/eco2mix-national-tr/exports/csv"
)

# Deux sources GeoJSON pour robustesse
URLS_GEOJSON = [
    "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson",
    "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson",
]

REGIONS_METROPOLE = {
    "11": "Île-de-France",
    "24": "Centre-Val de Loire",
    "27": "Bourgogne-Franche-Comté",
    "28": "Normandie",
    "32": "Hauts-de-France",
    "44": "Grand Est",
    "52": "Pays de la Loire",
    "53": "Bretagne",
    "75": "Nouvelle-Aquitaine",
    "76": "Occitanie",
    "84": "Auvergne-Rhône-Alpes",
    "93": "Provence-Alpes-Côte d'Azur",
}


# ============================================================
# 2. UTILITAIRES
# ============================================================

def normalize_code(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 1 and s.isdigit():
        s = "0" + s
    return s


def normaliser_texte(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("-", " ").replace("'", " ")
    s = " ".join(s.split())
    return s


NOMS_NORMALISES = {
    normaliser_texte(v): k for k, v in REGIONS_METROPOLE.items()
}


# ============================================================
# 3. PALETTE
# ============================================================

COULEURS = {
    "beige_fonce":    "#F5EFE6",
    "beige":          "#FAF6F0",
    "rose_clair":     "#F4D9D0",
    "rose":           "#E8B4A8",
    "bordeaux":       "#7B2D3B",
    "bordeaux_clair": "#A85566",
    "texte":          "#4A2C33",
    "gris_doux":      "#B8A9A0",
    "vert_doux":      "#7A9A7E",
    "bleu_gris":      "#5C7A8A",
}

# Couleurs des courbes nationales
COULEUR_MODELE_PARFAITE = COULEURS["bordeaux"]       # trait plein épais
COULEUR_MODELE_OPERATIONNEL = COULEURS["bleu_gris"]  # trait plein médian
COULEUR_SOMME_REGIONS = COULEURS["bordeaux_clair"]   # trait pointillé
COULEUR_RTE = COULEURS["rose"]                       # trait tirets
COULEUR_REEL = COULEURS["vert_doux"]                 # trait plein épais

CARTE_SCALE = [
    [0.00, "#FAF6F0"],
    [0.25, "#F4D9D0"],
    [0.55, "#E8B4A8"],
    [0.80, "#A85566"],
    [1.00, "#7B2D3B"],
]


# ============================================================
# 4. CSS
# ============================================================

def injecter_css():

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {COULEURS['beige']}; }}
        h1, h2, h3 {{
            color: {COULEURS['bordeaux']} !important;
            font-family: 'Georgia', serif;
        }}
        p, label, span, div {{ color: {COULEURS['texte']}; }}
        section[data-testid="stSidebar"] {{
            background-color: {COULEURS['rose_clair']};
        }}
        .stButton > button {{
            background-color: {COULEURS['bordeaux']};
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
        }}
        .stButton > button:hover {{
            background-color: {COULEURS['bordeaux_clair']};
        }}
        .stDateInput input {{
            background-color: {COULEURS['beige_fonce']};
            border: 1px solid {COULEURS['rose']};
            border-radius: 6px;
        }}
        div[data-testid="stMetric"] {{
            background-color: {COULEURS['beige_fonce']};
            border-left: 4px solid {COULEURS['bordeaux']};
            border-radius: 6px;
            padding: 12px;
        }}
        div[data-testid="stMetricValue"] {{
            color: {COULEURS['bordeaux']} !important;
        }}
        .carte-accueil {{
            background-color: {COULEURS['beige_fonce']};
            border: 1px solid {COULEURS['rose']};
            border-radius: 12px;
            padding: 24px;
            margin: 12px 0;
        }}
        .bandeau-info {{
            background-color: {COULEURS['rose_clair']};
            border-left: 5px solid {COULEURS['bordeaux']};
            border-radius: 6px;
            padding: 14px 18px;
            margin: 12px 0;
            font-size: 0.95rem;
        }}
        .bandeau-scenario {{
            background-color: #EEF3F6;
            border-left: 5px solid {COULEURS['bleu_gris']};
            border-radius: 6px;
            padding: 12px 16px;
            margin: 8px 0;
            font-size: 0.88rem;
        }}
        .warning-box {{
            background-color: #FFF3E0;
            border-left: 5px solid #E8A87C;
            border-radius: 6px;
            padding: 12px 16px;
            margin: 8px 0;
            font-size: 0.9rem;
        }}
        hr {{ border-color: {COULEURS['rose_clair']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 5. GÉOJSON ROBUSTE
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def charger_geojson_regions():

    for url in URLS_GEOJSON:

        try:
            with urlopen(url, timeout=30) as r:
                gj = json.loads(r.read().decode())

            features_gardees = []

            for f in gj["features"]:

                props = f.get("properties", {})

                code = None
                for key in ["code", "code_insee", "insee", "region_code"]:
                    if key in props:
                        code = props[key]
                        break

                if code is None or normalize_code(code) not in REGIONS_METROPOLE:
                    for key in ["nom", "name", "libelle"]:
                        if key in props:
                            nom_norm = normaliser_texte(props[key])
                            if nom_norm in NOMS_NORMALISES:
                                code = NOMS_NORMALISES[nom_norm]
                                break

                code = normalize_code(code)

                if code in REGIONS_METROPOLE:
                    f["properties"]["code"] = code
                    features_gardees.append(f)

            print(f"GeoJSON {url.split('/')[-1]} : "
                  f"{len(features_gardees)}/{len(REGIONS_METROPOLE)} régions")

            if len(features_gardees) >= 12:
                return {"type": "FeatureCollection", "features": features_gardees}

        except Exception as e:
            print(f"Échec {url} : {e}")
            continue

    st.error("Impossible de charger un GeoJSON contenant les 12 régions.")
    return None


# ============================================================
# 6. CHARGEMENTS
# ============================================================

def _lire_predictions(chemin, suffixe_col):
    """
    Lit un fichier de prédictions nationales et retourne un
    DataFrame avec timestamp | <suffixe_col>.
    """

    if not chemin.exists():
        return None

    df = pd.read_csv(chemin, sep=";", low_memory=False)

    if "date_cible" not in df.columns or "heure_cible" not in df.columns:
        return None

    df["date_cible"] = pd.to_datetime(df["date_cible"], errors="coerce")
    df["timestamp"] = df["date_cible"] + pd.to_timedelta(df["heure_cible"], unit="h")
    df = df.dropna(subset=["timestamp"])

    if "consommation_prevue" not in df.columns:
        return None

    df = df[["timestamp", "consommation_prevue"]].rename(
        columns={"consommation_prevue": suffixe_col}
    )

    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def charger_predictions_nationales():
    """
    Charge les deux scénarios météo nationaux :
        - predictions_operationnel.csv   -> modele_operationnel
        - predictions_meteo_parfaite.csv -> modele_parfaite

    Fallback : si seul predictions.csv existe (ancien format),
    il est utilisé comme scénario "météo parfaite" (modele_parfaite).
    """

    df_op = _lire_predictions(FICHIER_PRED_OPERATIONNEL, "modele_operationnel")
    df_parf = _lire_predictions(FICHIER_PRED_METEO_PARFAITE, "modele_parfaite")

    # Fallback legacy
    if df_op is None and df_parf is None:
        df_legacy = _lire_predictions(FICHIER_PRED_LEGACY, "modele_parfaite")
        if df_legacy is not None:
            return df_legacy

    # Fusion
    if df_op is not None and df_parf is not None:
        return df_op.merge(df_parf, on="timestamp", how="outer").sort_values(
            "timestamp"
        ).reset_index(drop=True)

    if df_parf is not None:
        return df_parf

    if df_op is not None:
        return df_op

    return None


@st.cache_data(ttl=3600, show_spinner=False)
def detecter_periode_autorisee():

    auj = pd.Timestamp(date.today())

    info = {
        "date_min": auj - pd.Timedelta(days=30),
        "date_max": auj + pd.Timedelta(days=1),
        "source": "valeur par défaut",
    }

    # On cherche la date min parmi les fichiers disponibles
    dates_min = []

    for chemin in [FICHIER_PRED_OPERATIONNEL,
                   FICHIER_PRED_METEO_PARFAITE,
                   FICHIER_PRED_LEGACY]:
        if chemin.exists():
            try:
                df = pd.read_csv(
                    chemin, sep=";", low_memory=False,
                    usecols=["date_cible"],
                )
                df["date_cible"] = pd.to_datetime(
                    df["date_cible"], errors="coerce"
                )
                df = df.dropna(subset=["date_cible"])
                if len(df) > 0:
                    dates_min.append(df["date_cible"].min().normalize())
            except Exception:
                pass

    if dates_min:
        info["date_min"] = min(dates_min)
        info["source"] = "déduite des fichiers de prédictions (test)"

    return info


@st.cache_data(ttl=3600, show_spinner=False)
def charger_previsions_regionales():

    if not FICHIER_PREVISIONS_REG.exists():
        return None

    df = pd.read_csv(FICHIER_PREVISIONS_REG, sep=";", low_memory=False)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["code_region"] = df["code_region"].apply(normalize_code)

    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def charger_eco2mix_national(date_debut, date_fin):

    def appel(url, nom):
        try:
            where = (
                f'date_heure >= "{date_debut}T00:00:00" '
                f'AND date_heure <= "{date_fin}T23:59:59"'
            )
            params = {
                "delimiter": ";", "timezone": "Europe/Paris",
                "use_labels": "false", "where": where,
            }
            r = requests.get(url, params=params, timeout=180)
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content), sep=";", low_memory=False)
        except Exception as e:
            st.warning(f"éCO2mix {nom} : {e}")
            return None

    def extraire(df):
        if df is None or df.empty:
            return None

        if "perimetre" in df.columns:
            df = df[df["perimetre"].astype(str).str.strip() == "France"]

        if df.empty or "date_heure" not in df.columns:
            return None

        df["date_heure_utc"] = pd.to_datetime(df["date_heure"], errors="coerce", utc=True)
        df = df.dropna(subset=["date_heure_utc"])
        df["date_heure"] = df["date_heure_utc"].dt.tz_convert("Europe/Paris")
        df = df[df["date_heure"].dt.minute == 0].copy()

        if "consommation" in df.columns:
            df["reel"] = pd.to_numeric(df["consommation"], errors="coerce")
        else:
            df["reel"] = np.nan

        if "prevision_j1" in df.columns:
            df["rte"] = pd.to_numeric(df["prevision_j1"], errors="coerce")
        else:
            df["rte"] = np.nan

        df["timestamp"] = df["date_heure"].dt.tz_localize(None)

        df = df[["timestamp", "reel", "rte"]].drop_duplicates("timestamp")
        return df.sort_values("timestamp").reset_index(drop=True)

    df_cons = extraire(appel(URL_ECO2MIX_CONSOLIDE, "consolidé"))
    df_tr = extraire(appel(URL_ECO2MIX_TEMPS_REEL, "temps réel"))

    frames = []
    if df_cons is not None and len(df_cons) > 0:
        frames.append(df_cons)
    if df_tr is not None and len(df_tr) > 0:
        if df_cons is not None and len(df_cons) > 0:
            ts = set(df_cons["timestamp"])
            df_tr = df_tr[~df_tr["timestamp"].isin(ts)]
        if len(df_tr) > 0:
            frames.append(df_tr)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


# ============================================================
# 7. GRAPHIQUES
# ============================================================

def graphique_courbes_nationales(df, titre):
    """
    5 courbes possibles :
        - Modèle national météo parfaite (bordeaux épais)
        - Modèle national opérationnel (bleu-gris épais)
        - Somme des modèles régionaux (bordeaux clair pointillé)
        - Prévision RTE J-1 (rose tirets)
        - Consommation réelle éCO2mix (vert doux épais)
    """

    fig = go.Figure()

    # 1. Modèle national — météo parfaite
    if "modele_parfaite" in df.columns and df["modele_parfaite"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["modele_parfaite"], mode="lines",
            name="Modèle national — météo parfaite (J+1)",
            line=dict(color=COULEUR_MODELE_PARFAITE, width=2.5),
        ))

    # 2. Modèle national — opérationnel
    if "modele_operationnel" in df.columns and df["modele_operationnel"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["modele_operationnel"], mode="lines",
            name="Modèle national — opérationnel (météo J-1)",
            line=dict(color=COULEUR_MODELE_OPERATIONNEL, width=2.2),
        ))

    # 3. Somme des modèles régionaux
    if "somme_regions" in df.columns and df["somme_regions"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["somme_regions"], mode="lines",
            name="Somme modèles régionaux",
            line=dict(color=COULEUR_SOMME_REGIONS, width=2, dash="dot"),
        ))

    # 4. Prévision RTE J-1
    if "rte" in df.columns and df["rte"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["rte"], mode="lines",
            name="Prévision RTE J-1",
            line=dict(color=COULEUR_RTE, width=2, dash="dash"),
        ))

    # 5. Consommation réelle
    if "reel" in df.columns and df["reel"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["reel"], mode="lines",
            name="Consommation réelle (éCO2mix)",
            line=dict(color=COULEUR_REEL, width=2.5),
        ))

    fig.update_layout(
        title=dict(text=titre, font=dict(size=16, color=COULEURS["bordeaux"])),
        xaxis_title="Date / Heure",
        yaxis_title="Consommation (MW)",
        plot_bgcolor=COULEURS["beige_fonce"],
        paper_bgcolor=COULEURS["beige"],
        font=dict(color=COULEURS["texte"], family="Georgia"),
        hovermode="x unified",
        legend=dict(
            bgcolor=COULEURS["beige_fonce"],
            bordercolor=COULEURS["rose"],
            borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        xaxis=dict(gridcolor=COULEURS["rose_clair"]),
        yaxis=dict(gridcolor=COULEURS["rose_clair"]),
        height=560,
        margin=dict(l=40, r=40, t=100, b=40),
    )
    return fig


def graphique_courbes_regionale(df_region, nom_region, titre):

    fig = go.Figure()

    if "consommation_prevue" in df_region.columns:
        fig.add_trace(go.Scatter(
            x=df_region["timestamp"], y=df_region["consommation_prevue"],
            mode="lines", name="Prévision modèle régional",
            line=dict(color=COULEUR_MODELE_PARFAITE, width=2.5),
        ))

    if "consommation_reelle" in df_region.columns:
        fig.add_trace(go.Scatter(
            x=df_region["timestamp"], y=df_region["consommation_reelle"],
            mode="lines", name="Consommation réelle (éCO2mix)",
            line=dict(color=COULEUR_REEL, width=2.5),
        ))

    fig.update_layout(
        title=dict(text=titre, font=dict(size=14, color=COULEURS["bordeaux"])),
        xaxis_title="Date / Heure",
        yaxis_title="Consommation (MW)",
        plot_bgcolor=COULEURS["beige_fonce"],
        paper_bgcolor=COULEURS["beige"],
        font=dict(color=COULEURS["texte"], family="Georgia"),
        hovermode="x unified",
        legend=dict(
            bgcolor=COULEURS["beige_fonce"],
            bordercolor=COULEURS["rose"],
            borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        xaxis=dict(gridcolor=COULEURS["rose_clair"]),
        yaxis=dict(gridcolor=COULEURS["rose_clair"]),
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def carte_choroplete(df_snapshot, geojson, colonne, titre, echelle, unite="GW"):

    fig = px.choropleth(
        df_snapshot,
        geojson=geojson,
        locations="code_region",
        featureidkey="properties.code",
        color=colonne,
        color_continuous_scale=echelle,
        labels={colonne: unite, "code_region": "Région"},
        hover_name="nom_region",
        hover_data={colonne: ":.2f", "code_region": False},
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor=COULEURS["beige"],
    )

    fig.update_layout(
        title=dict(text=titre, font=dict(size=14, color=COULEURS["bordeaux"])),
        paper_bgcolor=COULEURS["beige"],
        font=dict(color=COULEURS["texte"], family="Georgia"),
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title=unite, thickness=12, len=0.7),
    )

    return fig


# ============================================================
# 8. PAGE D'ACCUEIL
# ============================================================

def page_accueil(periode):

    st.markdown(
        f"""
        <div class="carte-accueil">
            <h1 style="text-align:center;">⚡ Prévision de consommation électrique</h1>
            <p style="text-align:center; font-size:1.1rem;">
                France métropolitaine — Modèle ensemble RF + XGBoost
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="bandeau-info">
            <b>📌 Période autorisée :</b>
            {periode['date_min'].strftime('%d/%m/%Y')}
            → {periode['date_max'].strftime('%d/%m/%Y')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_g, col_m, col_d = st.columns([1, 2, 1])

    with col_m:

        defaut_debut = max(periode["date_min"].date(), date.today() - timedelta(days=7))
        defaut_fin = min(periode["date_max"].date(), date.today())
        if defaut_debut > defaut_fin:
            defaut_debut = periode["date_min"].date()
            defaut_fin = periode["date_max"].date()

        date_debut = st.date_input(
            "Date de début",
            value=defaut_debut,
            min_value=periode["date_min"].date(),
            max_value=periode["date_max"].date(),
            key="accueil_debut",
        )

        date_fin = st.date_input(
            "Date de fin",
            value=defaut_fin,
            min_value=periode["date_min"].date(),
            max_value=periode["date_max"].date(),
            key="accueil_fin",
        )

        if st.button("Afficher le dashboard", use_container_width=True):
            if date_debut > date_fin:
                st.error("Date de début > date de fin.")
            else:
                st.session_state["date_debut"] = date_debut
                st.session_state["date_fin"] = date_fin
                st.session_state["dashboard_actif"] = True
                st.rerun()


# ============================================================
# 9. ONGLET NATIONAL
# ============================================================

def onglet_national(periode, date_debut, date_fin):

    st.markdown("### 📈 Courbes de consommation nationale")

    # --------------------------------------------------------
    # Bandeau explicatif sur les scénarios
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="bandeau-scenario">
            <b>Deux scénarios météo comparés :</b><br>
            • <b>Opérationnel</b> (bleu) : utilise la météo observée de J-1
            comme proxy. <i>Déployable à 14h le jour J.</i><br>
            • <b>Météo parfaite</b> (bordeaux) : utilise la météo observée
            de J+1. <i>Non déployable</i> — borne supérieure de performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Chargement des sources
    # --------------------------------------------------------

    df_modele = charger_predictions_nationales()
    df_rte = charger_eco2mix_national(date_debut.isoformat(), date_fin.isoformat())
    df_reg = charger_previsions_regionales()

    frames = []

    if df_modele is not None and len(df_modele) > 0:
        frames.append(df_modele)

    if df_rte is not None and len(df_rte) > 0:
        frames.append(df_rte)

    if not frames:
        st.warning("Aucune donnée nationale disponible.")
        return

    df = pd.concat(frames, ignore_index=True)

    # Colonnes attendues
    for c in ["modele_parfaite", "modele_operationnel", "rte", "reel"]:
        if c not in df.columns:
            df[c] = np.nan

    df = df.groupby("timestamp", as_index=False).agg({
        "modele_parfaite": "mean",
        "modele_operationnel": "mean",
        "rte": "mean",
        "reel": "mean",
    })

    # --------------------------------------------------------
    # Somme des modèles régionaux
    # --------------------------------------------------------

    if df_reg is not None and len(df_reg) > 0:

        df_reg_clean = df_reg.dropna(subset=["consommation_prevue"])
        nb_regions_total = df_reg_clean["code_region"].nunique()

        somme_reg = (
            df_reg_clean
            .groupby("timestamp")["consommation_prevue"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "somme_regions", "count": "nb_regions"})
        )

        somme_reg = somme_reg[
            somme_reg["nb_regions"] == nb_regions_total
        ].copy()

        if len(somme_reg) > 0:
            df = df.merge(
                somme_reg[["timestamp", "somme_regions"]],
                on="timestamp",
                how="left",
            )
        else:
            df["somme_regions"] = np.nan

    else:
        df["somme_regions"] = np.nan

    # --------------------------------------------------------
    # Filtre date
    # --------------------------------------------------------

    d_debut = pd.Timestamp(date_debut)
    d_fin = pd.Timestamp(date_fin) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df["timestamp"] >= d_debut) & (df["timestamp"] <= d_fin)]

    if df.empty:
        st.warning("Aucune donnée sur cette période.")
        return

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    reel = df["reel"]

    # 5 cibles, ordre d'affichage
    cibles = [
        ("modele_parfaite", "Modèle (météo parfaite)"),
        ("modele_operationnel", "Modèle (opérationnel)"),
        ("somme_regions", "Somme régions"),
        ("rte", "RTE J-1"),
    ]

    # Ligne 1 : MAE
    st.markdown("**MAE (MW)**")
    cols = st.columns(len(cibles))
    for i, (col_name, label) in enumerate(cibles):
        if col_name not in df.columns:
            continue
        m = df[col_name].notna() & reel.notna()
        if m.sum() == 0:
            continue
        err = df.loc[m, col_name] - reel[m]
        with cols[i]:
            st.metric(label, f"{np.mean(np.abs(err)):.0f}")

    # Ligne 2 : RMSE
    st.markdown("**RMSE (MW)**")
    cols = st.columns(len(cibles))
    for i, (col_name, label) in enumerate(cibles):
        if col_name not in df.columns:
            continue
        m = df[col_name].notna() & reel.notna()
        if m.sum() == 0:
            continue
        err = df.loc[m, col_name] - reel[m]
        with cols[i]:
            st.metric(label, f"{np.sqrt(np.mean(err**2)):.0f}")

    # --------------------------------------------------------
    # Graphique
    # --------------------------------------------------------

    fig = graphique_courbes_nationales(
        df,
        f"Prévision vs Réel — du {date_debut.strftime('%d/%m/%Y')} au {date_fin.strftime('%d/%m/%Y')}",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 10. ONGLET CARTE
# ============================================================

def onglet_carte(periode, date_debut, date_fin):

    st.markdown("### 🗺️ Carte régionale de la consommation")

    df_reg = charger_previsions_regionales()

    if df_reg is None:
        st.warning(
            "Aucune donnée régionale.\n\n"
            "Lance d'abord :\n"
            "- `python 08_consommation_regionale.py`\n"
            "- `python 09_meteo_regionale.py`\n"
            "- `python 11_modelisation_regionale.py`"
        )
        return

    geojson = charger_geojson_regions()
    if geojson is None:
        return

    df = df_reg.copy()
    df["nom_region"] = df["code_region"].map(REGIONS_METROPOLE)

    # Diagnostic
    with st.expander("🔍 Diagnostic régional", expanded=False):
        codes_presents = sorted(df["code_region"].unique())
        codes_attendus = sorted(REGIONS_METROPOLE.keys())
        st.write(f"**Régions attendues ({len(codes_attendus)})** : "
                 f"{', '.join(codes_attendus)}")
        st.write(f"**Régions dans les données ({len(codes_presents)})** : "
                 f"{', '.join(codes_presents)}")
        manquantes = set(codes_attendus) - set(codes_presents)
        if manquantes:
            st.error(f"⚠ Régions manquantes : {sorted(manquantes)}")
        st.write(f"**Période** : {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Filtre date
    d_debut = pd.Timestamp(date_debut)
    d_fin = pd.Timestamp(date_fin) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    df_periode = df[
        (df["timestamp"] >= d_debut) & (df["timestamp"] <= d_fin)
    ].copy()

    if df_periode.empty:
        st.warning("Pas de données régionales sur cette période.")
        return

    # Performance par région
    perf_rows = []

    for code_region in sorted(df_periode["code_region"].unique()):
        sub = df_periode[df_periode["code_region"] == code_region].copy()
        m = (
            sub["consommation_prevue"].notna()
            & sub["consommation_reelle"].notna()
        )
        if m.sum() == 0:
            continue

        err = sub.loc[m, "consommation_prevue"] - sub.loc[m, "consommation_reelle"]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        biais = float(np.mean(err))
        conso_moy = float(sub.loc[m, "consommation_reelle"].mean())
        mae_pct = (mae / conso_moy * 100) if conso_moy > 0 else np.nan

        perf_rows.append({
            "code_region": code_region,
            "nom_region": REGIONS_METROPOLE.get(code_region, code_region),
            "MAE_MW": mae,
            "MAE_pct": mae_pct,
            "RMSE_MW": rmse,
            "biais_MW": biais,
            "n": int(m.sum()),
        })

    df_perf = pd.DataFrame(perf_rows)

    # Choix valeur
    col1, col2 = st.columns([1, 3])
    with col1:
        type_valeur = st.selectbox(
            "Valeur affichée",
            options=["Prévision modèle (régional)", "Consommation réelle"],
            index=0,
        )

    colonne = (
        "consommation_prevue" if type_valeur == "Prévision modèle (régional)"
        else "consommation_reelle"
    )

    # Slider temporel
    timestamps_dispo = sorted(df_periode["timestamp"].unique())

    if len(timestamps_dispo) == 0:
        st.warning("Aucun timestamp disponible.")
        return

    ts_min = pd.Timestamp(timestamps_dispo[0])
    ts_max = pd.Timestamp(timestamps_dispo[-1])
    default_ts = pd.Timestamp(timestamps_dispo[len(timestamps_dispo) // 2])

    ts_selectionne = st.slider(
        "Heure sélectionnée",
        min_value=ts_min.to_pydatetime(),
        max_value=ts_max.to_pydatetime(),
        value=default_ts.to_pydatetime(),
        step=timedelta(hours=1),
        format="DD/MM/YYYY HH:mm",
    )

    ts_selectionne = pd.Timestamp(ts_selectionne)

    df_snap = df_periode[df_periode["timestamp"] == ts_selectionne].copy()

    if df_snap.empty:
        diffs = (df_periode["timestamp"] - ts_selectionne).abs()
        closest = df_periode.loc[diffs.idxmin(), "timestamp"]
        df_snap = df_periode[df_periode["timestamp"] == closest].copy()
        ts_selectionne = closest

    df_snap["valeur_gw"] = df_snap[colonne] / 1000.0

    col_g, col_d = st.columns([1, 1])

    with col_g:

        st.markdown(f"**Snapshot : {ts_selectionne.strftime('%d/%m/%Y %H:%M')}**")

        snapshot = (
            df_snap[["code_region", "nom_region", "valeur_gw"]]
            .dropna(subset=["valeur_gw"])
            .drop_duplicates("code_region")
        )

        if snapshot.empty:
            st.info("Aucune valeur à ce timestamp.")
        else:
            titre = f"{type_valeur} ({ts_selectionne.strftime('%d/%m %H:%M')}) — GW"
            fig_carte = carte_choroplete(
                snapshot, geojson, "valeur_gw", titre,
                CARTE_SCALE, unite="GW"
            )
            st.plotly_chart(fig_carte, use_container_width=True)

            manquants_snap = (
                set(REGIONS_METROPOLE.keys()) - set(snapshot["code_region"])
            )
            if manquants_snap:
                noms = [REGIONS_METROPOLE[c] for c in sorted(manquants_snap)]
                st.markdown(
                    f'<div class="warning-box">'
                    f'⚠ Régions sans données : {", ".join(noms)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Top / flop performance
        if not df_perf.empty:

            st.markdown("---")
            st.markdown(
                f"**🎯 Performance du modèle par région "
                f"({date_debut.strftime('%d/%m/%Y')} → "
                f"{date_fin.strftime('%d/%m/%Y')})**"
            )

            df_perf_tri = df_perf.dropna(subset=["MAE_pct"]).copy()
            df_perf_tri = df_perf_tri.sort_values("MAE_pct")

            meilleures = df_perf_tri.head(5)
            pires = df_perf_tri.tail(5).iloc[::-1]

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**🥇 Modèle le plus précis (MAE %)**")
                for _, r in meilleures.iterrows():
                    st.markdown(
                        f"- **{r['nom_region']}** — "
                        f"MAE {r['MAE_MW']:.0f} MW "
                        f"({r['MAE_pct']:.2f} %)"
                    )

            with c2:
                st.markdown("**🥉 Modèle le moins précis (MAE %)**")
                for _, r in pires.iterrows():
                    st.markdown(
                        f"- **{r['nom_region']}** — "
                        f"MAE {r['MAE_MW']:.0f} MW "
                        f"({r['MAE_pct']:.2f} %)"
                    )

            with st.expander("📊 Tableau complet des performances"):
                df_affichage = df_perf.copy()
                df_affichage = df_affichage.rename(columns={
                    "code_region": "Code",
                    "nom_region": "Région",
                    "MAE_MW": "MAE (MW)",
                    "MAE_pct": "MAE (%)",
                    "RMSE_MW": "RMSE (MW)",
                    "biais_MW": "Biais (MW)",
                    "n": "n_heures",
                })
                df_affichage = df_affichage.sort_values("MAE (%)")
                st.dataframe(df_affichage, use_container_width=True, height=400)

    with col_d:

        st.markdown("**Détail d'une région**")

        regions_dispo = sorted(df_periode["nom_region"].dropna().unique())
        region_choisie = st.selectbox(
            "Région",
            options=regions_dispo,
            index=0,
            key="region_select",
        )

        df_reg_sel = df_periode[
            df_periode["nom_region"] == region_choisie
        ].sort_values("timestamp")

        if df_reg_sel.empty:
            st.info("Pas de données pour cette région.")
        else:
            fig = graphique_courbes_regionale(
                df_reg_sel, region_choisie,
                f"{region_choisie} — du "
                f"{date_debut.strftime('%d/%m')} au "
                f"{date_fin.strftime('%d/%m')}",
            )
            st.plotly_chart(fig, use_container_width=True)

            code_region_sel = df_reg_sel["code_region"].iloc[0]
            perf_sel = df_perf[df_perf["code_region"] == code_region_sel]

            if not perf_sel.empty:
                row = perf_sel.iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("MAE (MW)", f"{row['MAE_MW']:.0f}")
                with c2:
                    st.metric("MAE (%)", f"{row['MAE_pct']:.2f} %")
                with c3:
                    st.metric("Biais (MW)", f"{row['biais_MW']:+.0f}".replace("mW", "MW"))


# ============================================================
# 11. PAGE DASHBOARD
# ============================================================

def page_dashboard(periode):

    st.markdown(
        f"""
        <div class="carte-accueil">
            <h1 style="text-align:center;">⚡ Dashboard Prévision</h1>
            <p style="text-align:center; color:{COULEURS['gris_doux']};">
                Consommation électrique France métropolitaine
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="bandeau-info">
            <b>📌 Période autorisée :</b>
            {periode['date_min'].strftime('%d/%m/%Y')}
            → {periode['date_max'].strftime('%d/%m/%Y')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        date_debut = st.date_input(
            "Date de début",
            value=st.session_state.get("date_debut", periode["date_min"].date()),
            min_value=periode["date_min"].date(),
            max_value=periode["date_max"].date(),
            key="dash_debut",
        )

    with col2:
        date_fin = st.date_input(
            "Date de fin",
            value=st.session_state.get("date_fin", periode["date_max"].date()),
            min_value=periode["date_min"].date(),
            max_value=periode["date_max"].date(),
            key="dash_fin",
        )

    with col3:
        st.markdown("")
        st.markdown("")
        if st.button("🔄 Rafraîchir les données", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if date_debut > date_fin:
        st.error("Date de début > date de fin.")
        return

    if date_debut < periode["date_min"].date():
        date_debut = periode["date_min"].date()
    if date_fin > periode["date_max"].date():
        date_fin = periode["date_max"].date()

    st.markdown("---")

    tab1, tab2 = st.tabs(["📈 Courbes nationales", "🗺️ Carte régionale"])

    with tab1:
        onglet_national(periode, date_debut, date_fin)

    with tab2:
        onglet_carte(periode, date_debut, date_fin)

    st.markdown("---")

    if st.button("↩️ Changer les dates"):
        st.session_state["dashboard_actif"] = False
        st.rerun()


# ============================================================
# 12. PROGRAMME PRINCIPAL
# ============================================================

def main():

    injecter_css()

    if "dashboard_actif" not in st.session_state:
        st.session_state["dashboard_actif"] = False

    periode = detecter_periode_autorisee()

    if st.session_state["dashboard_actif"]:
        page_dashboard(periode)
    else:
        page_accueil(periode)


if __name__ == "__main__":
    main()
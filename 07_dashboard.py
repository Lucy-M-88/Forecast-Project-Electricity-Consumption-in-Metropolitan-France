# -*- coding: utf-8 -*-

"""
07_dashboard.py

Dashboard Streamlit de prévision de consommation électrique.

Affiche sur une même courbe :
    - La prévision du modèle (ensemble RF + XGBoost)
    - La prévision J-1 de RTE
    - La consommation réelle observée (éCO2mix)

RÈGLE MÉTHODOLOGIQUE IMPORTANTE
-------------------------------
Le dashboard ne couvre QUE la période de test (post-entraînement).
Cela évite à l'utilisateur de tester le modèle sur des données
qu'il a vues pendant l'entraînement.

SOURCES DE DONNÉES
------------------
- Réel + prévision RTE : dataset CONSOLIDÉ (eco2mix-national-cons-def)
  qui couvre 2010 → il y a quelques semaines.
- Complément temps réel : dataset TEMPS RÉEL (eco2mix-national-tr)
  qui ne couvre que les dernières semaines.

Le consolidé est PRIORITAIRE (données définitives RTE).
Le temps réel ne sert qu'à combler les jours récents absents
du consolidé.

Lancement :
    streamlit run 07_dashboard.py

Prérequis :
    - Avoir lancé 05_modelisation.py (produit results/predictions.csv)
    - Accès réseau pour éCO2mix

Installation :
    pip install streamlit plotly pandas requests
"""

from pathlib import Path
from datetime import date, timedelta
import io

import numpy as np
import pandas as pd
import requests
import streamlit as st
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
FICHIER_PREDICTIONS = RESULTS_DIR / "predictions.csv"

# API éCO2mix
URL_ECO2MIX_CONSOLIDE = (
    "https://odre.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/eco2mix-national-cons-def/exports/csv"
)

URL_ECO2MIX_TEMPS_REEL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/eco2mix-national-tr/exports/csv"
)


# ============================================================
# 2. PALETTE DE COULEURS
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
}

COULEUR_MODELE = COULEURS["bordeaux"]
COULEUR_RTE = COULEURS["rose"]
COULEUR_REEL = COULEURS["vert_doux"]


# ============================================================
# 3. CSS PERSONNALISÉ
# ============================================================

def injecter_css():

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COULEURS['beige']};
        }}

        h1, h2, h3 {{
            color: {COULEURS['bordeaux']} !important;
            font-family: 'Georgia', serif;
        }}

        p, label, span, div {{
            color: {COULEURS['texte']};
        }}

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
            color: white;
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
            color: {COULEURS['texte']};
            font-size: 0.95rem;
        }}

        hr {{
            border-color: {COULEURS['rose_clair']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 4. DÉTECTION DE LA PÉRIODE AUTORISÉE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def detecter_periode_autorisee():
    """
    Détecte la période minimale autorisée dans le dashboard.

    La borne basse correspond à la première date de prévision
    disponible dans results/predictions.csv (donc à la première
    date du test, post-entraînement).
    """

    aujourd_hui = pd.Timestamp(date.today())

    info = {
        "date_min": aujourd_hui - pd.Timedelta(days=30),
        "date_max": aujourd_hui + pd.Timedelta(days=1),
        "source": "valeur par défaut (predictions.csv absent)",
    }

    if FICHIER_PREDICTIONS.exists():
        try:
            df = pd.read_csv(
                FICHIER_PREDICTIONS, sep=";", low_memory=False,
                usecols=["date_cible"],
            )
            df["date_cible"] = pd.to_datetime(df["date_cible"], errors="coerce")
            df = df.dropna(subset=["date_cible"])

            if len(df) > 0:
                info["date_min"] = df["date_cible"].min().normalize()
                info["source"] = (
                    f"déduite de {FICHIER_PREDICTIONS.name} "
                    f"(période de test)"
                )
        except Exception:
            pass

    return info


# ============================================================
# 5. CHARGEMENT DES PRÉDICTIONS DU MODÈLE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def charger_predictions_modele():
    """
    Charge les prédictions du modèle (results/predictions.csv).
    """

    if not FICHIER_PREDICTIONS.exists():
        return None

    df = pd.read_csv(FICHIER_PREDICTIONS, sep=";", low_memory=False)

    if "date_cible" not in df.columns or "heure_cible" not in df.columns:
        return None

    df["date_cible"] = pd.to_datetime(df["date_cible"], errors="coerce")

    df["timestamp"] = df["date_cible"] + pd.to_timedelta(
        df["heure_cible"], unit="h"
    )

    df = df.dropna(subset=["timestamp"]).copy()

    colonnes = ["timestamp"]
    if "consommation_reelle" in df.columns:
        colonnes.append("consommation_reelle")
    if "consommation_prevue" in df.columns:
        colonnes.append("consommation_prevue")

    df = df[colonnes].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def charger_prevision_operationnelle():
    """
    Charge les éventuels fichiers results/prevision_operationnelle_*.csv.
    """

    fichiers = sorted(RESULTS_DIR.glob("prevision_operationnelle_*.csv"))

    if not fichiers:
        return None

    frames = []

    for f in fichiers:
        try:
            df = pd.read_csv(f, sep=";", low_memory=False)

            nom = f.stem
            date_str = nom.replace("prevision_operationnelle_", "")

            df["date_cible"] = pd.to_datetime(date_str, errors="coerce")
            df["timestamp"] = df["date_cible"] + pd.to_timedelta(
                df["heure_cible"], unit="h"
            )

            if "consommation_prevue" not in df.columns:
                continue

            frames.append(df[["timestamp", "consommation_prevue"]])

        except Exception:
            continue

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# 6. CHARGEMENT éCO2mix (CONSOLIDÉ + TEMPS RÉEL)
# ============================================================

def _appel_eco2mix(url, date_debut, date_fin, nom):
    """
    Appel générique à l'API ODRÉ avec filtre date.
    Retourne un DataFrame brut ou None.
    """

    try:
        where_clause = (
            f'date_heure >= "{date_debut}T00:00:00" '
            f'AND date_heure <= "{date_fin}T23:59:59"'
        )

        params = {
            "delimiter": ";",
            "timezone": "Europe/Paris",
            "use_labels": "false",
            "where": where_clause,
        }

        response = requests.get(url, params=params, timeout=180)
        response.raise_for_status()

        df = pd.read_csv(
            io.BytesIO(response.content),
            sep=";",
            low_memory=False,
        )

        return df

    except Exception as e:
        st.warning(f"éCO2mix {nom} : {type(e).__name__} — {e}")
        return None


def _extraire_reel_rte(df, nom):
    """
    Transforme un DataFrame éCO2mix brut en :
        timestamp | reel | rte
    """

    if df is None or df.empty:
        return None

    # Périmètre France
    if "perimetre" in df.columns:
        df = df[df["perimetre"].astype(str).str.strip() == "France"]

    if df.empty:
        return None

    if "date_heure" not in df.columns:
        return None

    df["date_heure_utc"] = pd.to_datetime(
        df["date_heure"], errors="coerce", utc=True
    )

    df = df.dropna(subset=["date_heure_utc"]).copy()

    df["date_heure"] = df["date_heure_utc"].dt.tz_convert("Europe/Paris")

    # HH:00 uniquement
    df = df[df["date_heure"].dt.minute == 0].copy()

    if df.empty:
        return None

    # Réel
    if "consommation" in df.columns:
        df["reel"] = pd.to_numeric(df["consommation"], errors="coerce")
    else:
        df["reel"] = np.nan

    # Prévision J-1 RTE
    if "prevision_j1" in df.columns:
        df["rte"] = pd.to_numeric(df["prevision_j1"], errors="coerce")
    else:
        df["rte"] = np.nan

    # Timestamp naïf heure locale
    df["timestamp"] = df["date_heure"].dt.tz_localize(None)

    df = df[["timestamp", "reel", "rte"]].copy()

    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


@st.cache_data(ttl=1800, show_spinner=False)
def charger_eco2mix_complet(date_debut: str, date_fin: str):
    """
    Charge les données éCO2mix depuis DEUX sources :

        1. Consolidé (eco2mix-national-cons-def)
           → couvre 2010 → il y a quelques semaines
           → PRIORITAIRE (données définitives RTE)

        2. Temps réel (eco2mix-national-tr)
           → couvre seulement les dernières semaines
           → complément pour les jours récents

    Retourne un DataFrame :
        timestamp | reel | rte
    """

    # --- 1. Consolidé ---
    df_consolide_brut = _appel_eco2mix(
        URL_ECO2MIX_CONSOLIDE,
        date_debut,
        date_fin,
        "consolidé",
    )
    df_consolide = _extraire_reel_rte(df_consolide_brut, "consolidé")

    # --- 2. Temps réel ---
    df_tr_brut = _appel_eco2mix(
        URL_ECO2MIX_TEMPS_REEL,
        date_debut,
        date_fin,
        "temps réel",
    )
    df_tr = _extraire_reel_rte(df_tr_brut, "temps réel")

    # --- 3. Fusion avec priorité au consolidé ---
    frames = []

    if df_consolide is not None and len(df_consolide) > 0:
        frames.append(df_consolide)

    if df_tr is not None and len(df_tr) > 0:

        if df_consolide is not None and len(df_consolide) > 0:
            # On ne garde du temps réel que les timestamps absents
            # du consolidé
            ts_consolide = set(df_consolide["timestamp"])
            df_tr = df_tr[~df_tr["timestamp"].isin(ts_consolide)].copy()

        if len(df_tr) > 0:
            frames.append(df_tr)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# 7. CONSTRUCTION DU DATAFRAME UNIFIÉ
# ============================================================

def construire_df_unifie(df_modele, df_op, df_rte):
    """
    Fusionne les sources en un seul DataFrame :
        timestamp | modele | rte | reel
    """

    frames = []

    if (
        df_modele is not None
        and len(df_modele) > 0
        and "consommation_prevue" in df_modele.columns
    ):
        tmp = df_modele[["timestamp", "consommation_prevue"]].copy()
        tmp = tmp.rename(columns={"consommation_prevue": "modele"})
        frames.append(tmp)

    if (
        df_op is not None
        and len(df_op) > 0
        and "consommation_prevue" in df_op.columns
    ):
        tmp = df_op[["timestamp", "consommation_prevue"]].copy()
        tmp = tmp.rename(columns={"consommation_prevue": "modele"})
        frames.append(tmp)

    if df_rte is not None and len(df_rte) > 0:
        frames.append(df_rte)

    if not frames:
        return pd.DataFrame(
            columns=["timestamp", "modele", "rte", "reel"]
        )

    df = pd.concat(frames, ignore_index=True)

    for col in ["modele", "rte", "reel"]:
        if col not in df.columns:
            df[col] = np.nan

    agg_dict = {col: "mean" for col in ["modele", "rte", "reel"]}

    df = df.groupby("timestamp", as_index=False).agg(agg_dict)

    df = df[["timestamp", "modele", "rte", "reel"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# 8. GRAPHIQUE
# ============================================================

def graphique_courbes(df, titre):

    fig = go.Figure()

    if "modele" in df.columns and df["modele"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["modele"],
            mode="lines",
            name="Prévision modèle (RF + XGB)",
            line=dict(color=COULEUR_MODELE, width=2.5),
        ))

    if "rte" in df.columns and df["rte"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["rte"],
            mode="lines",
            name="Prévision RTE J-1",
            line=dict(color=COULEUR_RTE, width=2, dash="dash"),
        ))

    if "reel" in df.columns and df["reel"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["reel"],
            mode="lines",
            name="Consommation réelle (éCO2mix)",
            line=dict(color=COULEUR_REEL, width=2.5),
        ))

    fig.update_layout(
        title=dict(
            text=titre,
            font=dict(size=18, color=COULEURS["bordeaux"]),
        ),
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
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            gridcolor=COULEURS["rose_clair"],
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=COULEURS["rose_clair"],
            showgrid=True,
        ),
        height=550,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


# ============================================================
# 9. MÉTRIQUES
# ============================================================

def calculer_metriques(df):

    resultats = {}

    if "reel" not in df.columns or df["reel"].notna().sum() == 0:
        return resultats

    reel = df["reel"]

    for nom, col in [("modele", "modele"), ("rte", "rte")]:

        if col not in df.columns:
            continue

        masque = df[col].notna() & reel.notna()

        if masque.sum() == 0:
            continue

        err = df.loc[masque, col] - reel[masque]

        resultats[nom] = {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "Biais": float(np.mean(err)),
            "n": int(masque.sum()),
        }

    return resultats


# ============================================================
# 10. PAGE D'ACCUEIL
# ============================================================

def page_accueil(periode):

    st.markdown(
        f"""
        <div class="carte-accueil">
            <h1 style="text-align:center;">⚡ Prévision de consommation électrique</h1>
            <p style="text-align:center; font-size:1.1rem;">
                France métropolitaine — Modèle ensemble RF + XGBoost
            </p>
            <p style="text-align:center; color:{COULEURS['gris_doux']};">
                Pour commencer, choisissez la plage de dates à afficher.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="bandeau-info">
            <b> Période autorisée</b><br>
            Le dashboard ne couvre que la <b>période de test</b>
            (post-entraînement), soit à partir du
            <b>{periode['date_min'].strftime('%d/%m/%Y')}</b>.<br>
            <span style="font-size:0.85rem; color:{COULEURS['gris_doux']};">
                {periode['source']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_g, col_m, col_d = st.columns([1, 2, 1])

    with col_m:

        st.markdown(
            f"""
            <div class="carte-accueil">
                <h3 style="text-align:center;">📅 Choix de la période</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        defaut_debut = max(
            periode["date_min"].date(),
            date.today() - timedelta(days=7),
        )
        defaut_fin = min(
            periode["date_max"].date(),
            date.today(),
        )

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
                st.error(
                    "La date de début doit être antérieure à la date de fin."
                )
            else:
                st.session_state["date_debut"] = date_debut
                st.session_state["date_fin"] = date_fin
                st.session_state["dashboard_actif"] = True
                st.rerun()


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
            <b> Période autorisée :</b>
            {periode['date_min'].strftime('%d/%m/%Y')}
            → {periode['date_max'].strftime('%d/%m/%Y')}<br>
            <span style="font-size:0.85rem; color:{COULEURS['gris_doux']};">
                Le dashboard ne couvre que la période de test
                (post-entraînement). {periode['source']}
            </span>
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
        st.error("La date de début doit être antérieure à la date de fin.")
        return

    if date_debut < periode["date_min"].date():
        date_debut = periode["date_min"].date()
    if date_fin > periode["date_max"].date():
        date_fin = periode["date_max"].date()

    st.markdown("---")

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    with st.spinner("Chargement des données..."):

        df_modele = charger_predictions_modele()
        df_op = charger_prevision_operationnelle()
        df_rte = charger_eco2mix_complet(
            date_debut.isoformat(),
            date_fin.isoformat(),
        )

    # --------------------------------------------------------
    # Diagnostic
    # --------------------------------------------------------

    with st.expander(" Diagnostic des sources", expanded=False):

        st.write(
            f"**predictions.csv** : "
            f"{'absent' if df_modele is None else str(len(df_modele)) + ' lignes'}"
        )
        st.write(
            f"**prevision_operationnelle** : "
            f"{'absent' if df_op is None else str(len(df_op)) + ' lignes'}"
        )
        st.write(
            f"**éCO2mix (consolidé + temps réel)** : "
            f"{'absent' if df_rte is None else str(len(df_rte)) + ' lignes'}"
        )

        if df_modele is not None and len(df_modele) > 0:
            st.write(
                "Plage predictions.csv : "
                f"{df_modele['timestamp'].min()} → "
                f"{df_modele['timestamp'].max()}"
            )

        if df_rte is not None and len(df_rte) > 0:
            st.write(
                "Plage éCO2mix : "
                f"{df_rte['timestamp'].min()} → "
                f"{df_rte['timestamp'].max()}"
            )
            st.write(
                f"Réel disponible : {df_rte['reel'].notna().sum()} points"
            )
            st.write(
                f"RTE J-1 disponible : {df_rte['rte'].notna().sum()} points"
            )

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    df = construire_df_unifie(df_modele, df_op, df_rte)

    d_debut = pd.Timestamp(date_debut)
    d_fin = (
        pd.Timestamp(date_fin)
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )

    if not df.empty:
        df = df[
            (df["timestamp"] >= d_debut) & (df["timestamp"] <= d_fin)
        ].copy()

    if df.empty:
        st.warning(
            "Aucune donnée disponible sur cette période.\n\n"
            "Essaie une autre plage de dates, ou clique sur "
            "**Rafraîchir les données**."
        )
        return

    # --------------------------------------------------------
    # Vérification des séries
    # --------------------------------------------------------

    series_non_vides = {
        "modele": df["modele"].notna().any() if "modele" in df else False,
        "rte": df["rte"].notna().any() if "rte" in df else False,
        "reel": df["reel"].notna().any() if "reel" in df else False,
    }

    if not any(series_non_vides.values()):
        st.warning("Aucune série n'a de données sur cette période.")
        return

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    st.markdown("###  Indicateurs de performance")

    metriques = calculer_metriques(df)

    if metriques:

        cols = st.columns(len(metriques) * 2)
        idx = 0

        for nom, valeurs in metriques.items():
            label = "Modèle" if nom == "modele" else "RTE"
            with cols[idx]:
                st.metric(f"MAE {label} (MW)", f"{valeurs['MAE']:.0f}")
            idx += 1

        for nom, valeurs in metriques.items():
            label = "Modèle" if nom == "modele" else "RTE"
            with cols[idx]:
                st.metric(f"RMSE {label} (MW)", f"{valeurs['RMSE']:.0f}")
            idx += 1

    else:
        st.info(
            "Pas assez de données communes (réel + prévision) pour "
            "calculer les métriques sur cette période."
        )

    # --------------------------------------------------------
    # Graphique
    # --------------------------------------------------------

    st.markdown("### 📈 Courbes de consommation")

    titre = (
        f"Prévision vs Réel — "
        f"du {date_debut.strftime('%d/%m/%Y')} "
        f"au {date_fin.strftime('%d/%m/%Y')}"
    )

    fig = graphique_courbes(df, titre)

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # Tableau
    # --------------------------------------------------------

    with st.expander(" Voir les données détaillées"):

        df_affichage = df.copy()

        df_affichage = df_affichage.rename(columns={
            "timestamp": "Date/Heure",
            "modele": "Prévision modèle (MW)",
            "rte": "Prévision RTE (MW)",
            "reel": "Consommation réelle (MW)",
        })

        st.dataframe(
            df_affichage,
            use_container_width=True,
            height=400,
        )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Nombre de points", f"{len(df):,}")

    with col_b:
        st.metric(
            "Période",
            f"{date_debut.strftime('%d/%m')} → {date_fin.strftime('%d/%m')}",
        )

    with col_c:
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
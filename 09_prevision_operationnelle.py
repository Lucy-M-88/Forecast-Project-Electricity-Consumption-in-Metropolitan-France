# -*- coding: utf-8 -*-

"""
09_prevision_operationnelle.py

Objectif
--------
Produire une prévision opérationnelle de la consommation électrique
horaire pour la journée J+1, à partir des informations disponibles
à 14h le jour J.

Scénario opérationnel :
    - Météo prévue pour J+1 (via Open-Meteo Forecast API)
    - Météo observée de J-1 et J (via historique local)
    - Historique de consommation J-7 et J-1 an (via éCO2mix)
    - Calendrier (jours fériés, weekends)
    - Modèle : ensemble RF + XGBoost

Sortie :
    results/prevision_operationnelle_YYYY-MM-DD.csv
    results/prevision_operationnelle_YYYY-MM-DD.png

IMPORTANT :
    - Aucune donnée observée après 14h J n'est utilisée comme feature.
    - La météo de J+1 est une vraie prévision (Open-Meteo Forecast).
    - Comparaison avec la prévision J-1 de RTE (benchmark externe).

Prérequis :
    - Avoir lancé 03_creation_dataset.py pour avoir l'historique
    - Avoir lancé 05_modele_xgboost.py ET 08_ensemble_rf_xgboost.py
      pour avoir les modèles .joblib
"""

from pathlib import Path
from datetime import date, timedelta, datetime
import json
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers d'entrée
DATASET_PREVISION = DATA_DIR / "dataset_prevision.csv"
METEO_NATIONALE = DATA_DIR / "meteo_france_nationale_journaliere.csv"
ECO2MIX_HORAIRE = DATA_DIR / "eco2mix_national_horaire_fusionne.csv"

# Fichiers de modèles
MODEL_ENSEMBLE = MODELS_DIR / "modele_ensemble_rf_xgb.joblib"
MODEL_XGBOOST = MODELS_DIR / "modele_xgboost_meteo.joblib"

# API Open-Meteo
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Coordonnées géographiques (Paris, centre de la France)
# Open-Meteo fournit des prévisions pour un point, on utilise
# Paris comme approximation.
PARIS_LAT = 48.85
PARIS_LON = 2.35

# Features attendues par le modèle
FEATURES = [
    "consommation_j7",
    "consommation_j1an",
    "heure",
    "jour_semaine",
    "mois",
    "weekend",
    "est_ferie",
    "temperature_moyenne",
    "temperature_min",
    "temperature_max",
    "amplitude_thermique",
    "HDD_moyen",
    "HDD_min",
    "HDD_max",
    "TM_3j",
    "TM_7j",
    "HDD_3j",
    "HDD_7j",
    "duree_gel_min",
]

# Constantes météo
T_REF_CHAUFFAGE = 18.0


# ============================================================
# 2. JOURS FÉRIÉS (extrait)
# ============================================================
#
# On utilise la même liste que dans 03_creation_dataset.py.
# Pour simplifier ici, on inclut seulement les années
# pertinentes (2024-2026).
# ============================================================

JOURS_FERIES = {
    # 2024
    "2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08",
    "2024-05-09", "2024-05-20", "2024-07-14", "2024-08-15",
    "2024-11-01", "2024-11-11", "2024-12-25",
    # 2025
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-05-08",
    "2025-05-29", "2025-06-09", "2025-07-14", "2025-08-15",
    "2025-11-01", "2025-11-11", "2025-12-25",
    # 2026
    "2026-01-01", "2026-04-06", "2026-05-01", "2026-05-08",
    "2026-05-14", "2026-05-25", "2026-07-14", "2026-08-15",
    "2026-11-01", "2026-11-11", "2026-12-25",
}


def est_ferie(date_cible):
    return int(date_cible.strftime("%Y-%m-%d") in JOURS_FERIES)


# ============================================================
# 3. VÉRIFICATION DES PRÉREQUIS
# ============================================================

def verifier_prerequis():

    print("=" * 70)
    print("VÉRIFICATION DES PRÉREQUIS")
    print("=" * 70)

    manquants = []

    # Vérifier les fichiers de données
    for chemin, nom in [
        (DATASET_PREVISION, "dataset_prevision.csv"),
        (METEO_NATIONALE, "meteo_france_nationale_journaliere.csv"),
        (ECO2MIX_HORAIRE, "eco2mix_national_horaire_fusionne.csv"),
    ]:
        if chemin.exists():
            print(f"  [OK] {nom}")
        else:
            print(f"  [MANQUANT] {nom} ({chemin})")
            manquants.append(nom)

    # Vérifier le modèle (XGBoost ou ensemble)
    if MODEL_ENSEMBLE.exists():
        print(f"  [OK] modele_ensemble_rf_xgb.joblib")
        chemin_modele = MODEL_ENSEMBLE
    elif MODEL_XGBOOST.exists():
        print(f"  [OK] modele_xgboost_meteo.joblib (fallback)")
        chemin_modele = MODEL_XGBOOST
    else:
        print(
            "  [MANQUANT] Aucun modèle trouvé.\n"
            "  Lance d'abord 05_modele_xgboost.py puis "
            "08_ensemble_rf_xgboost.py."
        )
        manquants.append("modèle .joblib")
        chemin_modele = None

    if manquants:
        raise FileNotFoundError(
            "Prérequis manquants : " + ", ".join(manquants)
        )

    return chemin_modele


# ============================================================
# 4. RÉCUPÉRATION DE LA MÉTÉO PRÉVUE (OPEN-METEO)
# ============================================================

def recuperer_meteo_prevue(lat, lon, date_debut, date_fin):
    """
    Récupère la météo horaire prévue via Open-Meteo.
    """

    print(f"\nRécupération de la météo prévue ({date_debut} -> {date_fin})...")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum"
        ),
        "start_date": date_debut,
        "end_date": date_fin,
        "timezone": "Europe/Paris",
    }

    url = OPEN_METEO_URL + "?" + "&".join(
        f"{k}={v}" for k, v in params.items()
    )

    print(f"URL : {url[:80]}...")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())

    except urllib.error.URLError as e:
        print(f"  ERREUR : {e}")
        raise

    # --------------------------------------------------------
    # Extraction des données journalières
    # --------------------------------------------------------

    daily = data["daily"]

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temperature_max": daily["temperature_2m_max"],
        "temperature_min": daily["temperature_2m_min"],
        "precipitations_totales_mm": daily["precipitation_sum"],
    })

    # Température moyenne = (max + min) / 2
    df["temperature_moyenne"] = (
        df["temperature_max"] + df["temperature_min"]
    ) / 2

    print(f"  {len(df)} jours récupérés")

    return df


# ============================================================
# 5. CONSTRUCTION DES FEATURES POUR J+1
# ============================================================

def charger_historique():
    """
    Charge l'historique de consommation horaire et la météo nationale.

    IMPORTANT :
    Le fichier eco2mix_national_horaire.csv contient la colonne
    date_heure_utc (UTC, homogène) et date_heure (locale, fuseaux
    mixtes). On utilise date_heure_utc pour éviter l'erreur de
    fuseaux mixtes, puis on convertit en heure locale.
    """

    print("\nChargement de l'historique de consommation...")

    conso = pd.read_csv(
        ECO2MIX_HORAIRE,
        sep=";",
        low_memory=False,
    )

    # --------------------------------------------------------
    # Conversion robuste via UTC
    # --------------------------------------------------------

    conso["date_heure_utc"] = pd.to_datetime(
        conso["date_heure_utc"],
        errors="coerce",
        utc=True,
    )

    conso = conso.dropna(subset=["date_heure_utc"]).copy()

    # Reconstruire l'heure locale
    conso["date_heure_locale"] = (
        conso["date_heure_utc"]
        .dt.tz_convert("Europe/Paris")
    )

    conso["date_locale"] = conso["date_heure_locale"].dt.date
    conso["heure_locale"] = conso["date_heure_locale"].dt.hour

    print(f"  {len(conso):,} lignes")

    print("\nChargement de la météo nationale...")

    meteo = pd.read_csv(
        METEO_NATIONALE,
        sep=";",
        low_memory=False,
    )

    meteo["date"] = pd.to_datetime(
        meteo["date"],
        errors="coerce",
    )

    print(f"  {len(meteo):,} lignes")

    return conso, meteo

def construire_features_j1(
    date_j,
    date_cible,
    conso_historique,
    meteo_historique,
    meteo_prevue,
):
    """
    Construit les features pour les 24 heures de J+1.

    Utilise :
        - Conso J-7 (24 valeurs horaires)
        - Conso J-1 an (24 valeurs horaires)
        - Calendrier de J+1
        - Météo prévue pour J+1
        - Météo prévue pour J (pour les moyennes glissantes)
        - Météo observée de J-1 et avant (pour les moyennes glissantes)

    IMPORTANT : aucune donnée observée après 14h J n'est utilisée.
    """

    print(f"\nConstruction des features pour {date_cible}...")

    # --------------------------------------------------------
    # Dates clés
    # --------------------------------------------------------

    date_j7 = date_cible - timedelta(days=7)
    date_j1an = date_cible.replace(year=date_cible.year - 1)

    # --------------------------------------------------------
    # Météo de J-1 (observée)
    # --------------------------------------------------------

    meteo_j1 = meteo_historique[
        meteo_historique["date"].dt.date == date_j - timedelta(days=1)
    ]

    # --------------------------------------------------------
    # Météo prévue de J (pour moyennes glissantes)
    # --------------------------------------------------------

    meteo_j_prevu = meteo_prevue[
        meteo_prevue["date"].dt.date == date_j
    ]

    # --------------------------------------------------------
    # Météo prévue de J+1
    # --------------------------------------------------------

    meteo_cible = meteo_prevue[
        meteo_prevue["date"].dt.date == date_cible
    ]

    if len(meteo_cible) == 0:
        raise ValueError(
            f"Aucune météo prévue pour {date_cible}. "
            "L'API Open-Meteo n'a peut-être pas couvert cette date."
        )

    meteo_cible = meteo_cible.iloc[0]

    # --------------------------------------------------------
    # Calcul des indicateurs dérivés pour J+1
    # --------------------------------------------------------

    temp_moy = meteo_cible["temperature_moyenne"]
    temp_min = meteo_cible["temperature_min"]
    temp_max = meteo_cible["temperature_max"]

    HDD_moyen = max(0, T_REF_CHAUFFAGE - temp_moy)
    HDD_min = max(0, T_REF_CHAUFFAGE - temp_min)
    HDD_max = max(0, T_REF_CHAUFFAGE - temp_max)

    # --------------------------------------------------------
    # Moyennes glissantes sur 3j et 7j
    # --------------------------------------------------------
    #
    # On combine :
    #   - Météo observée (jusqu'à J-1)
    #   - Météo prévue de J
    #   - Météo prévue de J+1
    #
    # Pour simplifier, on prend les températures moyennes
    # disponibles dans meteo_historique et meteo_prevue.
    # --------------------------------------------------------

    # Températures disponibles
    temps_dispo = []

    # Observées (J-7 à J-1)
    for delta in range(1, 8):
        d = date_j - timedelta(days=delta)
        sub = meteo_historique[
            meteo_historique["date"].dt.date == d
        ]
        if len(sub) > 0:
            temps_dispo.append(sub.iloc[0]["temperature_moyenne"])

    # Prévisions (J et J+1)
    temps_dispo.append(meteo_j_prevu.iloc[0]["temperature_moyenne"] if len(meteo_j_prevu) > 0 else temp_moy)
    temps_dispo.append(temp_moy)

    # Moyennes
    TM_3j = np.mean(temps_dispo[-3:])
    TM_7j = np.mean(temps_dispo[-7:])

    # HDD associés
    HDD_3j = max(0, T_REF_CHAUFFAGE - TM_3j)
    HDD_7j = max(0, T_REF_CHAUFFAGE - TM_7j)

    # --------------------------------------------------------
    # Construction des 24 lignes
    # --------------------------------------------------------

    lignes = []

    for heure in range(24):

        # Conso J-7 à cette heure
        conso_j7 = conso_historique[
            (conso_historique["date_locale"] == date_j7)
            & (conso_historique["heure_locale"] == heure)
        ]

        # Conso J-1 an à cette heure
        conso_j1an = conso_historique[
            (conso_historique["date_locale"] == date_j1an)
            & (conso_historique["heure_locale"] == heure)
        ]

        if len(conso_j7) == 0:
            conso_j7_val = np.nan
        else:
            conso_j7_val = conso_j7.iloc[0]["consommation_horaire"]

        if len(conso_j1an) == 0:
            conso_j1an_val = np.nan
        else:
            conso_j1an_val = conso_j1an.iloc[0]["consommation_horaire"]

        lignes.append({
            "heure_cible": heure,
            "consommation_j7": conso_j7_val,
            "consommation_j1an": conso_j1an_val,
            "heure": heure,
            "jour_semaine": date_cible.weekday(),
            "mois": date_cible.month,
            "weekend": int(date_cible.weekday() >= 5),
            "est_ferie": est_ferie(date_cible),
            "temperature_moyenne": temp_moy,
            "temperature_min": temp_min,
            "temperature_max": temp_max,
            "amplitude_thermique": temp_max - temp_min,
            "HDD_moyen": HDD_moyen,
            "HDD_min": HDD_min,
            "HDD_max": HDD_max,
            "TM_3j": TM_3j,
            "TM_7j": TM_7j,
            "HDD_3j": HDD_3j,
            "HDD_7j": HDD_7j,
            "duree_gel_min": 0,  # pas dispo via Open-Meteo
        })

    df = pd.DataFrame(lignes)

    print(f"  {len(df)} heures préparées")
    print(f"  Valeurs manquantes : {df.isna().sum().sum()}")

    return df


# ============================================================
# 6. PRÉVISION
# ============================================================

def faire_prevision(modele, features_df):
    """
    Produit les 24 prévisions horaires.
    """

    print("\nPrévision en cours...")

    # Vérifier que les features sont présentes
    features_manquantes = [
        f for f in FEATURES
        if f not in features_df.columns
    ]

    if features_manquantes:
        raise ValueError(
            f"Features manquantes : {features_manquantes}"
        )

    # Vérifier les NaN
    nan_count = features_df[FEATURES].isna().sum().sum()

    if nan_count > 0:
        print(f"  ATTENTION : {nan_count} valeurs manquantes")
        print(features_df[FEATURES].isna().sum())

    X = features_df[FEATURES].fillna(0)

    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------

    if isinstance(modele, dict) and "rf" in modele and "xgb" in modele:
        # Ensemble
        alpha = modele.get("alpha", 0.2)
        pred_rf = modele["rf"].predict(X)
        pred_xgb = modele["xgb"].predict(X)
        predictions = alpha * pred_rf + (1 - alpha) * pred_xgb
    else:
        # XGBoost seul
        predictions = modele.predict(X)

    features_df = features_df.copy()
    features_df["consommation_prevue"] = predictions

    return features_df


# ============================================================
# 7. COMPARAISON AVEC RTE
# ============================================================

def recuperer_prevision_rte(date_cible):
    """
    Récupère la prévision J-1 de RTE pour la date cible.
    """

    print(f"\nRécupération de la prévision RTE pour {date_cible}...")

    url = (
        "https://odre.opendatasoft.com/api/explore/v2.1/"
        "catalog/datasets/eco2mix-national-tr/exports/csv"
    )

    params = {
        "delimiter": ";",
        "timezone": "Europe/Paris",
        "use_labels": "false",
    }

    try:
        import urllib.parse
        import io

        full_url = url + "?" + urllib.parse.urlencode(params)

        with urllib.request.urlopen(full_url, timeout=120) as response:
            content = response.read()

        df = pd.read_csv(io.BytesIO(content), sep=";", low_memory=False)

        df = df[df["perimetre"].astype(str).str.strip() == "France"]

        df["date_heure_utc"] = pd.to_datetime(
            df["date_heure"], errors="coerce", utc=True
        )

        df = df.dropna(subset=["date_heure_utc"])

        df["date_heure"] = df["date_heure_utc"].dt.tz_convert("Europe/Paris")

        df = df[df["date_heure"].dt.minute == 0]

        df["prevision_j1"] = pd.to_numeric(
            df["prevision_j1"], errors="coerce"
        )

        df["date_locale"] = df["date_heure"].dt.date
        df["heure_locale"] = df["date_heure"].dt.hour

        df = df[df["date_locale"] == date_cible]

        df = df[["heure_locale", "prevision_j1"]].copy()

        print(f"  {len(df)} valeurs RTE récupérées")

        return df

    except Exception as e:
        print(f"  ATTENTION : impossible de récupérer RTE : {e}")
        return None


# ============================================================
# 8. GRAPHIQUE
# ============================================================

def graphique(resultats, date_cible, chemin_sortie):
    """
    Graphique de la prévision avec comparaison RTE.
    """

    plt.figure(figsize=(12, 6))

    plt.plot(
        resultats["heure_cible"],
        resultats["consommation_prevue"],
        marker="o",
        label="Prévision (ensemble RF+XGB)",
        linewidth=2,
    )

    if "prevision_rte" in resultats.columns:
        plt.plot(
            resultats["heure_cible"],
            resultats["prevision_rte"],
            marker="s",
            linestyle="--",
            label="Prévision RTE J-1",
            linewidth=2,
        )

    plt.title(
        f"Prévision de consommation électrique pour le {date_cible}\n"
        f"(prévision effectuée à 14h le jour J)"
    )

    plt.xlabel("Heure")
    plt.ylabel("Consommation (MW)")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(chemin_sortie, dpi=150)
    print(f"\nGraphique sauvegardé : {chemin_sortie}")

    plt.close()


# ============================================================
# 9. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("PRÉVISION OPÉRATIONNELLE J+1")
    print("=" * 70)

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    date_j = date.today()
    date_cible = date_j + timedelta(days=1)

    print(f"\nDate J       : {date_j}")
    print(f"Date cible   : {date_cible} (J+1)")

    # --------------------------------------------------------
    # Vérification des prérequis
    # --------------------------------------------------------

    chemin_modele = verifier_prerequis()

    # --------------------------------------------------------
    # Chargement du modèle
    # --------------------------------------------------------

    print("\nChargement du modèle...")

    modele = joblib.load(chemin_modele)

    if isinstance(modele, dict):
        print("  Modèle : ensemble RF + XGBoost")
    else:
        print("  Modèle : XGBoost seul")

    # --------------------------------------------------------
    # Récupération de la météo prévue
    # --------------------------------------------------------

    meteo_prevue = recuperer_meteo_prevue(
        PARIS_LAT,
        PARIS_LON,
        (date_j - timedelta(days=7)).isoformat(),
        date_cible.isoformat(),
    )

    # --------------------------------------------------------
    # Chargement de l'historique
    # --------------------------------------------------------

    conso_historique, meteo_historique = charger_historique()

    # --------------------------------------------------------
    # Construction des features
    # --------------------------------------------------------

    features_df = construire_features_j1(
        date_j,
        date_cible,
        conso_historique,
        meteo_historique,
        meteo_prevue,
    )

    # --------------------------------------------------------
    # Prévision
    # --------------------------------------------------------

    resultats = faire_prevision(modele, features_df)

    # --------------------------------------------------------
    # Comparaison avec RTE
    # --------------------------------------------------------

    rte = recuperer_prevision_rte(date_cible)

    if rte is not None:
        resultats = resultats.merge(
            rte,
            left_on="heure_cible",
            right_on="heure_locale",
            how="left",
        )
        resultats = resultats.rename(
            columns={"prevision_j1": "prevision_rte"}
        )

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRÉVISION POUR J+1")
    print("=" * 70)

    affichage = resultats[[
        "heure_cible", "consommation_prevue"
    ]].copy()

    if "prevision_rte" in resultats.columns:
        affichage["prevision_rte"] = resultats["prevision_rte"]

    print(affichage.to_string(index=False))

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    fichier_csv = RESULTS_DIR / f"prevision_operationnelle_{date_cible}.csv"
    resultats.to_csv(fichier_csv, sep=";", index=False, encoding="utf-8")
    print(f"\nRésultats sauvegardés : {fichier_csv}")

    # --------------------------------------------------------
    # Graphique
    # --------------------------------------------------------

    fichier_png = RESULTS_DIR / f"prevision_operationnelle_{date_cible}.png"
    graphique(resultats, date_cible, fichier_png)

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)

if __name__ == "__main__":
    main()
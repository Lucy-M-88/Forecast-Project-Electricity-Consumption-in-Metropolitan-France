# -*- coding: utf-8 -*-

"""
10_prevision_regionale.py

Prévision de la consommation par région avec le modèle national.

LIMITATION MÉTHODOLOGIQUE
-------------------------
Le modèle a été entraîné sur des données NATIONALES. On l'applique
à des features régionales (consommation J-7 régionale, météo
régionale) pour produire une estimation de la consommation
régionale. C'est une approximation utile pour la visualisation
mais qui n'a pas la précision d'un modèle entraîné par région.

Entrées :
    data/processed/eco2mix_regional_horaire.csv
    data/processed/meteo_regionale_journaliere.csv
    models/modele_ensemble_rf_xgb.joblib
    (ou models/modele_xgboost_meteo.joblib)

Sortie :
    results/previsions_regionales.csv
        timestamp | code_region | nom_region
        | consommation_prevue | consommation_reelle

Lancement :
    python 10_prevision_regionale.py
"""

from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONSO_REGIONALE = DATA_DIR / "eco2mix_regional_horaire.csv"
METEO_REGIONALE = DATA_DIR / "meteo_regionale_journaliere.csv"

MODEL_ENSEMBLE = MODELS_DIR / "modele_ensemble_rf_xgb.joblib"
MODEL_XGBOOST = MODELS_DIR / "modele_xgboost_meteo.joblib"

FICHIER_SORTIE = RESULTS_DIR / "previsions_regionales.csv"

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

JOURS_FERIES = {
    "2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08",
    "2024-05-09", "2024-05-20", "2024-07-14", "2024-08-15",
    "2024-11-01", "2024-11-11", "2024-12-25",
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-05-08",
    "2025-05-29", "2025-06-09", "2025-07-14", "2025-08-15",
    "2025-11-01", "2025-11-11", "2025-12-25",
    "2026-01-01", "2026-04-06", "2026-05-01", "2026-05-08",
    "2026-05-14", "2026-05-25", "2026-07-14", "2026-08-15",
    "2026-11-01", "2026-11-11", "2026-12-25",
    "2027-01-01", "2027-03-29", "2027-05-01", "2027-05-08",
    "2027-05-06", "2027-05-17", "2027-07-14", "2027-08-15",
    "2027-11-01", "2027-11-11", "2027-12-25",
}


def est_ferie(d):
    return int(d.strftime("%Y-%m-%d") in JOURS_FERIES)


# ============================================================
# 2. CHARGEMENT
# ============================================================

def charger_modele():

    if MODEL_ENSEMBLE.exists():
        print(f"Modèle : {MODEL_ENSEMBLE.name}")
        return joblib.load(MODEL_ENSEMBLE)

    if MODEL_XGBOOST.exists():
        print(f"Modèle : {MODEL_XGBOOST.name} (fallback)")
        return joblib.load(MODEL_XGBOOST)

    raise FileNotFoundError(
        "Aucun modèle trouvé. Lance d'abord 05_modelisation.py."
    )


def charger_conso_regionale():

    print(f"\nChargement conso régionale : {CONSO_REGIONALE.name}")

    df = pd.read_csv(CONSO_REGIONALE, sep=";", low_memory=False)

    df["date_heure_utc"] = pd.to_datetime(
        df["date_heure_utc"], errors="coerce", utc=True
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    df["code_region"] = df["code_region"].astype(str).str.zfill(2)
    df["consommation_horaire"] = pd.to_numeric(
        df["consommation_horaire"], errors="coerce"
    )

    df = df.dropna(subset=["date_heure_utc", "consommation_horaire"])

    df = df.sort_values(
        ["code_region", "date_heure_utc"]
    ).reset_index(drop=True)

    print(f"  {len(df):,} lignes, {df['code_region'].nunique()} régions")

    return df


def charger_meteo_regionale():

    print(f"\nChargement météo régionale : {METEO_REGIONALE.name}")

    df = pd.read_csv(METEO_REGIONALE, sep=";", low_memory=False)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["code_region"] = df["code_region"].astype(str).str.zfill(2)

    print(f"  {len(df):,} lignes")

    return df


# ============================================================
# 3. PRÉVISION PAR RÉGION
# ============================================================

def predire_region(modele, conso_region, meteo_region, code_region):

    print(f"  Région {code_region}...", end=" ")

    # --------------------------------------------------------
    # Dictionnaire (date, heure) -> conso
    # --------------------------------------------------------

    conso_region = conso_region.copy()
    conso_region["date_locale"] = [
        d.date() if hasattr(d, "date") else d
        for d in conso_region["date_heure_utc"].dt.tz_convert("Europe/Paris")
    ]
    conso_region["heure_locale"] = [
        d.hour
        for d in conso_region["date_heure_utc"].dt.tz_convert("Europe/Paris")
    ]

    lookup = {}
    for row in conso_region.itertuples(index=False):
        lookup[(row.date_locale, int(row.heure_locale))] = float(
            row.consommation_horaire
        )

    meteo_dict = {}
    for row in meteo_region.itertuples(index=False):
        meteo_dict[row.date] = {
            "temperature_moyenne": row.temperature_moyenne,
            "temperature_min": row.temperature_min,
            "temperature_max": row.temperature_max,
            "amplitude_thermique": row.amplitude_thermique,
            "HDD_moyen": row.HDD_moyen,
            "HDD_min": row.HDD_min,
            "HDD_max": row.HDD_max,
            "TM_3j": row.TM_3j,
            "TM_7j": row.TM_7j,
            "HDD_3j": row.HDD_3j,
            "HDD_7j": row.HDD_7j,
            "duree_gel_min": row.duree_gel_min,
        }

    # --------------------------------------------------------
    # Dates disponibles
    # --------------------------------------------------------

    dates_dispo = sorted({d for d, _ in lookup.keys()})
    heures_dispo = set(h for _, h in lookup.keys())

    # On ne garde que les jours avec 24h complètes
    jours_24h = {}
    for d in dates_dispo:
        h = sum(1 for hh in range(24) if (d, hh) in lookup)
        if h == 24:
            jours_24h[d] = True

    # --------------------------------------------------------
    # Construction des exemples
    # --------------------------------------------------------

    rows = []

    for date_cible in dates_dispo:

        if date_cible not in jours_24h:
            continue

        date_j7 = date_cible - timedelta(days=7)
        try:
            date_j1an = date_cible.replace(year=date_cible.year - 1)
        except ValueError:
            continue

        if date_j7 not in jours_24h or date_j1an not in jours_24h:
            continue

        # Météo
        met = meteo_dict.get(date_cible)
        if met is None:
            continue

        # Toutes les features météo doivent être non-NaN
        if any(pd.isna(v) for v in met.values()):
            continue

        for heure in range(24):

            c7 = lookup.get((date_j7, heure))
            c1 = lookup.get((date_j1an, heure))
            y = lookup.get((date_cible, heure))

            if c7 is None or c1 is None or y is None:
                continue

            js = date_cible.weekday()

            rows.append({
                "timestamp": pd.Timestamp(date_cible) + pd.Timedelta(hours=heure),
                "code_region": code_region,
                "consommation_j7": c7,
                "consommation_j1an": c1,
                "heure": heure,
                "jour_semaine": js,
                "mois": date_cible.month,
                "weekend": int(js >= 5),
                "est_ferie": est_ferie(date_cible),
                "temperature_moyenne": met["temperature_moyenne"],
                "temperature_min": met["temperature_min"],
                "temperature_max": met["temperature_max"],
                "amplitude_thermique": met["amplitude_thermique"],
                "HDD_moyen": met["HDD_moyen"],
                "HDD_min": met["HDD_min"],
                "HDD_max": met["HDD_max"],
                "TM_3j": met["TM_3j"],
                "TM_7j": met["TM_7j"],
                "HDD_3j": met["HDD_3j"],
                "HDD_7j": met["HDD_7j"],
                "duree_gel_min": met["duree_gel_min"],
                "consommation_reelle": y,
            })

    if not rows:
        print("aucun exemple")
        return None

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------

    X = df[FEATURES].fillna(0)

    if isinstance(modele, dict) and "rf" in modele and "xgb" in modele:
        alpha = modele.get("alpha", 0.2)
        pred = (
            alpha * modele["rf"].predict(X)
            + (1 - alpha) * modele["xgb"].predict(X)
        )
    else:
        pred = modele.predict(X)

    df["consommation_prevue"] = pred

    print(f"{len(df):,} lignes")

    return df


# ============================================================
# 4. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PRÉVISIONS RÉGIONALES")
    print("=" * 70)

    modele = charger_modele()
    conso = charger_conso_regionale()
    meteo = charger_meteo_regionale()

    frames = []

    regions = sorted(conso["code_region"].unique())

    print(f"\nPrévision pour {len(regions)} régions :")

    for code_region in regions:

        conso_r = conso[conso["code_region"] == code_region].copy()
        meteo_r = meteo[meteo["code_region"] == code_region].copy()

        if meteo_r.empty:
            print(f"  Région {code_region} : pas de météo, ignorée")
            continue

        # Nom région
        nom_region = conso_r["nom_region"].iloc[0]

        df = predire_region(modele, conso_r, meteo_r, code_region)

        if df is not None and len(df) > 0:
            df["nom_region"] = nom_region
            frames.append(df)

    if not frames:
        raise ValueError("Aucune prévision produite.")

    df_final = pd.concat(frames, ignore_index=True)

    df_final = df_final[[
        "timestamp", "code_region", "nom_region",
        "consommation_prevue", "consommation_reelle",
    ]].copy()

    df_final = df_final.sort_values(
        ["timestamp", "code_region"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    df_final.to_csv(FICHIER_SORTIE, sep=";", index=False, encoding="utf-8")

    print()
    print("=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    print(f"Fichier : {FICHIER_SORTIE}")
    print(f"Lignes  : {len(df_final):,}")
    print(f"Régions : {df_final['code_region'].nunique()}")
    print(
        f"Période : {df_final['timestamp'].min()} -> "
        f"{df_final['timestamp'].max()}"
    )
    print()


if __name__ == "__main__":
    main()
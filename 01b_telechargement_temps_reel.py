# -*- coding: utf-8 -*-

"""
01b_telechargement_temps_reel.py

Objectif
--------
Télécharger les données éCO2mix "En-cours mensuel temps réel"
pour combler le délai de publication des données consolidées.

Le dataset temps réel (eco2mix-national-tr) contient les données
des dernières semaines, disponibles ~24h après leur production.

Entrée :
    Aucune

Sortie :
    data/raw/eco2mix_national_temps_reel.csv
    data/raw/telechargement_temps_reel_metadata.txt

IMPORTANT :
    - Les données temps réel peuvent être moins fiables que les
      données consolidées (valeurs provisoires).
    - Elles sont utilisées uniquement pour combler le trou entre
      la dernière date consolidée et aujourd'hui.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# Identifiant du dataset temps réel.
DATASET_ID_TR = "eco2mix-national-tr"

# API OpenDataSoft / ODRÉ.
API_BASE_URL = (
    "https://odre.opendatasoft.com/"
    "api/explore/v2.1/catalog/datasets/"
)

DOWNLOAD_URL = (
    f"{API_BASE_URL}"
    f"{DATASET_ID_TR}"
    "/exports/csv"
)

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_FILE_TR = RAW_DIR / "eco2mix_national_temps_reel.csv"

METADATA_FILE_TR = RAW_DIR / "telechargement_temps_reel_metadata.txt"

REQUEST_TIMEOUT = 180


# ============================================================================
# 2. LOGS
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# 3. TELECHARGEMENT
# ============================================================================

def telecharger_temps_reel() -> pd.DataFrame:
    """
    Télécharge les données temps réel éCO2mix.
    """

    logger.info("Début du téléchargement temps réel éCO2mix.")
    logger.info("Dataset : %s", DATASET_ID_TR)
    logger.info("URL : %s", DOWNLOAD_URL)

    params = {
        "delimiter": ";",
        "timezone": "Europe/Paris",
        "use_labels": "false",
    }

    try:
        response = requests.get(
            DOWNLOAD_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

    except requests.RequestException as exc:
        logger.error("Erreur lors du téléchargement : %s", exc)
        raise

    logger.info(
        "Téléchargement terminé : %.2f Mo",
        len(response.content) / 1024**2,
    )

    df = pd.read_csv(
        io.BytesIO(response.content),
        sep=";",
        low_memory=False,
    )

    logger.info("Nombre de lignes téléchargées : %d", len(df))
    logger.info("Nombre de colonnes : %d", len(df.columns))

    return df


# ============================================================================
# 4. VERIFICATION
# ============================================================================

def verifier_schema(df: pd.DataFrame) -> None:
    """
    Vérifie la présence des colonnes minimales.
    """

    colonnes_requises = {
        "perimetre",
        "date_heure",
        "consommation",
    }

    colonnes_presentes = set(df.columns)
    manquantes = colonnes_requises - colonnes_presentes

    if manquantes:
        raise ValueError(
            "Colonnes manquantes : "
            f"{sorted(manquantes)}\n"
            f"Colonnes disponibles : {sorted(colonnes_presentes)}"
        )

    logger.info("Vérification du schéma : OK.")


def verifier_perimetre(df: pd.DataFrame) -> None:
    """
    Vérifie que le périmètre est 'France'.
    """

    perimetres = (
        df["perimetre"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    logger.info("Périmètres présents : %s", list(perimetres))

    if len(perimetres) != 1 or perimetres[0].lower() != "france":
        raise ValueError(
            f"Périmètre inattendu : {list(perimetres)}"
        )

    logger.info("Périmètre national France : OK.")


# ============================================================================
# 5. PREPARATION MINIMALE
# ============================================================================

def preparer_donnees_brutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les données brutes pour la sauvegarde.
    """

    # --------------------------------------------------------
    # Colonnes utiles
    # --------------------------------------------------------

    colonnes_utiles = [
        c for c in [
            "perimetre",
            "date_heure",
            "consommation",
            "nature",
        ]
        if c in df.columns
    ]

    df = df[colonnes_utiles].copy()

    # --------------------------------------------------------
    # Conversion dates
    # --------------------------------------------------------

    df["date_heure"] = pd.to_datetime(
        df["date_heure"],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # Conversion consommation
    # --------------------------------------------------------

    df["consommation"] = pd.to_numeric(
        df["consommation"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Suppression des dates invalides
    # --------------------------------------------------------

    df = df.dropna(subset=["date_heure"])

    # --------------------------------------------------------
    # Tri
    # --------------------------------------------------------

    df = df.sort_values("date_heure")

    # --------------------------------------------------------
    # Doublons
    # --------------------------------------------------------

    doublons = df["date_heure"].duplicated().sum()

    if doublons > 0:
        logger.warning("%d doublons temporels détectés.", doublons)
        df = df.drop_duplicates(subset=["date_heure"], keep="first")

    df = df.reset_index(drop=True)

    return df


# ============================================================================
# 6. RESUME
# ============================================================================

def afficher_resume(df: pd.DataFrame) -> None:

    print()
    print("=" * 70)
    print("RESUME DU TELECHARGEMENT TEMPS REEL")
    print("=" * 70)

    print(f"Nombre de lignes : {len(df):,}")
    print(f"Première date    : {df['date_heure'].min()}")
    print(f"Dernière date    : {df['date_heure'].max()}")
    print(
        f"Valeurs manquantes consommation : "
        f"{df['consommation'].isna().sum():,}"
    )
    print(f"Nombre de dates uniques : {df['date_heure'].nunique():,}")

    print("=" * 70)
    print()


# ============================================================================
# 7. SAUVEGARDE
# ============================================================================

def sauvegarder_donnees(df: pd.DataFrame) -> None:

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Sauvegarde : %s", RAW_FILE_TR)

    df.to_csv(
        RAW_FILE_TR,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    logger.info("Sauvegarde terminée.")


def sauvegarder_metadata(df: pd.DataFrame) -> None:

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_FILE_TR, "w", encoding="utf-8") as fichier:

        fichier.write("METADONNEES - TEMPS REEL éCO2mix\n")
        fichier.write("==================================\n\n")

        fichier.write(f"Dataset ODRÉ : {DATASET_ID_TR}\n")
        fichier.write(f"URL : {DOWNLOAD_URL}\n")
        fichier.write("Producteur : RTE\n")
        fichier.write("Périmètre : France\n")
        fichier.write("Unité de consommation : MW\n")
        fichier.write(f"Nombre de lignes : {len(df)}\n")
        fichier.write(f"Première date : {df['date_heure'].min()}\n")
        fichier.write(f"Dernière date : {df['date_heure'].max()}\n")

        fichier.write("\nIMPORTANT :\n")
        fichier.write(
            "Données temps réel, valeurs provisoires.\n"
        )
        fichier.write(
            "Utilisées uniquement pour combler le délai de "
            "publication des données consolidées.\n"
        )


# ============================================================================
# 8. MAIN
# ============================================================================

def main():

    print()
    print("=" * 70)
    print("TELECHARGEMENT TEMPS REEL éCO2mix")
    print("=" * 70)
    print()

    df = telecharger_temps_reel()

    verifier_schema(df)
    verifier_perimetre(df)

    df = preparer_donnees_brutes(df)

    afficher_resume(df)

    sauvegarder_donnees(df)
    sauvegarder_metadata(df)

    print()
    print("Téléchargement temps réel terminé.")
    print(f"Données : {RAW_FILE_TR}")
    print()


if __name__ == "__main__":
    main()
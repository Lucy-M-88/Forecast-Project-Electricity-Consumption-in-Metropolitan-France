# -*- coding: utf-8 -*-

"""
01_telechargement.py

Projet de prévision de la consommation électrique
France métropolitaine.

Rôle de ce fichier :
    1. Télécharger les données éCO2mix consolidées (dataset
       eco2mix-national-cons-def) depuis l'API ODRÉ.
    2. Télécharger les données éCO2mix temps réel (dataset
       eco2mix-national-tr) pour combler le délai de publication
       des données consolidées.
    3. Vérifier les schémas et périmètres.
    4. Sauvegarder les données brutes localement.

IMPORTANT :
    Ce fichier ne construit AUCUNE variable de machine learning.
    Il ne crée aucun retard J-7 ou J-1 an.
    Il ne fait aucun entraînement.
    Il ne regarde jamais la consommation future pour fabriquer
    une variable explicative.

La construction du jeu d'apprentissage sera réalisée dans :
    03_creation_dataset.py

Installation :
    pip install pandas requests
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

# Identifiants des datasets ODRÉ.
DATASET_ID_CONSOLIDE = "eco2mix-national-cons-def"
DATASET_ID_TEMPS_REEL = "eco2mix-national-tr"

# API OpenDataSoft / ODRÉ.
API_BASE_URL = (
    "https://odre.opendatasoft.com/"
    "api/explore/v2.1/catalog/datasets/"
)

# URL de téléchargement CSV.
URL_CONSOLIDE = (
    f"{API_BASE_URL}"
    f"{DATASET_ID_CONSOLIDE}"
    "/exports/csv"
)

URL_TEMPS_REEL = (
    f"{API_BASE_URL}"
    f"{DATASET_ID_TEMPS_REEL}"
    "/exports/csv"
)

# Répertoire racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parent

# Répertoire où seront stockées les données.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# Fichiers de sortie.
RAW_FILE_CONSOLIDE = RAW_DIR / "eco2mix_national_brut.csv"
RAW_FILE_TEMPS_REEL = RAW_DIR / "eco2mix_national_temps_reel.csv"

METADATA_FILE_CONSOLIDE = RAW_DIR / "telechargement_metadata.txt"
METADATA_FILE_TEMPS_REEL = RAW_DIR / "telechargement_temps_reel_metadata.txt"

# Timeout réseau.
REQUEST_TIMEOUT = 180

# Paramètres d'export communs.
EXPORT_PARAMS = {
    "delimiter": ";",
    "timezone": "Europe/Paris",
    "use_labels": "false",
}


# ============================================================================
# 2. LOGS
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# 3. TELECHARGEMENT GÉNÉRIQUE
# ============================================================================

def telecharger_dataset(url: str, nom: str) -> pd.DataFrame:
    """
    Télécharge un dataset ODRÉ depuis son URL d'export CSV.

    Parameters
    ----------
    url : str
        URL d'export CSV du dataset.
    nom : str
        Nom du dataset (pour les logs).

    Returns
    -------
    pd.DataFrame
    """

    logger.info("Début du téléchargement : %s", nom)
    logger.info("URL : %s", url)

    try:
        response = requests.get(
            url,
            params=EXPORT_PARAMS,
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

    logger.info("Nombre de lignes : %d", len(df))
    logger.info("Nombre de colonnes : %d", len(df.columns))

    return df


# ============================================================================
# 4. VÉRIFICATIONS GÉNÉRIQUES
# ============================================================================

def verifier_schema(df: pd.DataFrame, colonnes_requises: set) -> None:
    """
    Vérifie la présence des colonnes indispensables.
    """

    colonnes_presentes = set(df.columns)
    colonnes_manquantes = colonnes_requises - colonnes_presentes

    if colonnes_manquantes:
        raise ValueError(
            "Schéma inattendu.\n"
            f"Colonnes manquantes : {sorted(colonnes_manquantes)}\n"
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

    On garde uniquement les colonnes utiles et on nettoie
    les dates et la consommation.
    """

    colonnes_utiles = [
        c for c in [
            "perimetre",
            "nature",
            "date_heure",
            "consommation",
        ]
        if c in df.columns
    ]

    df = df[colonnes_utiles].copy()

    # Conversion des dates en UTC
    df["date_heure"] = pd.to_datetime(
        df["date_heure"],
        errors="coerce",
        utc=True,
    )

    # Conversion de la consommation
    df["consommation"] = pd.to_numeric(
        df["consommation"],
        errors="coerce",
    )

    # Suppression des dates invalides
    df = df.dropna(subset=["date_heure"])

    # Tri chronologique
    df = df.sort_values("date_heure")

    # Doublons
    doublons = df["date_heure"].duplicated().sum()

    if doublons > 0:
        logger.warning("%d doublons temporels détectés.", doublons)
        df = df.drop_duplicates(
            subset=["date_heure"],
            keep="first",
        )

    df = df.reset_index(drop=True)

    return df


# ============================================================================
# 6. SAUVEGARDE
# ============================================================================

def sauvegarder_donnees(df: pd.DataFrame, chemin: Path) -> None:

    chemin.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Sauvegarde : %s", chemin)

    df.to_csv(
        chemin,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    logger.info("Sauvegarde terminée.")


def sauvegarder_metadata(
    df: pd.DataFrame,
    chemin: Path,
    dataset_id: str,
    url: str,
    nature: str,
) -> None:

    chemin.parent.mkdir(parents=True, exist_ok=True)

    with open(chemin, "w", encoding="utf-8") as f:

        f.write(f"METADONNEES - {nature}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Dataset ODRÉ : {dataset_id}\n")
        f.write(f"URL : {url}\n")
        f.write("Producteur : RTE\n")
        f.write("Périmètre : France\n")
        f.write("Unité de consommation : MW\n")
        f.write(f"Nombre de lignes : {len(df)}\n")
        f.write(f"Première date : {df['date_heure'].min()}\n")
        f.write(f"Dernière date : {df['date_heure'].max()}\n")
        f.write(f"Valeurs manquantes consommation : {df['consommation'].isna().sum()}\n")

        f.write("\nColonnes conservées :\n")
        for colonne in df.columns:
            f.write(f" - {colonne}\n")

        f.write("\nIMPORTANT :\n")
        f.write(
            "Ce fichier constitue uniquement l'étape de "
            "téléchargement.\n"
        )
        f.write(
            "Aucune variable de machine learning n'a été construite ici.\n"
        )


# ============================================================================
# 7. FONCTION PRINCIPALE
# ============================================================================

def main():

    print()
    print("=" * 70)
    print("TELECHARGEMENT DES DONNEES éCO2mix")
    print("Projet de prévision - France métropolitaine")
    print("=" * 70)
    print()

    # ========================================================
    # PARTIE 1 : DONNÉES CONSOLIDÉES
    # ========================================================

    print("=" * 70)
    print("PARTIE 1 : DONNÉES CONSOLIDÉES")
    print("=" * 70)
    print()

    df_consolide = telecharger_dataset(
        URL_CONSOLIDE,
        "eco2mix-national-cons-def",
    )

    verifier_schema(
        df_consolide,
        {"perimetre", "date_heure", "consommation", "nature"},
    )

    verifier_perimetre(df_consolide)

    df_consolide = preparer_donnees_brutes(df_consolide)

    sauvegarder_donnees(df_consolide, RAW_FILE_CONSOLIDE)

    sauvegarder_metadata(
        df_consolide,
        METADATA_FILE_CONSOLIDE,
        DATASET_ID_CONSOLIDE,
        URL_CONSOLIDE,
        "CONSOMMATION CONSOLIDÉE éCO2mix",
    )

    print()
    print(f"Données consolidées : {RAW_FILE_CONSOLIDE}")
    print(f"Métadonnées : {METADATA_FILE_CONSOLIDE}")
    print()

    # ========================================================
    # PARTIE 2 : DONNÉES TEMPS RÉEL
    # ========================================================

    print("=" * 70)
    print("PARTIE 2 : DONNÉES TEMPS RÉEL")
    print("=" * 70)
    print()

    df_temps_reel = telecharger_dataset(
        URL_TEMPS_REEL,
        "eco2mix-national-tr",
    )

    verifier_schema(
        df_temps_reel,
        {"perimetre", "date_heure", "consommation"},
    )

    verifier_perimetre(df_temps_reel)

    df_temps_reel = preparer_donnees_brutes(df_temps_reel)

    sauvegarder_donnees(df_temps_reel, RAW_FILE_TEMPS_REEL)

    sauvegarder_metadata(
        df_temps_reel,
        METADATA_FILE_TEMPS_REEL,
        DATASET_ID_TEMPS_REEL,
        URL_TEMPS_REEL,
        "CONSOMMATION TEMPS RÉEL éCO2mix",
    )

    print()
    print(f"Données temps réel : {RAW_FILE_TEMPS_REEL}")
    print(f"Métadonnées : {METADATA_FILE_TEMPS_REEL}")
    print()

    # ========================================================
    # RÉSUMÉ
    # ========================================================

    print("=" * 70)
    print("TÉLÉCHARGEMENT TERMINÉ")
    print("=" * 70)
    print()
    print(f"Consolidé  : {len(df_consolide):,} lignes "
          f"({df_consolide['date_heure'].min()} -> "
          f"{df_consolide['date_heure'].max()})")
    print(f"Temps réel : {len(df_temps_reel):,} lignes "
          f"({df_temps_reel['date_heure'].min()} -> "
          f"{df_temps_reel['date_heure'].max()})")
    print()


if __name__ == "__main__":
    main()
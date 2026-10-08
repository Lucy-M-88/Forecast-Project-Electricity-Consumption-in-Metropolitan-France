"""
01_telechargement_eco2mix.py

Projet de prévision de la consommation électrique
France métropolitaine.

Source :
    RTE / ODRÉ - éCO2mix
    Jeu : eco2mix-national-cons-def

Rôle de ce fichier :
    1. Télécharger les données éCO2mix depuis l'API ODRÉ.
    2. Vérifier que les colonnes nécessaires sont présentes.
    3. Vérifier le périmètre national utilisé.
    4. Convertir les données dans un format propre.
    5. Sauvegarder les données brutes téléchargées localement.

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
# 1. CONFIGURATION DU PROJET
# ============================================================================

# Identifiant officiel du jeu ODRÉ.
DATASET_ID = "eco2mix-national-cons-def"

# API OpenDataSoft / ODRÉ.
API_BASE_URL = (
    "https://odre.opendatasoft.com/"
    "api/explore/v2.1/catalog/datasets/"
)

# URL de téléchargement CSV.
DOWNLOAD_URL = (
    f"{API_BASE_URL}"
    f"{DATASET_ID}"
    "/exports/csv"
)

# Répertoire racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parent

# Répertoire où seront stockées les données.
DATA_DIR = PROJECT_ROOT / "data"

# Données brutes téléchargées depuis RTE/ODRÉ.
RAW_DIR = DATA_DIR / "raw"

# Fichier de sortie.
RAW_FILE = RAW_DIR / "eco2mix_national_brut.csv"

# Fichier contenant les informations sur le téléchargement.
METADATA_FILE = RAW_DIR / "telechargement_metadata.txt"

# Timeout réseau.
REQUEST_TIMEOUT = 180


# ============================================================================
# 2. CONFIGURATION DES LOGS
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# 3. TELECHARGEMENT
# ============================================================================

def telecharger_donnees() -> pd.DataFrame:
    """
    Télécharge le jeu éCO2mix national depuis l'API ODRÉ.

    Returns
    -------
    pandas.DataFrame
        Données téléchargées.

    Raises
    ------
    requests.HTTPError
        Si le serveur retourne une erreur HTTP.
    requests.RequestException
        En cas de problème réseau.
    """

    logger.info("Début du téléchargement éCO2mix.")
    logger.info("Dataset : %s", DATASET_ID)
    logger.info("URL : %s", DOWNLOAD_URL)

    # Paramètres de l'export.
    #
    # delimiter :
    #     séparateur CSV français.
    #
    # timezone :
    #     on demande explicitement le fuseau Europe/Paris.
    #
    # use_labels :
    #     False permet de récupérer les noms techniques
    #     des champs lorsque l'API les fournit.
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

        # Déclenche une exception si HTTP 4xx / 5xx.
        response.raise_for_status()

    except requests.RequestException as exc:
        logger.error(
            "Erreur lors du téléchargement : %s",
            exc,
        )
        raise

    logger.info(
        "Téléchargement terminé : %.2f Mo",
        len(response.content) / 1024**2,
    )

    # Lecture du CSV directement depuis la réponse HTTP.
    df = pd.read_csv(
        io.BytesIO(response.content),
        sep=";",
    )

    logger.info(
        "Nombre de lignes téléchargées : %d",
        len(df),
    )

    logger.info(
        "Nombre de colonnes téléchargées : %d",
        len(df.columns),
    )

    return df


# ============================================================================
# 4. VERIFICATION DU SCHEMA
# ============================================================================

def verifier_schema(df: pd.DataFrame) -> None:
    """
    Vérifie la présence des colonnes indispensables.

    Nous ne faisons volontairement pas dépendre la suite du projet
    de toutes les colonnes du jeu RTE.

    Les seules colonnes indispensables ici sont :

        perimetre
        date_heure
        consommation
        nature

    La colonne 'nature' est conservée afin de pouvoir distinguer
    les données consolidées des données définitives.
    """

    colonnes_requises = {
        "perimetre",
        "date_heure",
        "consommation",
        "nature",
    }

    colonnes_presentes = set(df.columns)

    colonnes_manquantes = (
        colonnes_requises - colonnes_presentes
    )

    if colonnes_manquantes:
        raise ValueError(
            "Le schéma du jeu de données a changé.\n"
            f"Colonnes manquantes : {sorted(colonnes_manquantes)}\n"
            f"Colonnes disponibles : {sorted(colonnes_presentes)}"
        )

    logger.info(
        "Vérification du schéma : OK."
    )


# ============================================================================
# 5. VERIFICATION DU PERIMETRE
# ============================================================================

def verifier_perimetre(df: pd.DataFrame) -> None:
    """
    Vérifie le périmètre géographique fourni par le jeu national.

    Le jeu national ODRÉ indique un périmètre 'France'.

    On vérifie explicitement cette information au lieu de supposer
    qu'une donnée téléchargée correspond automatiquement au périmètre
    du projet.
    """

    perimetres = (
        df["perimetre"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    logger.info(
        "Périmètres présents dans les données : %s",
        list(perimetres),
    )

    if len(perimetres) != 1:
        raise ValueError(
            "Plusieurs périmètres géographiques sont présents : "
            f"{list(perimetres)}"
        )

    if perimetres[0].lower() != "france":
        raise ValueError(
            "Le périmètre téléchargé n'est pas celui attendu. "
            f"Périmètre trouvé : {perimetres[0]}"
        )

    logger.info(
        "Périmètre national France : OK."
    )


# ============================================================================
# 6. PREPARATION MINIMALE
# ============================================================================

def preparer_donnees_brutes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Effectue uniquement les transformations nécessaires pour
    rendre le fichier exploitable.

    IMPORTANT :
        Ce n'est PAS la préparation du modèle.

    On :
        - garde uniquement les colonnes utiles au projet ;
        - convertit date_heure ;
        - convertit consommation ;
        - trie chronologiquement ;
        - élimine les doublons exacts de date_heure.

    On ne :
        - crée aucun lag ;
        - ne fait aucun remplissage de valeurs manquantes ;
        - ne passe pas encore au pas horaire ;
        - ne crée aucune variable calendaire.
    """

    # ------------------------------------------------------------------
    # Colonnes conservées
    # ------------------------------------------------------------------

    colonnes_utiles = [
        "perimetre",
        "nature",
        "date_heure",
        "consommation",
    ]

    df = df[colonnes_utiles].copy()

    # ------------------------------------------------------------------
    # Conversion date/heure
    # ------------------------------------------------------------------

    df["date_heure"] = pd.to_datetime(
        df["date_heure"],
        errors="coerce",
        utc=True,
    )

    # ------------------------------------------------------------------
    # Conversion consommation
    # ------------------------------------------------------------------

    df["consommation"] = pd.to_numeric(
        df["consommation"],
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # Vérification des dates invalides
    # ------------------------------------------------------------------

    nb_dates_invalides = (
        df["date_heure"].isna().sum()
    )

    if nb_dates_invalides > 0:
        logger.warning(
            "%d dates invalides détectées.",
            nb_dates_invalides,
        )

    # ------------------------------------------------------------------
    # Vérification des consommations invalides
    # ------------------------------------------------------------------

    nb_conso_invalides = (
        df["consommation"].isna().sum()
    )

    if nb_conso_invalides > 0:
        logger.warning(
            "%d consommations manquantes ou invalides détectées.",
            nb_conso_invalides,
        )

    # ------------------------------------------------------------------
    # Suppression uniquement des dates invalides
    # ------------------------------------------------------------------

    df = df.dropna(
        subset=["date_heure"]
    )

    # ------------------------------------------------------------------
    # Tri chronologique
    # ------------------------------------------------------------------

    df = df.sort_values(
        "date_heure"
    )

    # ------------------------------------------------------------------
    # Doublons temporels
    # ------------------------------------------------------------------

    doublons = (
        df["date_heure"]
        .duplicated()
        .sum()
    )

    if doublons > 0:
        logger.warning(
            "%d doublons temporels détectés.",
            doublons,
        )

        # On conserve la première observation.
        #
        # Cette opération sera documentée dans le README.
        df = df.drop_duplicates(
            subset=["date_heure"],
            keep="first",
        )

    # Réindexation propre.
    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================================
# 7. CONTROLE DE LA FREQUENCE TEMPORELLE
# ============================================================================

def verifier_frequence(df: pd.DataFrame) -> None:
    """
    Vérifie que les données correspondent bien au pas demi-heure
    attendu pour la consommation éCO2mix.

    On ne corrige PAS les trous ici.

    Les trous seront traités explicitement dans le fichier
    02_preparation.py.
    """

    dates = (
        df["date_heure"]
        .dropna()
        .sort_values()
    )

    if len(dates) < 2:
        logger.warning(
            "Pas assez de données pour vérifier la fréquence."
        )
        return

    differences = (
        dates
        .diff()
        .dropna()
    )

    frequences = (
        differences
        .value_counts()
        .head(10)
    )

    logger.info(
        "Principaux intervalles temporels observés :"
    )

    for delta, count in frequences.items():
        logger.info(
            "    %s : %d observations",
            delta,
            count,
        )

    # Fréquence attendue :
    # 30 minutes.
    frequence_attendue = pd.Timedelta(
        minutes=30
    )

    nb_pas_30min = (
        differences == frequence_attendue
    ).sum()

    taux_30min = (
        nb_pas_30min / len(differences)
    )

    logger.info(
        "Proportion des intervalles de 30 min : %.2f%%",
        taux_30min * 100,
    )

    # On ne lève pas d'erreur ici :
    # les changements d'heure et les données manquantes
    # doivent être analysés dans le fichier de préparation.


# ============================================================================
# 8. RESUME DES DONNEES
# ============================================================================

def afficher_resume(df: pd.DataFrame) -> None:
    """
    Affiche un résumé permettant de contrôler le téléchargement.
    """

    print()
    print("=" * 70)
    print("RESUME DU TELECHARGEMENT")
    print("=" * 70)

    print(
        f"Nombre de lignes : {len(df):,}"
    )

    print(
        f"Première date    : {df['date_heure'].min()}"
    )

    print(
        f"Dernière date    : {df['date_heure'].max()}"
    )

    print(
        f"Valeurs manquantes consommation : "
        f"{df['consommation'].isna().sum():,}"
    )

    print(
        f"Nombre de dates uniques : "
        f"{df['date_heure'].nunique():,}"
    )

    print()

    print("Nature des données :")

    print(
        df["nature"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()

    print("Périmètre :")

    print(
        df["perimetre"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("=" * 70)
    print()


# ============================================================================
# 9. SAUVEGARDE
# ============================================================================

def sauvegarder_donnees(
    df: pd.DataFrame,
) -> None:
    """
    Sauvegarde les données dans data/raw/.

    Le fichier reste volontairement au format CSV afin de conserver
    une trace simple et facilement inspectable du téléchargement.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Sauvegarde des données : %s",
        RAW_FILE,
    )

    df.to_csv(
        RAW_FILE,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    logger.info(
        "Sauvegarde terminée."
    )


# ============================================================================
# 10. METADONNEES DU TELECHARGEMENT
# ============================================================================

def sauvegarder_metadata(
    df: pd.DataFrame,
) -> None:
    """
    Sauvegarde les informations permettant de documenter
    précisément le téléchargement.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as fichier:

        fichier.write(
            "METADONNEES DU TELECHARGEMENT éCO2mix\n"
        )

        fichier.write(
            "=====================================\n\n"
        )

        fichier.write(
            f"Dataset ODRÉ : {DATASET_ID}\n"
        )

        fichier.write(
            f"URL : {DOWNLOAD_URL}\n"
        )

        fichier.write(
            "Producteur : RTE\n"
        )

        fichier.write(
            "Périmètre : France\n"
        )

        fichier.write(
            "Unité de consommation : MW\n"
        )

        fichier.write(
            "Pas source : demi-heure\n"
        )

        fichier.write(
            f"Nombre de lignes : {len(df)}\n"
        )

        fichier.write(
            f"Première date : {df['date_heure'].min()}\n"
        )

        fichier.write(
            f"Dernière date : {df['date_heure'].max()}\n"
        )

        fichier.write(
            "\nColonnes conservées :\n"
        )

        for colonne in df.columns:
            fichier.write(
                f" - {colonne}\n"
            )

        fichier.write(
            "\nIMPORTANT :\n"
        )

        fichier.write(
            "Ce fichier constitue uniquement l'étape de "
            "téléchargement.\n"
        )

        fichier.write(
            "Aucun retard temporel, variable calendaire ou "
            "variable de modèle n'a été construit ici.\n"
        )


# ============================================================================
# 11. FONCTION PRINCIPALE
# ============================================================================

def main() -> None:
    """
    Pipeline complet du téléchargement.
    """

    print()
    print("=" * 70)
    print("TELECHARGEMENT DES DONNEES éCO2mix")
    print("Projet de prévision - France métropolitaine")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # ETAPE 1
    # --------------------------------------------------------

    df = telecharger_donnees()

    # --------------------------------------------------------
    # ETAPE 2
    # --------------------------------------------------------

    verifier_schema(df)

    # --------------------------------------------------------
    # ETAPE 3
    # --------------------------------------------------------

    verifier_perimetre(df)

    # --------------------------------------------------------
    # ETAPE 4
    # --------------------------------------------------------

    df = preparer_donnees_brutes(df)

    # --------------------------------------------------------
    # ETAPE 5
    # --------------------------------------------------------

    verifier_frequence(df)

    # --------------------------------------------------------
    # ETAPE 6
    # --------------------------------------------------------

    afficher_resume(df)

    # --------------------------------------------------------
    # ETAPE 7
    # --------------------------------------------------------

    sauvegarder_donnees(df)

    # --------------------------------------------------------
    # ETAPE 8
    # --------------------------------------------------------

    sauvegarder_metadata(df)

    print()
    print("Téléchargement terminé avec succès.")
    print()
    print(
        f"Données : {RAW_FILE}"
    )
    print(
        f"Métadonnées : {METADATA_FILE}"
    )
    print()


# ============================================================================
# 12. POINT D'ENTREE
# ============================================================================

if __name__ == "__main__":
    main()
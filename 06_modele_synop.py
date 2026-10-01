# -*- coding: utf-8 -*-

"""
06_modele_synop.py

Objectif
--------
Tester l'effet de l'ajout de variables de température SYNOP
sur le modèle de prévision de consommation électrique.

Variables météo utilisées UNIQUEMENT :
    - température moyenne
    - température minimale
    - température maximale

Principe opérationnel
---------------------
À 14h du jour J, on ne doit utiliser que des informations
météorologiques disponibles avant 14h.

Pour éviter toute fuite d'information, les observations SYNOP
retenues sont celles dont l'heure de mesure UTC est <= 12:00 UTC,
ce qui correspond à une borne conservatrice par rapport à 14h
heure française.

Les températures du jour J sont ensuite agrégées sur les stations
SYNOP situées en France métropolitaine continentale.

IMPORTANT
---------
- Pas de température du jour J+1.
- Pas de température observée après 14h du jour J.
- Pas de météo RTE.
- Pas de prévision météo.
- Pas de variable "nombre de stations" utilisée par le modèle.
- Pas de moyenne artificielle de consommation.
- Même découpage chronologique que le modèle précédent.
"""

from pathlib import Path
import urllib.request
import gzip
import shutil
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(".")

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

for directory in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Période SYNOP
# ------------------------------------------------------------
#
# Le modèle de base couvre 2012 -> 2026.
#
# Pour conserver exactement le même découpage temporel,
# on télécharge donc les années nécessaires.
#
# Si le téléchargement de toutes les années est trop lourd,
# on pourra ensuite décider explicitement d'une période commune.
#

ANNEES_SYNOP = list(range(2012, 2027))


# ------------------------------------------------------------
# URLs des archives SYNOP Météo-France
# ------------------------------------------------------------

URL_SYNOP = (
    "https://donneespubliques.meteofrance.fr/"
    "?date={annee}&extension=csv.gz"
    "&fond=donnee_libre&prefixe=Txt%2FSynop%2FArchive%2Fsynop"
)


# ------------------------------------------------------------
# Fichiers locaux
# ------------------------------------------------------------

DATASET_PREVISION = (
    PROCESSED_DIR / "dataset_prevision.csv"
)

FICHIER_METEO = (
    PROCESSED_DIR / "temperature_synop_france_metropolitaine.csv"
)

FICHIER_DATASET_ENRICHI = (
    PROCESSED_DIR / "dataset_prevision_avec_temperature.csv"
)

FICHIER_MODELE = (
    MODELS_DIR / "modele_random_forest_synop.joblib"
)

FICHIER_RAPPORT = (
    RESULTS_DIR / "resultat_modele_synop.txt"
)


# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

def telecharger_fichier(url, destination):
    """
    Télécharge un fichier uniquement s'il n'existe pas déjà.
    """

    destination = Path(destination)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Fichier déjà présent : {destination}")
        return

    print(f"Téléchargement : {url}")

    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as e:
        if destination.exists():
            destination.unlink()

        raise RuntimeError(
            f"Impossible de télécharger {url}\n"
            f"Erreur : {e}"
        )

    print(
        f"Téléchargement terminé : "
        f"{destination} "
        f"({destination.stat().st_size / 1024 / 1024:.1f} Mo)"
    )


def lire_synop(fichier):
    """
    Lecture robuste d'une archive SYNOP Météo-France.
    """

    print(f"\nLecture : {fichier}")

    df = pd.read_csv(
        fichier,
        sep=";",
        compression="gzip",
        encoding="latin-1",
        low_memory=False
    )

    print(f"Nombre de lignes brutes : {len(df):,}")
    print(f"Nombre de colonnes : {len(df.columns)}")

    colonnes_requises = [
        "validity_time",
        "t",
        "lat",
        "lon",
        "geo_id_wmo"
    ]

    manquantes = [
        c for c in colonnes_requises
        if c not in df.columns
    ]

    if manquantes:
        raise ValueError(
            f"Colonnes SYNOP manquantes : {manquantes}\n"
            f"Colonnes disponibles : {df.columns.tolist()}"
        )

    return df


# ============================================================
# 3. FILTRE GÉOGRAPHIQUE FRANCE MÉTROPOLITAINE
# ============================================================

def filtrer_france_metropolitaine(df):
    """
    Filtre géographique volontairement strict.

    On veut uniquement la France métropolitaine continentale,
    donc :
        - pas de Guadeloupe
        - pas de Martinique
        - pas de Guyane
        - pas de Réunion
        - pas de Mayotte
        - pas de Polynésie
        - pas de Nouvelle-Calédonie
        - pas de Corse

    Le filtre utilise :
        1. validation numérique des coordonnées ;
        2. domaine longitude/latitude continental ;
        3. exclusion des zones étrangères à l'intérieur du
           rectangle via un polygone simplifié de la France
           continentale.

    Les coordonnées SYNOP sont bien des coordonnées géographiques
    en latitude/longitude selon la documentation Météo-France.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Conversion des coordonnées
    # --------------------------------------------------------

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Suppression coordonnées impossibles
    # --------------------------------------------------------

    mask_coord_valides = (
        df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
    )

    df = df.loc[mask_coord_valides].copy()

    # --------------------------------------------------------
    # Domaine géographique de la France continentale
    #
    # Ce rectangle est volontairement plus large que la France
    # afin de ne pas supprimer artificiellement les stations
    # proches des frontières.
    # --------------------------------------------------------

    mask_rectangle = (
        df["lat"].between(41.0, 51.5)
        & df["lon"].between(-5.5, 10.0)
    )

    df = df.loc[mask_rectangle].copy()

    # --------------------------------------------------------
    # Polygone simplifié de sécurité.
    #
    # Ce polygone n'est PAS utilisé pour faire de la cartographie.
    # Il sert uniquement à éliminer les stations manifestement
    # situées hors du territoire continental français.
    #
    # On garde volontairement une marge autour des frontières.
    # --------------------------------------------------------

    polygone_france = np.array([
        [-4.8, 48.7],
        [-4.6, 48.0],
        [-3.2, 47.3],
        [-2.3, 46.8],
        [-1.8, 46.1],
        [-1.3, 45.5],
        [-1.0, 44.8],
        [-1.1, 43.6],
        [-0.2, 43.2],
        [1.0, 42.8],
        [2.0, 42.5],
        [3.2, 42.4],
        [4.0, 43.0],
        [5.0, 43.2],
        [6.0, 43.2],
        [7.2, 43.5],
        [7.7, 44.2],
        [7.5, 45.0],
        [7.7, 45.7],
        [7.1, 46.3],
        [7.6, 47.0],
        [7.6, 47.6],
        [7.2, 48.0],
        [7.5, 48.7],
        [8.2, 49.0],
        [8.0, 49.6],
        [7.2, 49.9],
        [6.2, 49.8],
        [5.3, 50.0],
        [4.2, 50.8],
        [3.0, 50.9],
        [2.0, 51.1],
        [1.0, 51.2],
        [-0.2, 51.1],
        [-1.5, 51.0],
        [-2.5, 50.8],
        [-3.5, 50.5],
        [-4.4, 49.8],
        [-4.8, 48.7]
    ])

    # --------------------------------------------------------
    # Test point dans le polygone par algorithme ray casting.
    # Pas besoin d'une dépendance géographique supplémentaire.
    # --------------------------------------------------------

    def points_dans_polygone(lons, lats, polygon):
        x = np.asarray(lons)
        y = np.asarray(lats)

        poly_x = polygon[:, 0]
        poly_y = polygon[:, 1]

        n = len(polygon)

        inside = np.zeros(len(x), dtype=bool)

        j = n - 1

        for i in range(n):

            xi = poly_x[i]
            yi = poly_y[i]

            xj = poly_x[j]
            yj = poly_y[j]

            intersection = (
                ((yi > y) != (yj > y))
                &
                (
                    x
                    <
                    (xj - xi)
                    * (y - yi)
                    / (
                        (yj - yi)
                        + 1e-12
                    )
                    + xi
                )
            )

            inside ^= intersection

            j = i

        return inside

    mask_france = points_dans_polygone(
        df["lon"].values,
        df["lat"].values,
        polygone_france
    )

    df = df.loc[mask_france].copy()

    # --------------------------------------------------------
    # Contrôle final explicite
    # --------------------------------------------------------

    if len(df) == 0:
        raise ValueError(
            "Le filtre géographique a supprimé toutes les "
            "stations SYNOP. Vérifier les coordonnées."
        )

    print(
        f"Stations/observations après filtre "
        f"France métropolitaine : {len(df):,}"
    )

    print(
        "Latitude min/max : "
        f"{df['lat'].min():.3f} / "
        f"{df['lat'].max():.3f}"
    )

    print(
        "Longitude min/max : "
        f"{df['lon'].min():.3f} / "
        f"{df['lon'].max():.3f}"
    )

    return df


# ============================================================
# 4. PRÉPARATION SYNOP
# ============================================================

def preparer_synop(df):
    """
    Prépare les observations SYNOP pour une utilisation
    opérationnelle à 14h.

    On utilise validity_time, qui correspond à l'heure de mesure.

    On conserve uniquement les observations suffisamment tôt
    pour être disponibles avant 14h heure française.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Date de mesure
    # --------------------------------------------------------

    df["date_utc"] = pd.to_datetime(
        df["validity_time"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(subset=["date_utc"]).copy()

    # --------------------------------------------------------
    # Température
    # --------------------------------------------------------

    df["t"] = pd.to_numeric(
        df["t"],
        errors="coerce"
    )

    # Météo-France documente t en Kelvin.
    df["temperature_c"] = df["t"] - 273.15

    # --------------------------------------------------------
    # Élimination valeurs aberrantes / sentinelles
    #
    # Une température terrestre réaliste pour notre usage
    # doit rester dans une plage physique raisonnable.
    # --------------------------------------------------------

    df.loc[
        ~df["temperature_c"].between(-60, 60),
        "temperature_c"
    ] = np.nan

    df = df.dropna(
        subset=["temperature_c"]
    ).copy()

    # --------------------------------------------------------
    # Heure UTC
    #
    # 14h heure française correspond à :
    #   - 13h UTC en hiver
    #   - 12h UTC en été
    #
    # On retient <= 12h UTC afin de ne jamais prendre une
    # observation pouvant être postérieure à 14h locale.
    #
    # C'est volontairement conservateur.
    # --------------------------------------------------------

    df["heure_utc"] = df["date_utc"].dt.hour

    df = df.loc[
        df["heure_utc"] <= 12
    ].copy()

    # --------------------------------------------------------
    # Passage en heure française
    # --------------------------------------------------------

    df["date_locale"] = (
        df["date_utc"]
        .dt.tz_convert("Europe/Paris")
    )

    df["jour"] = (
        df["date_locale"]
        .dt.date
    )

    df["jour"] = pd.to_datetime(
        df["jour"]
    )

    # --------------------------------------------------------
    # Tri
    # --------------------------------------------------------

    df = df.sort_values(
        ["jour", "date_utc"]
    ).reset_index(drop=True)

    print(
        f"Observations SYNOP disponibles avant 14h : "
        f"{len(df):,}"
    )

    return df


# ============================================================
# 5. AGRÉGATION NATIONALE
# ============================================================

def agreger_temperature(df):
    """
    Calcule pour chaque journée J :

        température_moyenne
        température_minimale
        température_maximale

    sur les observations SYNOP disponibles avant 14h.

    Le nombre de stations n'est PAS une variable du modèle.
    """

    print("\nAgrégation nationale des températures...")

    groupe = (
        df.groupby("jour")["temperature_c"]
        .agg(
            temperature_moyenne="mean",
            temperature_min="min",
            temperature_max="max"
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Contrôle qualité
    # --------------------------------------------------------

    groupe = groupe.dropna(
        subset=[
            "temperature_moyenne",
            "temperature_min",
            "temperature_max"
        ]
    ).copy()

    groupe = groupe.sort_values(
        "jour"
    ).reset_index(drop=True)

    print(
        f"Nombre de jours météo disponibles : "
        f"{len(groupe):,}"
    )

    print(
        "\nExemple :"
    )

    print(
        groupe.head().to_string(index=False)
    )

    return groupe


# ============================================================
# 6. TÉLÉCHARGEMENT DES DONNÉES SYNOP
# ============================================================

def telecharger_synop():
    """
    Télécharge les archives annuelles SYNOP nécessaires.
    """

    fichiers = []

    for annee in ANNEES_SYNOP:

        fichier = (
            RAW_DIR
            / f"synop_{annee}.csv.gz"
        )

        url = URL_SYNOP.format(
            annee=annee
        )

        try:
            telecharger_fichier(
                url,
                fichier
            )

            fichiers.append(fichier)

        except Exception as e:

            print(
                f"\nATTENTION : année {annee} "
                f"non téléchargée."
            )

            print(e)

    if not fichiers:
        raise RuntimeError(
            "Aucune archive SYNOP n'a pu être téléchargée."
        )

    return fichiers


# ============================================================
# 7. CONSTRUCTION DU DATASET MÉTÉO
# ============================================================

def construire_dataset_meteo(fichiers):
    """
    Lit les archives, filtre la métropole et agrège
    les températures.
    """

    toutes_les_donnees = []

    for fichier in fichiers:

        try:

            df = lire_synop(fichier)

            print(
                f"Filtrage géographique : {fichier.name}"
            )

            df = filtrer_france_metropolitaine(
                df
            )

            df = preparer_synop(
                df
            )

            toutes_les_donnees.append(
                df[
                    [
                        "jour",
                        "temperature_c"
                    ]
                ]
            )

        except Exception as e:

            print(
                f"\nATTENTION : impossible de traiter "
                f"{fichier.name}"
            )

            print(e)

    if not toutes_les_donnees:
        raise RuntimeError(
            "Aucune donnée SYNOP exploitable."
        )

    df_total = pd.concat(
        toutes_les_donnees,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Suppression des doublons éventuels
    # --------------------------------------------------------

    df_total = (
        df_total
        .drop_duplicates()
        .reset_index(drop=True)
    )

    temperature = agreger_temperature(
        df_total
    )

    temperature.to_csv(
        FICHIER_METEO,
        sep=";",
        index=False
    )

    print(
        f"\nDataset météo sauvegardé : "
        f"{FICHIER_METEO}"
    )

    return temperature


# ============================================================
# 8. CHARGEMENT DATASET DE PRÉVISION
# ============================================================

def charger_dataset_prevision():
    """
    Charge le dataset créé par 03_creation_dataset.py.
    """

    print(
        f"\nLecture du dataset de prévision : "
        f"{DATASET_PREVISION}"
    )

    df = pd.read_csv(
        DATASET_PREVISION,
        sep=";"
    )

    print(
        f"Lignes : {len(df):,}"
    )

    print(
        "Colonnes :"
    )

    print(
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df["origine_prevision"] = pd.to_datetime(
        df["origine_prevision"],
        errors="coerce"
    )

    df["date_cible"] = pd.to_datetime(
        df["date_cible"],
        errors="coerce"
    )

    return df


# ============================================================
# 9. AJOUT DE LA MÉTÉO
# ============================================================

def ajouter_temperature(
    df,
    temperature
):
    """
    Joint les températures du jour J au dataset.

    Les variables météo concernent uniquement J,
    jamais J+1.
    """

    df = df.copy()

    temperature = temperature.copy()

    temperature["jour"] = pd.to_datetime(
        temperature["jour"]
    )

    # --------------------------------------------------------
    # Jointure sur origine_prevision = jour J
    # --------------------------------------------------------

    df = df.merge(
        temperature,
        left_on="origine_prevision",
        right_on="jour",
        how="left"
    )

    df = df.drop(
        columns=["jour"]
    )

    print(
        "\nAprès ajout de la météo :"
    )

    print(
        f"Lignes : {len(df):,}"
    )

    print(
        "Valeurs météo manquantes :"
    )

    print(
        df[
            [
                "temperature_moyenne",
                "temperature_min",
                "temperature_max"
            ]
        ]
        .isna()
        .sum()
    )

    # --------------------------------------------------------
    # On garde uniquement les jours avec les 3 variables.
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "temperature_moyenne",
            "temperature_min",
            "temperature_max"
        ]
    ).copy()

    print(
        f"\nLignes après suppression des jours "
        f"sans météo : {len(df):,}"
    )

    return df


# ============================================================
# 10. SPLIT CHRONOLOGIQUE IDENTIQUE AU MODÈLE DE BASE
# ============================================================

def split_chronologique(df):
    """
    Même principe que 04_modele.py :

        70 % train
        15 % validation
        15 % test

    Le split est effectué sur les jours d'origine,
    jamais ligne par ligne.
    """

    jours = (
        df["origine_prevision"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n = len(jours)

    n_train = int(
        n * 0.70
    )

    n_validation = int(
        n * 0.15
    )

    jours_train = jours.iloc[
        :n_train
    ]

    jours_validation = jours.iloc[
        n_train:
        n_train + n_validation
    ]

    jours_test = jours.iloc[
        n_train + n_validation:
    ]

    train = df[
        df["origine_prevision"]
        .isin(jours_train)
    ].copy()

    validation = df[
        df["origine_prevision"]
        .isin(jours_validation)
    ].copy()

    test = df[
        df["origine_prevision"]
        .isin(jours_test)
    ].copy()

    print(
        "\nDÉCOUPAGE CHRONOLOGIQUE"
    )

    print(
        "-" * 70
    )

    print(
        f"TRAIN      : "
        f"{len(jours_train):,} jours | "
        f"{jours_train.min().date()} -> "
        f"{jours_train.max().date()}"
    )

    print(
        f"VALIDATION : "
        f"{len(jours_validation):,} jours | "
        f"{jours_validation.min().date()} -> "
        f"{jours_validation.max().date()}"
    )

    print(
        f"TEST       : "
        f"{len(jours_test):,} jours | "
        f"{jours_test.min().date()} -> "
        f"{jours_test.max().date()}"
    )

    return (
        train,
        validation,
        test
    )


# ============================================================
# 11. MODÈLE RANDOM FOREST AVEC TEMPÉRATURE
# ============================================================

def entrainer_modele(
    train,
    validation,
    test
):
    """
    Entraîne le nouveau modèle.

    Variables :
        consommation_j7
        consommation_j1an
        heure
        jour_semaine
        mois
        weekend
        temperature_moyenne
        temperature_min
        temperature_max
    """

    features = [
        "consommation_j7",
        "consommation_j1an",
        "heure",
        "jour_semaine",
        "mois",
        "weekend",
        "temperature_moyenne",
        "temperature_min",
        "temperature_max"
    ]

    cible = "consommation_cible"

    X_train = train[features]
    y_train = train[cible]

    X_validation = validation[features]
    y_validation = validation[cible]

    X_test = test[features]
    y_test = test[cible]

    print(
        "\nVariables utilisées :"
    )

    for variable in features:
        print(
            f"  - {variable}"
        )

    print(
        "\nEntraînement Random Forest..."
    )

    modele = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    modele.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    prediction_validation = (
        modele.predict(
            X_validation
        )
    )

    mae_validation = mean_absolute_error(
        y_validation,
        prediction_validation
    )

    rmse_validation = np.sqrt(
        mean_squared_error(
            y_validation,
            prediction_validation
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    prediction_test = (
        modele.predict(
            X_test
        )
    )

    mae_test = mean_absolute_error(
        y_test,
        prediction_test
    )

    rmse_test = np.sqrt(
        mean_squared_error(
            y_test,
            prediction_test
        )
    )

    print(
        "\nRÉSULTATS MODÈLE AVEC TEMPÉRATURE"
    )

    print(
        "-" * 70
    )

    print(
        f"Validation MAE  : "
        f"{mae_validation:.2f} MW"
    )

    print(
        f"Validation RMSE : "
        f"{rmse_validation:.2f} MW"
    )

    print(
        f"Test MAE        : "
        f"{mae_test:.2f} MW"
    )

    print(
        f"Test RMSE       : "
        f"{rmse_test:.2f} MW"
    )

    # --------------------------------------------------------
    # Importance des variables
    # --------------------------------------------------------

    importances = pd.DataFrame({
        "variable": features,
        "importance": modele.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    print(
        "\nIMPORTANCE DES VARIABLES"
    )

    print(
        importances.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Sauvegarde modèle
    # --------------------------------------------------------

    joblib.dump(
        modele,
        FICHIER_MODELE
    )

    print(
        f"\nModèle sauvegardé : "
        f"{FICHIER_MODELE}"
    )

    return (
        modele,
        prediction_test,
        y_test,
        importances,
        mae_validation,
        rmse_validation,
        mae_test,
        rmse_test
    )


# ============================================================
# 12. COMPARAISON AVEC LE MODÈLE EXISTANT
# ============================================================

def comparaison(
    mae_test_synop,
    rmse_test_synop
):
    """
    Compare le nouveau modèle aux résultats déjà obtenus
    avec le modèle sans température.
    """

    # Résultats obtenus précédemment
    mae_base = 2996.611661
    rmse_base = 4203.457484

    gain_mae = (
        (mae_base - mae_test_synop)
        / mae_base
        * 100
    )

    gain_rmse = (
        (rmse_base - rmse_test_synop)
        / rmse_base
        * 100
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPARAISON MODÈLE SANS / AVEC TEMPÉRATURE"
    )

    print(
        "=" * 70
    )

    print(
        f"\n{'':35s}"
        f"{'Sans météo':>15s}"
        f"{'Avec météo':>15s}"
    )

    print(
        f"{'MAE (MW)':35s}"
        f"{mae_base:15.2f}"
        f"{mae_test_synop:15.2f}"
    )

    print(
        f"{'RMSE (MW)':35s}"
        f"{rmse_base:15.2f}"
        f"{rmse_test_synop:15.2f}"
    )

    print(
        f"\nVariation MAE  : "
        f"{gain_mae:+.2f} %"
    )

    print(
        f"Variation RMSE : "
        f"{gain_rmse:+.2f} %"
    )

    return (
        gain_mae,
        gain_rmse
    )


# ============================================================
# 13. SAUVEGARDE DU RAPPORT
# ============================================================

def sauvegarder_rapport(
    df,
    mae_validation,
    rmse_validation,
    mae_test,
    rmse_test,
    importances,
    gain_mae,
    gain_rmse
):

    lignes = []

    lignes.append(
        "ÉVALUATION DU MODÈLE AVEC TEMPÉRATURE SYNOP"
    )

    lignes.append(
        "=" * 70
    )

    lignes.append(
        ""
    )

    lignes.append(
        "Variables météo utilisées :"
    )

    lignes.append(
        "  - température_moyenne"
    )

    lignes.append(
        "  - température_min"
    )

    lignes.append(
        "  - température_max"
    )

    lignes.append(
        ""
    )

    lignes.append(
        "Les températures proviennent uniquement des observations "
        "SYNOP disponibles avant 14h du jour J."
    )

    lignes.append(
        "Aucune donnée météo du jour J+1 n'est utilisée."
    )

    lignes.append(
        ""
    )

    lignes.append(
        "Nombre de lignes du dataset enrichi : "
        f"{len(df):,}"
    )

    lignes.append(
        ""
    )

    lignes.append(
        f"Validation MAE  : {mae_validation:.2f} MW"
    )

    lignes.append(
        f"Validation RMSE : {rmse_validation:.2f} MW"
    )

    lignes.append(
        ""
    )

    lignes.append(
        f"Test MAE        : {mae_test:.2f} MW"
    )

    lignes.append(
        f"Test RMSE       : {rmse_test:.2f} MW"
    )

    lignes.append(
        ""
    )

    lignes.append(
        "Comparaison avec le modèle sans météo :"
    )

    lignes.append(
        f"MAE : {gain_mae:+.2f} %"
    )

    lignes.append(
        f"RMSE : {gain_rmse:+.2f} %"
    )

    lignes.append(
        ""
    )

    lignes.append(
        "Importance des variables :"
    )

    lignes.append(
        importances.to_string(
            index=False
        )
    )

    lignes.append(
        ""
    )

    lignes.append(
        "AUDIT DE DISPONIBILITÉ"
    )

    lignes.append(
        "-" * 70
    )

    lignes.append(
        "Origine consommation J-7 : observation historique"
    )

    lignes.append(
        "Origine consommation J-1 an : observation historique"
    )

    lignes.append(
        "Calendrier : connu à l'avance"
    )

    lignes.append(
        "Température moyenne : observations SYNOP avant 14h"
    )

    lignes.append(
        "Température minimale : observations SYNOP avant 14h"
    )

    lignes.append(
        "Température maximale : observations SYNOP avant 14h"
    )

    lignes.append(
        "Cible : consommation de J+1"
    )

    lignes.append(
        ""
    )

    lignes.append(
        "Aucune variable de consommation de J+1 n'est utilisée "
        "comme prédicteur."
    )

    FICHIER_RAPPORT.write_text(
        "\n".join(lignes),
        encoding="utf-8"
    )

    print(
        f"\nRapport sauvegardé : "
        f"{FICHIER_RAPPORT}"
    )


# ============================================================
# 14. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODÈLE RANDOM FOREST + TEMPÉRATURE SYNOP"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Étape 1 : télécharger SYNOP
    # --------------------------------------------------------

    fichiers_synop = (
        telecharger_synop()
    )

    # --------------------------------------------------------
    # Étape 2 : construire dataset météo
    # --------------------------------------------------------

    temperature = (
        construire_dataset_meteo(
            fichiers_synop
        )
    )

    # --------------------------------------------------------
    # Étape 3 : charger dataset prévision
    # --------------------------------------------------------

    dataset = (
        charger_dataset_prevision()
    )

    # --------------------------------------------------------
    # Étape 4 : ajouter météo du jour J
    # --------------------------------------------------------

    dataset = (
        ajouter_temperature(
            dataset,
            temperature
        )
    )

    # --------------------------------------------------------
    # Sauvegarde dataset enrichi
    # --------------------------------------------------------

    dataset.to_csv(
        FICHIER_DATASET_ENRICHI,
        sep=";",
        index=False
    )

    print(
        f"\nDataset enrichi sauvegardé : "
        f"{FICHIER_DATASET_ENRICHI}"
    )

    # --------------------------------------------------------
    # Étape 5 : split chronologique
    # --------------------------------------------------------

    train, validation, test = (
        split_chronologique(
            dataset
        )
    )

    # --------------------------------------------------------
    # Étape 6 : entraînement
    # --------------------------------------------------------

    (
        modele,
        prediction_test,
        y_test,
        importances,
        mae_validation,
        rmse_validation,
        mae_test,
        rmse_test
    ) = entrainer_modele(
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # Étape 7 : comparaison
    # --------------------------------------------------------

    gain_mae, gain_rmse = comparaison(
        mae_test,
        rmse_test
    )

    # --------------------------------------------------------
    # Étape 8 : rapport
    # --------------------------------------------------------

    sauvegarder_rapport(
        dataset,
        mae_validation,
        rmse_validation,
        mae_test,
        rmse_test,
        importances,
        gain_mae,
        gain_rmse
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TERMINÉ"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
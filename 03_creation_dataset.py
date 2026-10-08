"""
03_creation_dataset.py

Création du dataset supervisé pour la prévision de consommation électrique.

OBJECTIF
--------
À 14h du jour J, on souhaite prévoir les 24 consommations horaires
du jour J+1.

Pour chaque heure h de J+1 :

    X :
        - consommation exactement à h de J-7
        - consommation exactement à h de J-1 an
        - variables calendaires connues à l'avance

    y :
        - consommation réellement observée à h de J+1

IMPORTANT
---------
- Aucune moyenne.
- Aucune interpolation.
- Aucune donnée future dans X.
- Les consommations utilisées comme variables explicatives
  sont des observations historiques.
- J-7 signifie exactement 7 jours calendaires avant.
- J-1 an signifie exactement la même date de l'année précédente.
- L'heure doit être exactement la même.
- Les journées cibles doivent contenir exactement 24 heures HH:00.
- Les journées DST de 23 ou 25 heures ne sont pas utilisées comme
  journées cibles.
- Le 29 février est exclu comme journée cible si la correspondance
  J-1 an n'existe pas.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# VACANCES SCOLAIRES FRANÇAISES
# ============================================================
#
# Calendrier officiel des vacances scolaires 2010-2027.
#
# On utilise une liste de tuples (date_debut, date_fin, zone)
# où les dates sont au format "YYYY-MM-DD".
#
# Zones :
#   - A : Besançon, Bordeaux, Clermont-Ferrand, Dijon,
#         Grenoble, Limoges, Lyon, Poitiers
#   - B : Aix-Marseille, Amiens, Caen, Lille, Nancy-Metz,
#         Nantes, Nice, Orléans-Tours, Reims, Rennes,
#         Rouen, Strasbourg
#   - C : Créteil, Paris, Versailles, Montpellier, Toulouse
#
# Périodes :
#   - Toussaint : ~2 semaines fin octobre
#   - Noël : ~2 semaines fin décembre
#   - Hiver : ~2 semaines février-mars (dates par zone)
#   - Printemps : ~2 semaines avril (dates par zone)
#   - Été : ~2 mois juillet-août
#
# Les vacances de Noël et d'été ont les mêmes dates pour
# toutes les zones. Les vacances de Toussaint sont souvent
# alignées mais peuvent différer.
# ============================================================

VACANCES_SCOLAIRES = [
    # --------------------------------------------------------
    # 2010
    # --------------------------------------------------------
    ("2010-02-06", "2010-02-22", "A"),
    ("2010-02-13", "2010-03-01", "B"),
    ("2010-02-20", "2010-03-08", "C"),
    ("2010-04-03", "2010-04-19", "A"),
    ("2010-04-10", "2010-04-26", "B"),
    ("2010-04-17", "2010-05-03", "C"),
    # Toussaint 2010 : dates non renseignées (déjà passé dans le dataset)

    # --------------------------------------------------------
    # 2011
    # --------------------------------------------------------
    ("2011-02-12", "2011-02-28", "A"),
    ("2011-02-19", "2011-03-07", "B"),
    ("2011-02-26", "2011-03-14", "C"),
    ("2011-04-09", "2011-04-25", "A"),
    ("2011-04-16", "2011-05-02", "B"),
    ("2011-04-23", "2011-05-09", "C"),

    # --------------------------------------------------------
    # 2012
    # --------------------------------------------------------
    ("2012-02-11", "2012-02-27", "A"),
    ("2012-02-18", "2012-03-05", "B"),
    ("2012-02-25", "2012-03-12", "C"),
    ("2012-04-07", "2012-04-23", "A"),
    ("2012-04-14", "2012-04-30", "B"),
    ("2012-04-21", "2012-05-07", "C"),

    # --------------------------------------------------------
    # 2013
    # --------------------------------------------------------
    ("2013-02-16", "2013-03-04", "A"),
    ("2013-02-23", "2013-03-11", "B"),
    ("2013-03-02", "2013-03-18", "C"),
    ("2013-04-06", "2013-04-22", "A"),
    ("2013-04-13", "2013-04-29", "B"),
    ("2013-04-20", "2013-05-06", "C"),

    # --------------------------------------------------------
    # 2014
    # --------------------------------------------------------
    ("2014-02-15", "2014-03-03", "A"),
    ("2014-02-22", "2014-03-10", "B"),
    ("2014-03-01", "2014-03-17", "C"),
    ("2014-04-05", "2014-04-21", "A"),
    ("2014-04-12", "2014-04-28", "B"),
    ("2014-04-19", "2014-05-05", "C"),

    # --------------------------------------------------------
    # 2015
    # --------------------------------------------------------
    ("2015-02-07", "2015-02-23", "A"),
    ("2015-02-14", "2015-03-02", "B"),
    ("2015-02-21", "2015-03-09", "C"),
    ("2015-04-04", "2015-04-20", "A"),
    ("2015-04-11", "2015-04-27", "B"),
    ("2015-04-18", "2015-05-04", "C"),

    # --------------------------------------------------------
    # 2016
    # --------------------------------------------------------
    ("2016-02-06", "2016-02-22", "A"),
    ("2016-02-13", "2016-02-29", "B"),
    ("2016-02-20", "2016-03-07", "C"),
    ("2016-04-02", "2016-04-18", "A"),
    ("2016-04-09", "2016-04-25", "B"),
    ("2016-04-16", "2016-05-02", "C"),

    # --------------------------------------------------------
    # 2017
    # --------------------------------------------------------
    ("2017-02-11", "2017-02-27", "A"),
    ("2017-02-18", "2017-03-06", "B"),
    ("2017-02-25", "2017-03-13", "C"),
    ("2017-04-08", "2017-04-24", "A"),
    ("2017-04-15", "2017-05-01", "B"),
    ("2017-04-22", "2017-05-08", "C"),

    # --------------------------------------------------------
    # 2018
    # --------------------------------------------------------
    ("2018-02-10", "2018-02-26", "A"),
    ("2018-02-17", "2018-03-05", "B"),
    ("2018-02-24", "2018-03-12", "C"),
    ("2018-04-07", "2018-04-23", "A"),
    ("2018-04-14", "2018-04-30", "B"),
    ("2018-04-21", "2018-05-07", "C"),

    # --------------------------------------------------------
    # 2019
    # --------------------------------------------------------
    ("2019-02-09", "2019-02-25", "A"),
    ("2019-02-16", "2019-03-04", "B"),
    ("2019-02-23", "2019-03-11", "C"),
    ("2019-04-06", "2019-04-22", "A"),
    ("2019-04-13", "2019-04-29", "B"),
    ("2019-04-20", "2019-05-06", "C"),

    # --------------------------------------------------------
    # 2020
    # --------------------------------------------------------
    ("2020-02-08", "2020-02-24", "A"),
    ("2020-02-15", "2020-03-02", "B"),
    ("2020-02-22", "2020-03-09", "C"),
    ("2020-04-04", "2020-04-20", "A"),
    ("2020-04-11", "2020-04-27", "B"),
    ("2020-04-18", "2020-05-04", "C"),

    # --------------------------------------------------------
    # 2021
    # --------------------------------------------------------
    ("2021-02-06", "2021-02-22", "A"),
    ("2021-02-13", "2021-03-01", "B"),
    ("2021-02-20", "2021-03-08", "C"),
    ("2021-04-10", "2021-04-26", "A"),
    ("2021-04-17", "2021-05-03", "B"),
    ("2021-04-24", "2021-05-10", "C"),

    # --------------------------------------------------------
    # 2022
    # --------------------------------------------------------
    ("2022-02-05", "2022-02-21", "A"),
    ("2022-02-12", "2022-02-28", "B"),
    ("2022-02-19", "2022-03-07", "C"),
    ("2022-04-09", "2022-04-25", "A"),
    ("2022-04-16", "2022-05-02", "B"),
    ("2022-04-23", "2022-05-09", "C"),

    # --------------------------------------------------------
    # 2023
    # --------------------------------------------------------
    ("2023-02-04", "2023-02-20", "A"),
    ("2023-02-11", "2023-02-27", "B"),
    ("2023-02-18", "2023-03-06", "C"),
    ("2023-04-08", "2023-04-24", "A"),
    ("2023-04-15", "2023-05-01", "B"),
    ("2023-04-22", "2023-05-08", "C"),

    # --------------------------------------------------------
    # 2024
    # --------------------------------------------------------
    ("2024-02-10", "2024-02-26", "A"),
    ("2024-02-17", "2024-03-04", "B"),
    ("2024-02-24", "2024-03-11", "C"),
    ("2024-04-06", "2024-04-22", "A"),
    ("2024-04-13", "2024-04-29", "B"),
    ("2024-04-20", "2024-05-06", "C"),

    # --------------------------------------------------------
    # 2025
    # --------------------------------------------------------
    ("2025-02-08", "2025-02-24", "A"),
    ("2025-02-15", "2025-03-03", "B"),
    ("2025-02-22", "2025-03-10", "C"),
    ("2025-04-05", "2025-04-21", "A"),
    ("2025-04-12", "2025-04-28", "B"),
    ("2025-04-19", "2025-05-05", "C"),

    # --------------------------------------------------------
    # 2026
    # --------------------------------------------------------
    ("2026-02-07", "2026-02-23", "A"),
    ("2026-02-14", "2026-03-02", "B"),
    ("2026-02-21", "2026-03-09", "C"),
    ("2026-04-04", "2026-04-20", "A"),
    ("2026-04-11", "2026-04-27", "B"),
    ("2026-04-18", "2026-05-04", "C"),

    # --------------------------------------------------------
    # 2027
    # --------------------------------------------------------
    ("2027-02-06", "2027-02-22", "A"),
    ("2027-02-13", "2027-03-01", "B"),
    ("2027-02-20", "2027-03-08", "C"),
    ("2027-04-03", "2027-04-19", "A"),
    ("2027-04-10", "2027-04-26", "B"),
    ("2027-04-17", "2027-05-03", "C"),
]


# ------------------------------------------------------------
# Vacances communes à toutes les zones (Noël, été, Toussaint)
# ------------------------------------------------------------
#
# Ces vacances ont les mêmes dates pour toutes les zones.
# On les traite à part pour simplifier.
# ------------------------------------------------------------

VACANCES_COMMUNES = [
    # Noël : 2 semaines fin décembre / début janvier
    # On les encode par année scolaire, mais on les applique
    # à la fin décembre de l'année N et début janvier de N+1.

    # Toussaint : dernière semaine d'octobre + 1re de novembre
    # (~2 semaines)

    # Été : juillet-août
    # (on peut considérer "vacances d'été" du 1er juillet au 31 août)
]


def get_zones_vacances(date):
    """
    Retourne le set des zones en vacances pour une date donnée.
    """
    date_str = date.strftime("%Y-%m-%d")
    zones = set()

    # Vacances de printemps et d'hiver (par zone)
    for debut, fin, zone in VACANCES_SCOLAIRES:
        if debut <= date_str <= fin:
            zones.add(zone)

    # Vacances communes : Noël et été
    mois = date.month
    jour = date.day

    # Vacances de Noël : du 20 décembre au 5 janvier environ
    if (mois == 12 and jour >= 20) or (mois == 1 and jour <= 5):
        zones.update(["A", "B", "C"])

    # Vacances d'été : du 7 juillet au 31 août
    if mois == 7 and jour >= 7:
        zones.update(["A", "B", "C"])
    if mois == 8:
        zones.update(["A", "B", "C"])

    # Vacances de la Toussaint : du 20 octobre au 4 novembre environ
    if mois == 10 and jour >= 20:
        zones.update(["A", "B", "C"])
    if mois == 11 and jour <= 4:
        zones.update(["A", "B", "C"])

    return zones

def indicateurs_vacances(date):
    """
    Retourne 3 indicateurs binaires de vacances scolaires :
        - zone_A_vacances : 1 si la zone A est en vacances
        - zone_B_vacances : 1 si la zone B est en vacances
        - zone_C_vacances : 1 si la zone C est en vacances

    Une date est considérée en vacances pour une zone si elle
    tombe dans une période de vacances de cette zone, que ce
    soit pour les vacances zonées (hiver, printemps) ou les
    vacances communes (Noël, été, Toussaint).
    """

    date_str = date.strftime("%Y-%m-%d")

    zones = {"A": 0, "B": 0, "C": 0}

    # --------------------------------------------------------
    # Vacances zonées (hiver, printemps) : A, B, C séparées
    # --------------------------------------------------------

    for debut, fin, zone in VACANCES_SCOLAIRES:
        if debut <= date_str <= fin:
            zones[zone] = 1

    # --------------------------------------------------------
    # Vacances communes : Noël, été, Toussaint
    # --------------------------------------------------------

    mois = date.month
    jour = date.day

    vacances_communes = False

    # Vacances de Noël : 20 décembre -> 5 janvier
    if (mois == 12 and jour >= 20) or (mois == 1 and jour <= 5):
        vacances_communes = True

    # Vacances d'été : 7 juillet -> 31 août
    if mois == 7 and jour >= 7:
        vacances_communes = True
    if mois == 8:
        vacances_communes = True

    # Vacances de la Toussaint : 20 octobre -> 4 novembre
    if mois == 10 and jour >= 20:
        vacances_communes = True
    if mois == 11 and jour <= 4:
        vacances_communes = True

    if vacances_communes:
        zones["A"] = 1
        zones["B"] = 1
        zones["C"] = 1

    return zones["A"], zones["B"], zones["C"]

# ============================================================
# JOURS FÉRIÉS FRANÇAIS
# ============================================================
#
# Liste des jours fériés français pour la période 2010-2027.
# Fériés fixes : 1er jan, 1er mai, 8 mai, 14 juil, 15 août,
#                1er nov, 11 nov, 25 déc.
# Fériés mobiles : lundi de Pâques, Ascension, Pentecôte.
#
# Les dates ont été calculées via la méthode de Meeus/Jones/
# Butcher pour Pâques. Elles sont figées ici pour éviter toute
# dépendance externe et garantir la reproductibilité.
# ============================================================

JOURS_FERIES = set([
    # 2010
    "2010-01-01", "2010-04-05", "2010-05-01", "2010-05-08",
    "2010-05-13", "2010-05-24", "2010-07-14", "2010-08-15",
    "2010-11-01", "2010-11-11", "2010-12-25",
    # 2011
    "2011-01-01", "2011-04-25", "2011-05-01", "2011-05-08",
    "2011-06-02", "2011-06-13", "2011-07-14", "2011-08-15",
    "2011-11-01", "2011-11-11", "2011-12-25",
    # 2012
    "2012-01-01", "2012-04-09", "2012-05-01", "2012-05-08",
    "2012-05-17", "2012-05-28", "2012-07-14", "2012-08-15",
    "2012-11-01", "2012-11-11", "2012-12-25",
    # 2013
    "2013-01-01", "2013-04-01", "2013-05-01", "2013-05-08",
    "2013-05-09", "2013-05-20", "2013-07-14", "2013-08-15",
    "2013-11-01", "2013-11-11", "2013-12-25",
    # 2014
    "2014-01-01", "2014-04-21", "2014-05-01", "2014-05-08",
    "2014-05-29", "2014-06-09", "2014-07-14", "2014-08-15",
    "2014-11-01", "2014-11-11", "2014-12-25",
    # 2015
    "2015-01-01", "2015-04-06", "2015-05-01", "2015-05-08",
    "2015-05-14", "2015-05-25", "2015-07-14", "2015-08-15",
    "2015-11-01", "2015-11-11", "2015-12-25",
    # 2016
    "2016-01-01", "2016-03-28", "2016-05-01", "2016-05-08",
    "2016-05-05", "2016-05-16", "2016-07-14", "2016-08-15",
    "2016-11-01", "2016-11-11", "2016-12-25",
    # 2017
    "2017-01-01", "2017-04-17", "2017-05-01", "2017-05-08",
    "2017-05-25", "2017-06-05", "2017-07-14", "2017-08-15",
    "2017-11-01", "2017-11-11", "2017-12-25",
    # 2018
    "2018-01-01", "2018-04-02", "2018-05-01", "2018-05-08",
    "2018-05-10", "2018-05-21", "2018-07-14", "2018-08-15",
    "2018-11-01", "2018-11-11", "2018-12-25",
    # 2019
    "2019-01-01", "2019-04-22", "2019-05-01", "2019-05-08",
    "2019-05-30", "2019-06-10", "2019-07-14", "2019-08-15",
    "2019-11-01", "2019-11-11", "2019-12-25",
    # 2020
    "2020-01-01", "2020-04-13", "2020-05-01", "2020-05-08",
    "2020-05-21", "2020-06-01", "2020-07-14", "2020-08-15",
    "2020-11-01", "2020-11-11", "2020-12-25",
    # 2021
    "2021-01-01", "2021-04-05", "2021-05-01", "2021-05-08",
    "2021-05-13", "2021-05-24", "2021-07-14", "2021-08-15",
    "2021-11-01", "2021-11-11", "2021-12-25",
    # 2022
    "2022-01-01", "2022-04-18", "2022-05-01", "2022-05-08",
    "2022-05-26", "2022-06-06", "2022-07-14", "2022-08-15",
    "2022-11-01", "2022-11-11", "2022-12-25",
    # 2023
    "2023-01-01", "2023-04-10", "2023-05-01", "2023-05-08",
    "2023-05-18", "2023-05-29", "2023-07-14", "2023-08-15",
    "2023-11-01", "2023-11-11", "2023-12-25",
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
    # 2027
    "2027-01-01", "2027-03-29", "2027-05-01", "2027-05-08",
    "2027-05-06", "2027-05-17", "2027-07-14", "2027-08-15",
    "2027-11-01", "2027-11-11", "2027-12-25",
])


def est_ferie(date):
    """
    Retourne 1 si la date est un jour férié français, 0 sinon.
    """
    return int(date.strftime("%Y-%m-%d") in JOURS_FERIES)


# ============================================================
# 1. PARAMÈTRES
# ============================================================

INPUT_FILE = Path(
    "data/processed/eco2mix_national_horaire.csv"
)

OUTPUT_FILE = Path(
    "data/processed/dataset_prevision.csv"
)

REPORT_FILE = Path(
    "data/processed/rapport_creation_dataset.txt"
)

TIMEZONE = "Europe/Paris"


# ============================================================
# 2. CRÉATION DU DOSSIER DE SORTIE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

print("=" * 70)
print("CRÉATION DU DATASET DE PRÉVISION")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_FILE}\n"
        "Lance d'abord 02_preparation.py."
    )

df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    low_memory=False
)

print(
    f"Nombre de lignes chargées : {len(df):,}"
)


# ============================================================
# 4. VÉRIFICATION DES COLONNES
# ============================================================

required_columns = [
    "date_heure",
    "date_heure_utc",
    "date",
    "heure",
    "minute",
    "jour_semaine",
    "weekend",
    "mois",
    "annee",
    "consommation_horaire",
    "jour_complet",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Colonnes manquantes dans le fichier préparé : "
        + ", ".join(missing_columns)
    )


# ============================================================
# 5. CONVERSION DES DATES
# ============================================================

print("\nConversion des dates...")

# ============================================================
# 5. CONVERSION DES DATES
# ============================================================

print("\nConversion des dates...")

# IMPORTANT :
# date_heure contient des heures locales Europe/Paris.
#
# Lors du passage à l'heure d'hiver, on rencontre :
#
#   02:00 +02:00
#   02:00 +01:00
#
# Ces deux timestamps ont le même affichage local mais
# correspondent à deux instants physiques différents.
#
# Pour éviter les "mixed time zones" de pandas, on utilise
# UTC comme représentation technique, puis on reconstruit
# l'heure locale avec Europe/Paris.

df["date_heure_utc"] = pd.to_datetime(
    df["date_heure_utc"],
    errors="coerce",
    utc=True
)

df = df.dropna(
    subset=["date_heure_utc"]
).copy()

# Reconstitution systématique de l'heure locale française.
df["date_heure"] = (
    df["date_heure_utc"]
    .dt.tz_convert(TIMEZONE)
)

df["consommation_horaire"] = pd.to_numeric(
    df["consommation_horaire"],
    errors="coerce"
)

df = df.dropna(
    subset=["consommation_horaire"]
).copy()

df["consommation_horaire"] = pd.to_numeric(
    df["consommation_horaire"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "date_heure",
        "date_heure_utc",
        "consommation_horaire"
    ]
).copy()


# ============================================================
# 6. VÉRIFICATION : UNIQUEMENT HH:00
# ============================================================

if not (df["minute"] == 0).all():
    raise ValueError(
        "ERREUR : le fichier contient des observations "
        "qui ne sont pas à une heure entière HH:00."
    )

print("✓ Toutes les observations sont à HH:00.")


# ============================================================
# 7. TRI CHRONOLOGIQUE
# ============================================================

df = df.sort_values(
    "date_heure_utc"
).reset_index(drop=True)


# ============================================================
# 8. VÉRIFICATION DES DOUBLONS
# ============================================================

if df["date_heure_utc"].duplicated().any():
    raise ValueError(
        "ERREUR : plusieurs observations correspondent "
        "au même instant UTC."
    )


# ============================================================
# 9. INDEX DE RECHERCHE PAR DATE + HEURE
# ============================================================

"""
Nous créons un index basé sur :

    date locale + heure

Cela permet de demander précisément :

    2024-01-15 07:00

et de récupérer uniquement la consommation de 07:00.

Aucune moyenne n'est effectuée.
"""

df["date_locale"] = (
    df["date_heure"]
    .dt.date
)

df["heure_locale"] = (
    df["date_heure"]
    .dt.hour
)


# ============================================================
# 10. DÉTECTION DES DOUBLONS DATE + HEURE
# ============================================================

duplicates_local = df.duplicated(
    subset=[
        "date_locale",
        "heure_locale"
],
    keep=False
)

nb_duplicates_local = duplicates_local.sum()

if nb_duplicates_local > 0:

    print(
        "\nAttention : certaines journées contiennent plusieurs "
        "observations pour la même heure locale."
    )

    print(
        f"Nombre de lignes concernées : "
        f"{nb_duplicates_local:,}"
    )

    print(
        "Cela peut notamment apparaître lors du passage "
        "à l'heure d'hiver."
    )

    print(
        "Ces journées ne seront pas utilisées comme journées "
        "cibles afin d'éviter toute ambiguïté."
    )


# ============================================================
# 11. CRÉATION D'UN IDENTIFIANT DATE + HEURE
# ============================================================

df["cle_date_heure"] = list(
    zip(
        df["date_locale"],
        df["heure_locale"]
    )
)


# ============================================================
# 12. DICTIONNAIRE DE CONSOMMATION
# ============================================================

"""
Dictionnaire :

    (date, heure) -> consommation

Exemple :

    (2024-01-15, 7) -> consommation observée à 07:00

Aucune transformation de la consommation.
"""

consommation_lookup = {}

for row in df.itertuples(index=False):

    key = (
        row.date_locale,
        int(row.heure_locale)
    )

    # En cas de doublon local, on ne choisit pas arbitrairement
    # une valeur. On conserve une liste pour détecter l'ambiguïté.

    if key not in consommation_lookup:
        consommation_lookup[key] = []

    consommation_lookup[key].append(
        float(row.consommation_horaire)
    )


# ============================================================
# 13. JOURS DISPONIBLES
# ============================================================

jours_disponibles = sorted(
    df["date_locale"].unique()
)

print(
    f"\nNombre de jours présents dans les données : "
    f"{len(jours_disponibles):,}"
)


# ============================================================
# 14. IDENTIFICATION DES JOURNÉES DE 24 HEURES
# ============================================================

heures_par_jour = (
    df.groupby("date_locale")["heure_locale"]
    .nunique()
)

jours_24h = set(
    heures_par_jour[
        heures_par_jour == 24
    ].index
)

print(
    f"Journées contenant 24 heures distinctes : "
    f"{len(jours_24h):,}"
)


# ============================================================
# 15. FONCTION DE RÉCUPÉRATION D'UNE CONSOMMATION
# ============================================================

def get_consumption(date_cible, heure):
    """
    Retourne la consommation correspondant exactement
    à une date et une heure.

    Exemple :
        get_consumption(2024-01-15, 7)

    retourne uniquement la valeur observée à :

        2024-01-15 07:00

    Aucun calcul n'est réalisé.
    """

    key = (
        date_cible,
        int(heure)
    )

    valeurs = consommation_lookup.get(
        key,
        []
    )

    # Aucune observation
    if len(valeurs) == 0:
        return np.nan

    # Plusieurs observations pour la même heure locale :
    # situation ambiguë, on ne choisit pas arbitrairement.
    if len(valeurs) > 1:
        return np.nan

    return valeurs[0]


# ============================================================
# 16. CRÉATION DES EXEMPLES
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTION DES EXEMPLES DE PRÉVISION")
print("=" * 70)

rows = []

nb_origines_testees = 0
nb_origines_conservees = 0

nb_cibles_incompletes = 0
nb_lag_j7_manquant = 0
nb_lag_j1an_manquant = 0
nb_cibles_ambiguës = 0


# ------------------------------------------------------------
# Chaque date correspond à un jour J.
#
# À 14h de J :
#
#     prévision de J+1
#
# Pour chaque heure de J+1 :
#
#     X1 = consommation J+1 - 7 jours
#     X2 = consommation J+1 - 1 an
#     y  = consommation J+1
# ------------------------------------------------------------

for date_j in jours_disponibles:

    nb_origines_testees += 1

    # --------------------------------------------------------
    # Origine de prévision
    # --------------------------------------------------------

    date_j_timestamp = pd.Timestamp(
        date_j
    )

    date_cible = (
        date_j_timestamp
        + pd.Timedelta(days=1)
    ).date()

    # --------------------------------------------------------
    # On ne peut créer une prévision 24h que si J+1
    # possède exactement les 24 heures HH:00.
    # --------------------------------------------------------

    if date_cible not in jours_24h:
        nb_cibles_incompletes += 1
        continue

    # --------------------------------------------------------
    # Vérification que J+1 contient bien les heures 0 à 23.
    # --------------------------------------------------------

    heures_cible = []

    for heure in range(24):

        valeur = get_consumption(
            date_cible,
            heure
        )

        if pd.isna(valeur):
            heures_cible = []
            break

        heures_cible.append(
            heure
        )

    if len(heures_cible) != 24:
        nb_cibles_incompletes += 1
        continue

    # --------------------------------------------------------
    # Pour construire les variables J-7 et J-1 an,
    # on utilise des dates calendaires.
    # --------------------------------------------------------

    date_j7 = (
        date_cible_timestamp :=
        pd.Timestamp(date_cible)
    ) - pd.Timedelta(days=7)

    date_j7 = date_j7.date()

    # --------------------------------------------------------
    # J-1 an :
    #
    # On utilise DateOffset(years=1), et NON :
    #
    #     365 jours
    #
    # afin de conserver la même date calendaire.
    # --------------------------------------------------------

    try:

        date_j1an = (
            pd.Timestamp(date_cible)
            - pd.DateOffset(years=1)
        ).date()

    except Exception:

        nb_lag_j1an_manquant += 1
        continue

    # --------------------------------------------------------
    # Vérification des 24 heures historiques.
    # --------------------------------------------------------

    exemple_temporaire = []

    exemple_valide = True

    for heure in range(24):

        # ====================================================
        # VARIABLE J-7
        # ====================================================

        conso_j7 = get_consumption(
            date_j7,
            heure
        )

        if pd.isna(conso_j7):

            nb_lag_j7_manquant += 1
            exemple_valide = False
            break

        # ====================================================
        # VARIABLE J-1 AN
        # ====================================================

        conso_j1an = get_consumption(
            date_j1an,
            heure
        )

        if pd.isna(conso_j1an):

            nb_lag_j1an_manquant += 1
            exemple_valide = False
            break

        # ====================================================
        # CIBLE RÉELLE
        # ====================================================

        y = get_consumption(
            date_cible,
            heure
        )

        if pd.isna(y):

            exemple_valide = False
            break

        # ====================================================
        # VARIABLES CALENDAIRES
        # ====================================================

        timestamp_cible = pd.Timestamp(
            date_cible
        )

        jour_semaine = (
            timestamp_cible.dayofweek
        )

        mois = (
            timestamp_cible.month
        )

        weekend = int(
            jour_semaine >= 5
        )

        est_ferie_jour = est_ferie(timestamp_cible)

        (
            zone_A_vacances,
            zone_B_vacances,
            zone_C_vacances,
        ) = indicateurs_vacances(timestamp_cible)

        # ====================================================
        # LIGNE DU DATASET
        # ====================================================

        exemple_temporaire.append({

            # ----------------------------------------------
            # Identification de l'exemple
            # ----------------------------------------------

            "origine_prevision": (
                date_j_timestamp
            ),

            "date_cible": (
                pd.Timestamp(date_cible)
            ),

            "heure_cible": heure,

            # ----------------------------------------------
            # Variables explicatives
            # ----------------------------------------------

            "consommation_j7": conso_j7,

            "consommation_j1an": conso_j1an,

            "heure": heure,

            "jour_semaine": jour_semaine,

            "mois": mois,

            "weekend": weekend,

            "est_ferie": est_ferie_jour,

            "zone_A_vacances": zone_A_vacances,
            "zone_B_vacances": zone_B_vacances,
            "zone_C_vacances": zone_C_vacances,

            # ----------------------------------------------
            # Variable cible
            # ----------------------------------------------

            "consommation_cible": y,
        })

    # --------------------------------------------------------
    # Si une seule heure est impossible à construire,
    # on rejette l'exemple entier.
    #
    # Pas question de créer une journée de 23 ou 22 valeurs.
    # --------------------------------------------------------

    if not exemple_valide:
        continue

    # --------------------------------------------------------
    # Les 24 heures sont valides.
    # --------------------------------------------------------

    rows.extend(
        exemple_temporaire
    )

    nb_origines_conservees += 1


# ============================================================
# 17. CRÉATION DU DATAFRAME FINAL
# ============================================================

dataset = pd.DataFrame(rows)

if dataset.empty:
    raise ValueError(
        "Le dataset final est vide. "
        "Vérifie la période des données et les historiques J-7/J-1 an."
    )


# ============================================================
# 18. TRI FINAL
# ============================================================

dataset = dataset.sort_values(
    [
        "origine_prevision",
        "heure_cible"
    ]
).reset_index(drop=True)


# ============================================================
# 19. VÉRIFICATION : 24 LIGNES PAR JOUR DE PRÉVISION
# ============================================================

lignes_par_origine = (
    dataset
    .groupby("origine_prevision")
    .size()
)

origines_invalides = lignes_par_origine[
    lignes_par_origine != 24
]

if len(origines_invalides) > 0:
    raise ValueError(
        "ERREUR : certaines origines ne contiennent pas "
        "exactement 24 heures."
    )

print(
    f"\nNombre d'origines de prévision conservées : "
    f"{nb_origines_conservees:,}"
)

print(
    f"Nombre total de lignes : "
    f"{len(dataset):,}"
)

print(
    f"Vérification : "
    f"{len(dataset):,} / {nb_origines_conservees:,} = "
    f"{len(dataset) / nb_origines_conservees:.0f} lignes par origine."
)


# ============================================================
# 20. VÉRIFICATION DE LA DATE CIBLE
# ============================================================

ecarts = (
    dataset["date_cible"]
    - dataset["origine_prevision"]
)

if not (ecarts == pd.Timedelta(days=1)).all():
    raise ValueError(
        "ERREUR : date_cible n'est pas systématiquement "
        "J+1 par rapport à origine_prevision."
    )


# ============================================================
# 21. VÉRIFICATION DES HEURES
# ============================================================

heures_par_origine = (
    dataset
    .groupby("origine_prevision")["heure_cible"]
    .apply(list)
)

for origine, heures in heures_par_origine.items():

    if heures != list(range(24)):

        raise ValueError(
            f"ERREUR : les heures de {origine} "
            f"ne correspondent pas exactement à 0...23."
        )


# ============================================================
# 22. VÉRIFICATION J-7
# ============================================================

print("\nVérification des variables J-7...")

if dataset["consommation_j7"].isna().any():
    raise ValueError(
        "ERREUR : consommation_j7 contient des valeurs manquantes."
    )

print(
    "✓ consommation_j7 disponible pour toutes les lignes."
)


# ============================================================
# 23. VÉRIFICATION J-1 AN
# ============================================================

print("Vérification des variables J-1 an...")

if dataset["consommation_j1an"].isna().any():
    raise ValueError(
        "ERREUR : consommation_j1an contient des valeurs manquantes."
    )

print(
    "✓ consommation_j1an disponible pour toutes les lignes."
)


# ============================================================
# 24. VÉRIFICATION DE LA CIBLE
# ============================================================

print("Vérification des cibles...")

if dataset["consommation_cible"].isna().any():
    raise ValueError(
        "ERREUR : la cible contient des valeurs manquantes."
    )

print(
    "✓ Toutes les cibles sont observées."
)


# ============================================================
# 25. VÉRIFICATION CRITIQUE ANTI-LEAKAGE
# ============================================================

"""
À l'origine de prévision J à 14h :

    consommation_j7
        -> historique

    consommation_j1an
        -> historique

    variables calendaires de J+1
        -> connues à l'avance

    consommation_cible
        -> J+1
        -> UNIQUEMENT utilisée comme Y

Aucune variable explicative ne doit contenir
la consommation de J+1.
"""

variables_X = [
    "consommation_j7",
    "consommation_j1an",
    "heure",
    "jour_semaine",
    "mois",
    "weekend",
]

variable_y = "consommation_cible"

print("\n" + "=" * 70)
print("AUDIT ANTI-LEAKAGE")
print("=" * 70)

print("Variables explicatives :")

for variable in variables_X:
    print(f"  - {variable}")

print(
    f"\nVariable cible : {variable_y}"
)

print(
    "\n✓ La consommation de J+1 n'est présente "
    "que dans la variable cible."
)


# ============================================================
# 26. VÉRIFICATION : PAS DE VARIABLE RTE DE PRÉVISION
# ============================================================

colonnes_interdites = [
    "prevision_j",
    "prevision_j1",
    "forecast",
    "prediction",
]

colonnes_dataset = set(
    dataset.columns
)

variables_interdites_presentes = (
    colonnes_dataset.intersection(
        colonnes_interdites
    )
)

if variables_interdites_presentes:

    raise ValueError(
        "ERREUR : des variables de prévision RTE "
        "ont été introduites dans le dataset : "
        + ", ".join(
            variables_interdites_presentes
        )
    )

print(
    "✓ Aucune prévision RTE n'est utilisée comme variable explicative."
)


# ============================================================
# 27. STATISTIQUES DESCRIPTIVES
# ============================================================

print("\n" + "=" * 70)
print("STATISTIQUES DU DATASET")
print("=" * 70)

print(
    f"Première origine : "
    f"{dataset['origine_prevision'].min()}"
)

print(
    f"Dernière origine : "
    f"{dataset['origine_prevision'].max()}"
)

print(
    f"Nombre d'origines : "
    f"{dataset['origine_prevision'].nunique():,}"
)

print(
    f"Nombre de lignes : "
    f"{len(dataset):,}"
)

print(
    f"Nombre de variables : "
    f"{len(dataset.columns)}"
)


# ============================================================
# 28. CONTRÔLE DES VALEURS
# ============================================================

for variable in [
    "consommation_j7",
    "consommation_j1an",
    "consommation_cible",
]:

    if not np.isfinite(
        dataset[variable]
    ).all():

        raise ValueError(
            f"ERREUR : valeurs non finies dans {variable}."
        )


# ============================================================
# 29. SAUVEGARDE
# ============================================================

dataset.to_csv(
    OUTPUT_FILE,
    sep=";",
    index=False
)

print(
    f"\nDataset enregistré dans : {OUTPUT_FILE}"
)


# ============================================================
# 30. RAPPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RAPPORT DE CRÉATION DU DATASET DE PRÉVISION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "OBJECTIF\n"
    )

    f.write(
        "--------\n"
    )

    f.write(
        "À 14h du jour J, prévoir les 24 consommations "
        "horaires du jour J+1.\n\n"
    )

    f.write(
        "VARIABLES EXPLICATIVES\n"
    )

    f.write(
        "----------------------\n"
    )

    f.write(
        "consommation_j7 : consommation exactement à "
        "la même heure de J-7.\n"
    )

    f.write(
        "consommation_j1an : consommation exactement à "
        "la même heure et à la même date de l'année précédente.\n"
    )

    f.write(
        "heure : heure de la journée cible.\n"
    )

    f.write(
        "jour_semaine : jour de la semaine de J+1.\n"
    )

    f.write(
        "mois : mois de J+1.\n"
    )

    f.write(
        "weekend : indicateur samedi/dimanche.\n\n"
    )

    f.write(
        "VARIABLE CIBLE\n"
    )

    f.write(
        "--------------\n"
    )

    f.write(
        "consommation_cible : consommation réellement "
        "observée à l'heure correspondante de J+1.\n\n"
    )

    f.write(
        "RÈGLES IMPORTANTES\n"
    )

    f.write(
        "-------------------\n"
    )

    f.write(
        "- aucune moyenne entre demi-heures\n"
    )

    f.write(
        "- aucune interpolation\n"
    )

    f.write(
        "- aucune valeur artificielle\n"
    )

    f.write(
        "- uniquement des observations HH:00\n"
    )

    f.write(
        "- J-7 = exactement 7 jours calendaires avant\n"
    )

    f.write(
        "- J-1 an = même date de l'année précédente\n"
    )

    f.write(
        "- 24 heures exactement pour chaque journée cible\n"
    )

    f.write(
        "- aucune prévision RTE utilisée comme variable X\n"
    )

    f.write(
        "- aucune donnée de J+1 utilisée dans X\n\n"
    )

    f.write(
        "STATISTIQUES\n"
    )

    f.write(
        "------------\n"
    )

    f.write(
        f"Origines testées : {nb_origines_testees:,}\n"
    )

    f.write(
        f"Origines conservées : {nb_origines_conservees:,}\n"
    )

    f.write(
        f"Nombre total de lignes : {len(dataset):,}\n"
    )

    f.write(
        f"Cibles incomplètes : {nb_cibles_incompletes:,}\n"
    )

    f.write(
        f"Lags J-7 manquants : {nb_lag_j7_manquant:,}\n"
    )

    f.write(
        f"Lags J-1 an manquants : {nb_lag_j1an_manquant:,}\n"
    )

    f.write(
        f"Ambiguïtés horaires locales : {nb_cibles_ambiguës:,}\n"
    )

    f.write(
        "\nFIN DU RAPPORT\n"
    )


print(
    f"Rapport enregistré dans : {REPORT_FILE}"
)

print("\n" + "=" * 70)
print("ÉTAPE 3 TERMINÉE")
print("=" * 70)

print(
    f"\nNombre de jours fériés dans le dataset : "
    f"{dataset['est_ferie'].sum()}"
)

print(
    f"Proportion de jours fériés : "
    f"{dataset['est_ferie'].mean() * 100:.2f} %"
)

print("\n" + "=" * 70)
print("VÉRIFICATION DES VACANCES SCOLAIRES")
print("=" * 70)

print(
    f"\nNombre de jours en vacances zone A : "
    f"{dataset['zone_A_vacances'].sum()}"
)
print(
    f"Nombre de jours en vacances zone B : "
    f"{dataset['zone_B_vacances'].sum()}"
)
print(
    f"Nombre de jours en vacances zone C : "
    f"{dataset['zone_C_vacances'].sum()}"
)

print(
    f"\nProportion de jours en vacances :"
)
print(
    f"  Zone A : "
    f"{dataset['zone_A_vacances'].mean() * 100:.2f} %"
)
print(
    f"  Zone B : "
    f"{dataset['zone_B_vacances'].mean() * 100:.2f} %"
)
print(
    f"  Zone C : "
    f"{dataset['zone_C_vacances'].mean() * 100:.2f} %"
)

# Vérifier qu'il n'y a pas de jours avec les 3 zones à 0
# en dehors des périodes de non-vacances

nb_zones_actives = (
    dataset["zone_A_vacances"]
    + dataset["zone_B_vacances"]
    + dataset["zone_C_vacances"]
)

print(
    f"\nRépartition du nombre de zones en vacances :"
)
print(
    nb_zones_actives
    .value_counts()
    .sort_index()
    .to_string()
)
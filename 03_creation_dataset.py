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
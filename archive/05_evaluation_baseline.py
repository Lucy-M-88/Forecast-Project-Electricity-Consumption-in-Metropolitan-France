"""
05_evaluation_baseline.py
Script comparant le modèle à plusieurs baselines. Nous l'avons remplacé par un autre plus tard dans le projet.

Évaluation hors échantillon du modèle de prévision.

Comparaison :
    1. Random Forest
    2. Benchmark J-7
    3. Benchmark J-1 an
    4. Moyenne J-7 / J-1 an
    5. Dernier jour complet disponible à 14h
    6. Prévision officielle RTE éCO2mix prevision_j1

IMPORTANT :
    - Aucun benchmark n'est utilisé pour entraîner le Random Forest.
    - prevision_j1 de RTE est uniquement un benchmark externe.
    - Aucun calcul de moyenne n'est utilisé pour fabriquer
      une observation horaire.
    - Les données réelles utilisées sont uniquement les observations
      HH:00 déjà disponibles.
    - L'évaluation est réalisée uniquement sur le TEST chronologique.
"""

from pathlib import Path
import io
import requests
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. CHEMINS
# ============================================================

DATASET_PATH = Path(
    "data/processed/dataset_prevision.csv"
)

MODEL_PATH = Path(
    "models/modele_random_forest.joblib"
)

HORAIRE_PATH = Path(
    "data/processed/eco2mix_national_horaire.csv"
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTATS_HORAIRES = (
    OUTPUT_DIR / "resultats_horaires.csv"
)

RESULTATS_JOURNALIERS = (
    OUTPUT_DIR / "resultats_journaliers.csv"
)

COMPARAISON_MODELES = (
    OUTPUT_DIR / "comparaison_modeles.csv"
)

RESULTATS_PICS = (
    OUTPUT_DIR / "resultats_pics.csv"
)

PIRES_JOURS = (
    OUTPUT_DIR / "pires_jours.csv"
)

RAPPORT = (
    OUTPUT_DIR / "rapport_evaluation.txt"
)


# ============================================================
# 2. VARIABLES DU MODÈLE
# ============================================================

FEATURES = [
    "consommation_j7",
    "consommation_j1an",
    "heure",
    "jour_semaine",
    "mois",
    "weekend",
]

TARGET = "consommation_cible"


# ============================================================
# 3. API RTE / éCO2mix
# ============================================================

RTE_API_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/eco2mix-national-cons-def/exports/csv"
)

RTE_PARAMS = {
    "delimiter": ";",
    "timezone": "Europe/Paris",
    "use_labels": "false",
}


# ============================================================
# 4. FONCTIONS
# ============================================================

def mae(y_true, y_pred):
    """
    Mean Absolute Error.
    """
    return np.mean(
        np.abs(y_true - y_pred)
    )


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    """
    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def reconstruire_split_test(df):
    """
    Reproduit exactement le découpage chronologique
    utilisé dans 04_modele.py.

    70 % TRAIN
    15 % VALIDATION
    15 % TEST

    Le découpage se fait par journée d'origine,
    jamais ligne par ligne.
    """

    dates = sorted(
        df["origine_prevision"]
        .dropna()
        .unique()
    )

    n_dates = len(dates)

    n_train = int(
        n_dates * 0.70
    )

    n_validation = int(
        n_dates * 0.15
    )

    dates_train = dates[
        :n_train
    ]

    dates_validation = dates[
        n_train:
        n_train + n_validation
    ]

    dates_test = dates[
        n_train + n_validation:
    ]

    train = df[
        df["origine_prevision"].isin(
            dates_train
        )
    ].copy()

    validation = df[
        df["origine_prevision"].isin(
            dates_validation
        )
    ].copy()

    test = df[
        df["origine_prevision"].isin(
            dates_test
        )
    ].copy()

    return (
        train,
        validation,
        test,
    )


def telecharger_prevision_rte():
    """
    Télécharge éCO2mix et récupère uniquement :

        date_heure
        prevision_j1

    prevision_j1 est utilisée comme benchmark externe.

    Elle n'est JAMAIS utilisée pour entraîner notre modèle.
    """

    print(
        "\nTéléchargement des données "
        "éCO2mix pour récupérer prevision_j1..."
    )

    response = requests.get(
        RTE_API_URL,
        params=RTE_PARAMS,
        timeout=120,
    )

    response.raise_for_status()

    print(
        "✓ Téléchargement RTE terminé."
    )

    df = pd.read_csv(
        io.BytesIO(
            response.content
        ),
        sep=";",
        low_memory=False,
    )

    # --------------------------------------------------------
    # Vérification des colonnes
    # --------------------------------------------------------

    colonnes_requises = [
        "perimetre",
        "date_heure",
        "prevision_j1",
    ]

    colonnes_manquantes = [
        colonne
        for colonne in colonnes_requises
        if colonne not in df.columns
    ]

    if colonnes_manquantes:
        raise ValueError(
            "Colonnes RTE manquantes : "
            + ", ".join(
                colonnes_manquantes
            )
        )

    # --------------------------------------------------------
    # France
    # --------------------------------------------------------

    df = df[
        df["perimetre"]
        .astype(str)
        .str.strip()
        == "France"
    ].copy()

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df["date_heure_utc"] = pd.to_datetime(
        df["date_heure"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=[
            "date_heure_utc"
        ]
    ).copy()

    df["date_heure"] = (
        df["date_heure_utc"]
        .dt.tz_convert(
            "Europe/Paris"
        )
    )

    # --------------------------------------------------------
    # Prévision J-1
    # --------------------------------------------------------

    df["prevision_j1"] = pd.to_numeric(
        df["prevision_j1"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Uniquement les heures exactes HH:00.
    #
    # Pas de moyenne.
    # Pas d'interpolation.
    # --------------------------------------------------------

    df = df[
        df["date_heure"].dt.minute == 0
    ].copy()

    df = df[
        [
            "date_heure",
            "prevision_j1",
        ]
    ].copy()

    # Sécurité contre les doublons
    df = df.drop_duplicates(
        subset=[
            "date_heure"
        ],
        keep="first",
    )

    df = df.sort_values(
        "date_heure"
    )

    print(
        "Nombre de prévisions RTE "
        f"horaires : {len(df):,}"
    )

    return df


def ajouter_benchmark_dernier_jour_complet(
    test,
    df_horaire,
):
    """
    Benchmark opérationnel :

        À 14h le jour J,
        le dernier jour entièrement connu est J-1.

    Pour chaque heure h de J+1 :

        prévision = consommation de J-1 à l'heure h

    Exemple :

        origine = lundi 14h
        cible   = mardi

        benchmark = consommation du dimanche
                   à chaque heure correspondante.

    Cette méthode respecte donc la contrainte
    de disponibilité à 14h.
    """

    test = test.copy()

    df_horaire = df_horaire.copy()

    # --------------------------------------------------------
    # Date de J-1
    # --------------------------------------------------------

    test["date_veille"] = (
        pd.to_datetime(
            test["origine_prevision"]
        )
        - pd.Timedelta(
            days=1
        )
    )

    # --------------------------------------------------------
    # Préparation des observations horaires
    # --------------------------------------------------------

    df_horaire["date"] = (
        df_horaire["date_heure"]
        .dt.date
    )

    df_horaire["heure"] = (
        df_horaire["date_heure"]
        .dt.hour
    )

    # --------------------------------------------------------
    # Clé = date + heure
    # --------------------------------------------------------

    df_horaire["cle"] = list(
        zip(
            df_horaire["date"],
            df_horaire["heure"],
        )
    )

    mapping = dict(
        zip(
            df_horaire["cle"],
            df_horaire[
                "consommation_horaire"
            ],
        )
    )

    test["cle_veille"] = list(
        zip(
            test["date_veille"].dt.date,
            test["heure_cible"].astype(int),
        )
    )

    test[
        "benchmark_dernier_jour_complet"
    ] = test[
        "cle_veille"
    ].map(
        mapping
    )

    test = test.drop(
        columns=[
            "cle_veille",
        ]
    )

    return test


def calculer_metrics_horaires(
    resultats,
    colonnes_modeles,
):
    """
    MAE et RMSE sur toutes les heures disponibles.
    """

    lignes = []

    for colonne in colonnes_modeles:

        sous = resultats[
            [
                TARGET,
                colonne,
            ]
        ].dropna()

        if len(sous) == 0:
            continue

        lignes.append(
            {
                "modele": colonne,
                "nombre_heures": len(
                    sous
                ),
                "MAE_MW": mae(
                    sous[TARGET].values,
                    sous[colonne].values,
                ),
                "RMSE_MW": rmse(
                    sous[TARGET].values,
                    sous[colonne].values,
                ),
            }
        )

    return pd.DataFrame(
        lignes
    )


def calculer_metrics_journalieres(
    resultats,
    colonnes_modeles,
):
    """
    Compare les totaux journaliers.

    Pour chaque journée :

        total réel
        total prévu

    Puis :

        MAE du total journalier
        RMSE du total journalier
        biais moyen
    """

    agregations = {
        TARGET: "sum"
    }

    for colonne in colonnes_modeles:
        agregations[
            colonne
        ] = "sum"

    journaliers = (
        resultats
        .groupby(
            "date_cible"
        )
        .agg(
            agregations
        )
        .reset_index()
    )

    lignes = []

    for colonne in colonnes_modeles:

        sous = journaliers[
            [
                TARGET,
                colonne,
            ]
        ].dropna()

        if len(sous) == 0:
            continue

        erreur = (
            sous[colonne]
            - sous[TARGET]
        )

        lignes.append(
            {
                "modele": colonne,
                "nombre_jours": len(
                    sous
                ),
                "MAE_total_MWh": np.mean(
                    np.abs(erreur)
                ),
                "RMSE_total_MWh": np.sqrt(
                    np.mean(
                        erreur ** 2
                    )
                ),
                "biais_total_MWh": np.mean(
                    erreur
                ),
            }
        )

    return (
        pd.DataFrame(lignes),
        journaliers,
    )


def calculer_metrics_pics(
    resultats,
    colonnes_modeles,
):
    """
    Compare le pic journalier prévu
    au pic journalier réellement observé.

    Pour chaque journée :

        - pic réel
        - pic prévu
        - erreur du pic
        - heure réelle du pic
        - heure prévue du pic
        - écart d'heure
    """

    lignes = []

    for date, groupe in resultats.groupby(
        "date_cible"
    ):

        # ----------------------------------------------------
        # Réalité
        # ----------------------------------------------------

        groupe_reel = groupe.dropna(
            subset=[
                TARGET
            ]
        )

        if len(groupe_reel) != 24:
            continue

        index_reel = (
            groupe_reel[
                TARGET
            ].idxmax()
        )

        pic_reel = (
            groupe_reel.loc[
                index_reel,
                TARGET,
            ]
        )

        heure_pic_reel = int(
            groupe_reel.loc[
                index_reel,
                "heure_cible",
            ]
        )

        # ----------------------------------------------------
        # Chaque modèle
        # ----------------------------------------------------

        for colonne in colonnes_modeles:

            groupe_modele = (
                groupe.dropna(
                    subset=[
                        TARGET,
                        colonne,
                    ]
                )
            )

            if len(groupe_modele) != 24:
                continue

            index_prevu = (
                groupe_modele[
                    colonne
                ].idxmax()
            )

            pic_prevu = (
                groupe_modele.loc[
                    index_prevu,
                    colonne,
                ]
            )

            heure_pic_prevue = int(
                groupe_modele.loc[
                    index_prevu,
                    "heure_cible",
                ]
            )

            erreur_pic = (
                pic_prevu
                - pic_reel
            )

            ecart_heure = (
                heure_pic_prevue
                - heure_pic_reel
            )

            lignes.append(
                {
                    "date_cible": date,
                    "modele": colonne,
                    "pic_reel_MW": pic_reel,
                    "pic_prevu_MW": pic_prevu,
                    "erreur_pic_MW": erreur_pic,
                    "erreur_pic_absolue_MW": abs(
                        erreur_pic
                    ),
                    "heure_pic_reel": (
                        heure_pic_reel
                    ),
                    "heure_pic_prevue": (
                        heure_pic_prevue
                    ),
                    "ecart_heure_pic": (
                        ecart_heure
                    ),
                    "pic_correct": (
                        heure_pic_prevue
                        == heure_pic_reel
                    ),
                }
            )

    details = pd.DataFrame(
        lignes
    )

    if len(details) == 0:
        return (
            pd.DataFrame(),
            details,
        )

    resume = (
        details
        .groupby("modele")
        .agg(
            nombre_jours=(
                "date_cible",
                "count",
            ),
            MAE_pic_MW=(
                "erreur_pic_absolue_MW",
                "mean",
            ),
            biais_pic_MW=(
                "erreur_pic_MW",
                "mean",
            ),
            ecart_moyen_heure_pic=(
                "ecart_heure_pic",
                lambda x: np.mean(
                    np.abs(x)
                ),
            ),
            taux_heure_pic_correct=(
                "pic_correct",
                "mean",
            ),
        )
        .reset_index()
    )

    resume[
        "taux_heure_pic_correct"
    ] *= 100

    return (
        resume,
        details,
    )


# ============================================================
# 5. DÉBUT DU PROGRAMME
# ============================================================

print("=" * 70)
print("ÉTAPE 5 - ÉVALUATION")
print("=" * 70)


# ============================================================
# 6. CHARGEMENT DU DATASET
# ============================================================

print(
    "\nChargement du dataset..."
)

# IMPORTANT :
# le dataset produit par 03_creation_dataset.py
# utilise le séparateur ;
df = pd.read_csv(
    DATASET_PATH,
    sep=";",
)

print(
    f"Nombre de lignes : {len(df):,}"
)

print(
    "Colonnes détectées :"
)

for colonne in df.columns:
    print(
        f"  - {colonne}"
    )


# ============================================================
# 7. VÉRIFICATIONS DU DATASET
# ============================================================

colonnes_requises = [
    "origine_prevision",
    "date_cible",
    "heure_cible",
    "consommation_j7",
    "consommation_j1an",
    "heure",
    "jour_semaine",
    "mois",
    "weekend",
    "consommation_cible",
]

colonnes_manquantes = [
    colonne
    for colonne in colonnes_requises
    if colonne not in df.columns
]

if colonnes_manquantes:

    raise ValueError(
        "Colonnes manquantes dans "
        "dataset_prevision.csv : "
        + ", ".join(
            colonnes_manquantes
        )
    )


# ============================================================
# 8. TYPES
# ============================================================

df["origine_prevision"] = pd.to_datetime(
    df["origine_prevision"],
    errors="coerce",
)

df["date_cible"] = pd.to_datetime(
    df["date_cible"],
    errors="coerce",
)

df["heure_cible"] = pd.to_numeric(
    df["heure_cible"],
    errors="coerce",
)

df["consommation_j7"] = pd.to_numeric(
    df["consommation_j7"],
    errors="coerce",
)

df["consommation_j1an"] = pd.to_numeric(
    df["consommation_j1an"],
    errors="coerce",
)

df["consommation_cible"] = pd.to_numeric(
    df["consommation_cible"],
    errors="coerce",
)


# ============================================================
# 9. CHARGEMENT DU RANDOM FOREST
# ============================================================

print(
    "\nChargement du Random Forest..."
)

modele = joblib.load(
    MODEL_PATH
)

print(
    "✓ Modèle chargé."
)


# ============================================================
# 10. RECONSTRUCTION TRAIN / VALIDATION / TEST
# ============================================================

print(
    "\nReconstruction du découpage chronologique..."
)

train, validation, test = (
    reconstruire_split_test(
        df
    )
)

print(
    f"TRAIN       : {len(train):,} lignes"
)

print(
    f"VALIDATION  : {len(validation):,} lignes"
)

print(
    f"TEST        : {len(test):,} lignes"
)

print(
    "\nPériode TRAIN : "
    f"{train['origine_prevision'].min().date()} "
    "-> "
    f"{train['origine_prevision'].max().date()}"
)

print(
    "Période VALIDATION : "
    f"{validation['origine_prevision'].min().date()} "
    "-> "
    f"{validation['origine_prevision'].max().date()}"
)

print(
    "Période TEST : "
    f"{test['origine_prevision'].min().date()} "
    "-> "
    f"{test['origine_prevision'].max().date()}"
)


# ============================================================
# 11. PRÉDICTION RANDOM FOREST
# ============================================================

print(
    "\nPrédiction du Random Forest..."
)

X_test = test[
    FEATURES
]

test[
    "prediction_random_forest"
] = modele.predict(
    X_test
)

print(
    "✓ Prédictions réalisées."
)


# ============================================================
# 12. BENCHMARK J-7
# ============================================================

test[
    "benchmark_j7"
] = test[
    "consommation_j7"
]


# ============================================================
# 13. BENCHMARK J-1 AN
# ============================================================

test[
    "benchmark_j1an"
] = test[
    "consommation_j1an"
]


# ============================================================
# 14. MOYENNE J-7 / J-1 AN
# ============================================================

test[
    "benchmark_moyenne"
] = (
    test[
        "consommation_j7"
    ]
    + test[
        "consommation_j1an"
    ]
) / 2


# ============================================================
# 15. CHARGEMENT DES DONNÉES HORAIRES
# ============================================================

print(
    "\nChargement des observations horaires..."
)

df_horaire = pd.read_csv(
    HORAIRE_PATH,
    sep=";",
)

print(
    f"Nombre d'observations horaires : "
    f"{len(df_horaire):,}"
)

# ------------------------------------------------------------
# Dates UTC
# ------------------------------------------------------------

df_horaire[
    "date_heure_utc"
] = pd.to_datetime(
    df_horaire[
        "date_heure_utc"
    ],
    errors="coerce",
    utc=True,
)

df_horaire[
    "date_heure"
] = (
    df_horaire[
        "date_heure_utc"
    ]
    .dt.tz_convert(
        "Europe/Paris"
    )
)

df_horaire[
    "consommation_horaire"
] = pd.to_numeric(
    df_horaire[
        "consommation_horaire"
    ],
    errors="coerce",
)


# ============================================================
# 16. BENCHMARK DERNIER JOUR COMPLET
# ============================================================

print(
    "\nCalcul du benchmark "
    "dernier jour complet..."
)

test = (
    ajouter_benchmark_dernier_jour_complet(
        test,
        df_horaire,
    )
)

nombre_disponible = (
    test[
        "benchmark_dernier_jour_complet"
    ]
    .notna()
    .sum()
)

print(
    "Observations disponibles : "
    f"{nombre_disponible:,} / "
    f"{len(test):,}"
)


# ============================================================
# 17. TÉLÉCHARGEMENT RTE
# ============================================================

df_rte = (
    telecharger_prevision_rte()
)


# ============================================================
# 18. ALIGNEMENT DE LA PRÉVISION RTE
# ============================================================

print(
    "\nAlignement des prévisions "
    "RTE avec les 24 heures de test..."
)

# ------------------------------------------------------------
# Le dataset de l'étape 3 fournit déjà :
#
#     date_cible
#     heure_cible
#
# On construit directement le timestamp local.
# ------------------------------------------------------------

test[
    "timestamp_cible"
] = (
    pd.to_datetime(
        test["date_cible"]
    )
    + pd.to_timedelta(
        test["heure_cible"].astype(int),
        unit="h",
    )
)

test[
    "timestamp_cible"
] = (
    test[
        "timestamp_cible"
    ]
    .dt.tz_localize(
        "Europe/Paris",
        ambiguous="NaT",
        nonexistent="NaT",
    )
)

# ------------------------------------------------------------
# Mapping RTE :
#
# timestamp local -> prevision_j1
# ------------------------------------------------------------

rte_mapping = dict(
    zip(
        df_rte[
            "date_heure"
        ],
        df_rte[
            "prevision_j1"
        ],
    )
)

test[
    "prediction_rte_j1"
] = test[
    "timestamp_cible"
].map(
    rte_mapping
)

nombre_rte = (
    test[
        "prediction_rte_j1"
    ]
    .notna()
    .sum()
)

print(
    "Prévisions RTE disponibles : "
    f"{nombre_rte:,} / "
    f"{len(test):,} heures"
)


# ============================================================
# 19. MODÈLES À COMPARER
# ============================================================

MODELES_A_COMPARER = [
    "prediction_random_forest",
    "benchmark_j7",
    "benchmark_j1an",
    "benchmark_moyenne",
    "benchmark_dernier_jour_complet",
    "prediction_rte_j1",
]


# ============================================================
# 20. MÉTRIQUES HORAIRES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "MÉTRIQUES HORAIRES"
)

print(
    "=" * 70
)

metrics_horaires = (
    calculer_metrics_horaires(
        test,
        MODELES_A_COMPARER,
    )
)

print(
    metrics_horaires.to_string(
        index=False
    )
)


# ============================================================
# 21. MÉTRIQUES JOURNALIÈRES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "MÉTRIQUES JOURNALIÈRES"
)

print(
    "=" * 70
)

(
    metrics_journalieres,
    resultats_journaliers,
) = (
    calculer_metrics_journalieres(
        test,
        MODELES_A_COMPARER,
    )
)

print(
    metrics_journalieres.to_string(
        index=False
    )
)


# ============================================================
# 22. MÉTRIQUES SUR LES PICS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "MÉTRIQUES SUR LES PICS"
)

print(
    "=" * 70
)

(
    metrics_pics,
    details_pics,
) = calculer_metrics_pics(
    test,
    MODELES_A_COMPARER,
)

if len(metrics_pics) > 0:

    print(
        metrics_pics.to_string(
            index=False
        )
    )

else:

    print(
        "Aucune métrique de pic disponible."
    )


# ============================================================
# 23. PIRES JOURNÉES DU RANDOM FOREST
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PIRES JOURNÉES DU RANDOM FOREST"
)

print(
    "=" * 70
)

rf = test[
    [
        "origine_prevision",
        "date_cible",
        TARGET,
        "prediction_random_forest",
    ]
].copy()

rf["erreur"] = (
    rf[
        "prediction_random_forest"
    ]
    - rf[
        TARGET
    ]
)

rf["erreur_absolue"] = (
    rf["erreur"].abs()
)

pires_jours = (
    rf
    .groupby(
        "date_cible"
    )
    .agg(
        MAE_jour_MW=(
            "erreur_absolue",
            "mean",
        ),
        RMSE_jour_MW=(
            "erreur",
            lambda x: np.sqrt(
                np.mean(
                    x ** 2
                )
            ),
        ),
        erreur_totale_MWh=(
            "erreur",
            "sum",
        ),
        erreur_totale_absolue_MWh=(
            "erreur",
            lambda x: abs(
                x.sum()
            ),
        ),
    )
    .reset_index()
    .sort_values(
        "MAE_jour_MW",
        ascending=False,
    )
)

print(
    pires_jours.head(
        10
    ).to_string(
        index=False
    )
)


# ============================================================
# 24. SAUVEGARDES
# ============================================================

print(
    "\nSauvegarde des résultats..."
)

test.to_csv(
    RESULTATS_HORAIRES,
    sep=";",
    index=False,
)

resultats_journaliers.to_csv(
    RESULTATS_JOURNALIERS,
    sep=";",
    index=False,
)

metrics_horaires.to_csv(
    COMPARAISON_MODELES,
    sep=";",
    index=False,
)

if len(details_pics) > 0:

    details_pics.to_csv(
        RESULTATS_PICS,
        sep=";",
        index=False,
    )

pires_jours.to_csv(
    PIRES_JOURS,
    sep=";",
    index=False,
)


# ============================================================
# 25. RAPPORT TEXTE
# ============================================================

with open(
    RAPPORT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "ÉVALUATION DU MODÈLE DE PRÉVISION\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        "Période TEST : "
        f"{test['origine_prevision'].min().date()} "
        "-> "
        f"{test['origine_prevision'].max().date()}\n\n"
    )

    f.write(
        "Variables utilisées par le Random Forest :\n"
    )

    for feature in FEATURES:

        f.write(
            f"  - {feature}\n"
        )

    f.write(
        "\n"
    )

    f.write(
        "La prévision RTE prevision_j1 est utilisée "
        "uniquement comme benchmark externe.\n"
    )

    f.write(
        "Elle n'est pas utilisée comme variable "
        "d'entrée du Random Forest.\n\n"
    )

    f.write(
        "MÉTRIQUES HORAIRES\n"
        + "-" * 70
        + "\n"
    )

    f.write(
        metrics_horaires.to_string(
            index=False
        )
    )

    f.write(
        "\n\nMÉTRIQUES JOURNALIÈRES\n"
        + "-" * 70
        + "\n"
    )

    f.write(
        metrics_journalieres.to_string(
            index=False
        )
    )

    f.write(
        "\n\nMÉTRIQUES PICS\n"
        + "-" * 70
        + "\n"
    )

    if len(metrics_pics) > 0:

        f.write(
            metrics_pics.to_string(
                index=False
            )
        )

    else:

        f.write(
            "Aucune métrique disponible."
        )

    f.write(
        "\n\nPIRES JOURNÉES\n"
        + "-" * 70
        + "\n"
    )

    f.write(
        pires_jours.head(
            10
        ).to_string(
            index=False
        )
    )


# ============================================================
# 26. FIN
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ÉVALUATION TERMINÉE"
)

print(
    "=" * 70
)

print(
    "\nFichiers produits :"
)

print(
    f"  - {RESULTATS_HORAIRES}"
)

print(
    f"  - {RESULTATS_JOURNALIERS}"
)

print(
    f"  - {COMPARAISON_MODELES}"
)

print(
    f"  - {RESULTATS_PICS}"
)

print(
    f"  - {PIRES_JOURS}"
)

print(
    f"  - {RAPPORT}"
)
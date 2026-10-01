"""
04_modele.py

Entraînement du modèle de prévision de consommation électrique.

OBJECTIF
--------
Utiliser le dataset créé par 03_creation_dataset.py pour entraîner
un seul modèle capable de prévoir la consommation de chaque heure
du jour J+1.

VARIABLES EXPLICATIVES
---------------------
- consommation_j7
- consommation_j1an
- heure
- jour_semaine
- mois
- weekend

VARIABLE CIBLE
--------------
- consommation_cible

IMPORTANT
---------
- Pas de séparation aléatoire.
- Séparation chronologique.
- Les observations futures ne sont jamais utilisées pour entraîner
  le modèle évalué sur une période antérieure.
- Un seul modèle est utilisé pour toutes les heures.
- Aucun "feature engineering" utilisant la cible future.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# ============================================================
# 1. PARAMÈTRES
# ============================================================

INPUT_FILE = Path(
    "data/processed/dataset_prevision.csv"
)

MODEL_DIR = Path(
    "models"
)

MODEL_FILE = MODEL_DIR / "modele_random_forest.joblib"

SPLIT_REPORT_FILE = Path(
    "data/processed/rapport_modele.txt"
)

# Proportions chronologiques.
#
# 70 % -> apprentissage
# 15 % -> validation
# 15 % -> test
#
# Le découpage est réalisé sur les DATES D'ORIGINE de prévision,
# pas sur les lignes individuelles.

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# 2. CRÉATION DES DOSSIERS
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SPLIT_REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. CHARGEMENT DU DATASET
# ============================================================

print("=" * 70)
print("ÉTAPE 4 - ENTRAÎNEMENT DU MODÈLE")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_FILE}\n"
        "Lance d'abord 03_creation_dataset.py."
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

FEATURES = [
    "consommation_j7",
    "consommation_j1an",
    "heure",
    "jour_semaine",
    "mois",
    "weekend",
]

TARGET = "consommation_cible"

REQUIRED_COLUMNS = (
    FEATURES
    + [
        TARGET,
        "origine_prevision",
        "date_cible",
        "heure_cible",
    ]
)

missing_columns = [
    col
    for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Colonnes manquantes : "
        + ", ".join(missing_columns)
    )


# ============================================================
# 5. CONVERSION DES DATES
# ============================================================

df["origine_prevision"] = pd.to_datetime(
    df["origine_prevision"],
    errors="coerce"
)

df["date_cible"] = pd.to_datetime(
    df["date_cible"],
    errors="coerce"
)

if (
    df["origine_prevision"].isna().any()
    or df["date_cible"].isna().any()
):
    raise ValueError(
        "Des dates d'origine ou de cible sont invalides."
    )


# ============================================================
# 6. TRI CHRONOLOGIQUE
# ============================================================

df = df.sort_values(
    [
        "origine_prevision",
        "heure_cible",
    ]
).reset_index(drop=True)


# ============================================================
# 7. VÉRIFICATION DU NOMBRE D'HEURES PAR ORIGINE
# ============================================================

lignes_par_origine = (
    df.groupby("origine_prevision")
    .size()
)

if not (
    lignes_par_origine == 24
).all():

    mauvaises_origines = lignes_par_origine[
        lignes_par_origine != 24
    ]

    raise ValueError(
        "Certaines origines ne contiennent pas exactement "
        "24 heures."
    )

print(
    f"Nombre d'origines de prévision : "
    f"{len(lignes_par_origine):,}"
)


# ============================================================
# 8. LISTE CHRONOLOGIQUE DES ORIGINES
# ============================================================

origines = (
    df["origine_prevision"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

n_origines = len(origines)

if n_origines < 10:
    raise ValueError(
        "Pas suffisamment d'origines de prévision "
        "pour effectuer un découpage train/validation/test."
    )


# ============================================================
# 9. DÉCOUPAGE CHRONOLOGIQUE
# ============================================================

"""
IMPORTANT :

On ne fait surtout PAS :

    train_test_split(..., shuffle=True)

car cela mélangerait le passé et le futur.

Nous faisons :

    PASSÉ ------------------------------> FUTUR

    |--------- TRAIN --------|-- VAL --|-- TEST --|

Le modèle est donc entraîné uniquement sur les origines
les plus anciennes.
"""

train_end = int(
    n_origines * TRAIN_RATIO
)

validation_end = int(
    n_origines
    * (TRAIN_RATIO + VALIDATION_RATIO)
)

train_origines = origines[
    :train_end
]

validation_origines = origines[
    train_end:validation_end
]

test_origines = origines[
    validation_end:
]


# ============================================================
# 10. CONSTRUCTION DES TROIS DATASETS
# ============================================================

train_df = df[
    df["origine_prevision"]
    .isin(train_origines)
].copy()

validation_df = df[
    df["origine_prevision"]
    .isin(validation_origines)
].copy()

test_df = df[
    df["origine_prevision"]
    .isin(test_origines)
].copy()


# ============================================================
# 11. VÉRIFICATION DU DÉCOUPAGE
# ============================================================

print("\n" + "=" * 70)
print("DÉCOUPAGE CHRONOLOGIQUE")
print("=" * 70)

print(
    f"TRAIN      : {len(train_origines):,} journées"
)

print(
    f"VALIDATION : {len(validation_origines):,} journées"
)

print(
    f"TEST       : {len(test_origines):,} journées"
)

print()

print(
    f"TRAIN      : {len(train_df):,} lignes"
)

print(
    f"VALIDATION : {len(validation_df):,} lignes"
)

print(
    f"TEST       : {len(test_df):,} lignes"
)


# ============================================================
# 12. VÉRIFICATION CHRONOLOGIQUE CRITIQUE
# ============================================================

train_max = train_df[
    "origine_prevision"
].max()

validation_min = validation_df[
    "origine_prevision"
].min()

validation_max = validation_df[
    "origine_prevision"
].max()

test_min = test_df[
    "origine_prevision"
].min()

if not train_max < validation_min:
    raise ValueError(
        "ERREUR DE FUITE TEMPORELLE : "
        "TRAIN chevauche VALIDATION."
    )

if not validation_max < test_min:
    raise ValueError(
        "ERREUR DE FUITE TEMPORELLE : "
        "VALIDATION chevauche TEST."
    )

print(
    "\n✓ Séparation chronologique correcte."
)


# ============================================================
# 13. AFFICHAGE DES PÉRIODES
# ============================================================

print("\nPériodes utilisées :")

print(
    f"TRAIN : "
    f"{train_df['origine_prevision'].min().date()} "
    f"-> "
    f"{train_df['origine_prevision'].max().date()}"
)

print(
    f"VALIDATION : "
    f"{validation_df['origine_prevision'].min().date()} "
    f"-> "
    f"{validation_df['origine_prevision'].max().date()}"
)

print(
    f"TEST : "
    f"{test_df['origine_prevision'].min().date()} "
    f"-> "
    f"{test_df['origine_prevision'].max().date()}"
)


# ============================================================
# 14. CONSTRUCTION DE X ET y
# ============================================================

X_train = train_df[
    FEATURES
].copy()

y_train = train_df[
    TARGET
].copy()

X_validation = validation_df[
    FEATURES
].copy()

y_validation = validation_df[
    TARGET
].copy()

X_test = test_df[
    FEATURES
].copy()

y_test = test_df[
    TARGET
].copy()


# ============================================================
# 15. VÉRIFICATION DES VALEURS MANQUANTES
# ============================================================

for name, data in [
    ("X_train", X_train),
    ("X_validation", X_validation),
    ("X_test", X_test),
    ("y_train", y_train),
    ("y_validation", y_validation),
    ("y_test", y_test),
]:

    if data.isna().any().any():

        raise ValueError(
            f"Valeurs manquantes détectées dans {name}."
        )

print(
    "\n✓ Aucune valeur manquante dans X ou y."
)


# ============================================================
# 16. VÉRIFICATION DES VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("VARIABLES DU MODÈLE")
print("=" * 70)

for feature in FEATURES:
    print(f"  X -> {feature}")

print(
    f"\n  y -> {TARGET}"
)


# ============================================================
# 17. MODÈLE
# ============================================================

"""
UN SEUL MODÈLE pour toutes les heures.

La variable "heure" indique au modèle si la ligne correspond
par exemple à 03:00, 07:00, 14:00, etc.

Nous n'entraînons donc pas :

    modèle_00h
    modèle_01h
    modèle_02h
    ...

mais un seul modèle :

    modèle_global
"""

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# 18. ENTRAÎNEMENT
# ============================================================

print("\n" + "=" * 70)
print("ENTRAÎNEMENT")
print("=" * 70)

print(
    "Entraînement du Random Forest..."
)

model.fit(
    X_train,
    y_train
)

print(
    "✓ Modèle entraîné."
)


# ============================================================
# 19. PRÉDICTION VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("ÉVALUATION SUR LA VALIDATION")
print("=" * 70)

prediction_validation = model.predict(
    X_validation
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

print(
    f"MAE validation  : {mae_validation:.2f} MW"
)

print(
    f"RMSE validation : {rmse_validation:.2f} MW"
)


# ============================================================
# 20. PRÉDICTION TEST
# ============================================================

print("\n" + "=" * 70)
print("ÉVALUATION SUR LE TEST")
print("=" * 70)

prediction_test = model.predict(
    X_test
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
    f"MAE test  : {mae_test:.2f} MW"
)

print(
    f"RMSE test : {rmse_test:.2f} MW"
)


# ============================================================
# 21. IMPORTANCE DES VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANCE DES VARIABLES")
print("=" * 70)

feature_importance = pd.DataFrame({
    "variable": FEATURES,
    "importance": model.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

for _, row in feature_importance.iterrows():

    print(
        f"{row['variable']:25s} "
        f"{row['importance']:.4f}"
    )


# ============================================================
# 22. SAUVEGARDE DU MODÈLE
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"\nModèle sauvegardé : {MODEL_FILE}"
)


# ============================================================
# 23. SAUVEGARDE DES INFORMATIONS DU MODÈLE
# ============================================================

MODEL_INFO_FILE = MODEL_DIR / "modele_information.txt"

with open(
    MODEL_INFO_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "INFORMATIONS DU MODÈLE\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Modèle : RandomForestRegressor\n"
    )

    f.write(
        "Nombre d'arbres : 300\n"
    )

    f.write(
        "min_samples_leaf : 2\n"
    )

    f.write(
        "max_features : sqrt\n"
    )

    f.write(
        "random_state : 42\n"
    )

    f.write(
        "\nVARIABLES EXPLICATIVES\n"
    )

    for feature in FEATURES:
        f.write(
            f"- {feature}\n"
        )

    f.write(
        f"\nVARIABLE CIBLE\n"
        f"- {TARGET}\n"
    )

    f.write(
        "\nDÉCOUPAGE\n"
    )

    f.write(
        f"Train : {train_df['origine_prevision'].min()} "
        f"-> {train_df['origine_prevision'].max()}\n"
    )

    f.write(
        f"Validation : "
        f"{validation_df['origine_prevision'].min()} "
        f"-> {validation_df['origine_prevision'].max()}\n"
    )

    f.write(
        f"Test : {test_df['origine_prevision'].min()} "
        f"-> {test_df['origine_prevision'].max()}\n"
    )

    f.write(
        "\nPERFORMANCES\n"
    )

    f.write(
        f"MAE validation : "
        f"{mae_validation:.4f} MW\n"
    )

    f.write(
        f"RMSE validation : "
        f"{rmse_validation:.4f} MW\n"
    )

    f.write(
        f"MAE test : "
        f"{mae_test:.4f} MW\n"
    )

    f.write(
        f"RMSE test : "
        f"{rmse_test:.4f} MW\n"
    )

    f.write(
        "\nIMPORTANCE DES VARIABLES\n"
    )

    for _, row in feature_importance.iterrows():

        f.write(
            f"{row['variable']} : "
            f"{row['importance']:.6f}\n"
        )


# ============================================================
# 24. RAPPORT DE DÉCOUPAGE
# ============================================================

with open(
    SPLIT_REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RAPPORT DE L'ÉTAPE 4 - MODÈLE\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "PRINCIPE\n"
    )

    f.write(
        "--------\n"
    )

    f.write(
        "Un seul modèle Random Forest est utilisé pour "
        "les 24 heures.\n"
    )

    f.write(
        "La variable heure permet au modèle de distinguer "
        "les différentes heures de la journée.\n\n"
    )

    f.write(
        "VARIABLES\n"
    )

    f.write(
        "---------\n"
    )

    for feature in FEATURES:
        f.write(
            f"- {feature}\n"
        )

    f.write(
        f"\nCible : {TARGET}\n\n"
    )

    f.write(
        "DÉCOUPAGE CHRONOLOGIQUE\n"
    )

    f.write(
        "-----------------------\n"
    )

    f.write(
        f"Train : {train_df['origine_prevision'].min()} "
        f"-> {train_df['origine_prevision'].max()}\n"
    )

    f.write(
        f"Validation : "
        f"{validation_df['origine_prevision'].min()} "
        f"-> {validation_df['origine_prevision'].max()}\n"
    )

    f.write(
        f"Test : {test_df['origine_prevision'].min()} "
        f"-> {test_df['origine_prevision'].max()}\n\n"
    )

    f.write(
        "PERFORMANCES\n"
    )

    f.write(
        f"MAE validation : "
        f"{mae_validation:.4f} MW\n"
    )

    f.write(
        f"RMSE validation : "
        f"{rmse_validation:.4f} MW\n"
    )

    f.write(
        f"MAE test : "
        f"{mae_test:.4f} MW\n"
    )

    f.write(
        f"RMSE test : "
        f"{rmse_test:.4f} MW\n"
    )


# ============================================================
# 25. FIN
# ============================================================

print("\n" + "=" * 70)
print("ÉTAPE 4 TERMINÉE")
print("=" * 70)

print(
    f"Modèle : {MODEL_FILE}"
)

print(
    f"Rapport : {MODEL_INFO_FILE}"
)

print(
    "\nLe modèle est prêt pour l'étape 5 : évaluation détaillée."
)
# -*- coding: utf-8 -*-

"""
08_xgboost_pre-opti.py
Modèle XGBoost avant qu'on essaie de l'optimiser pour avoir les meilleurs hyperparamètres.

Objectif
--------
Tester un modèle XGBoost pour la prévision de consommation
électrique avec les variables météo enrichies.

Ce script est autonome : il ne modifie ni le Random Forest
(06_modele_meteo.py) ni les fichiers associés.

Entrée :
    data/processed/dataset_prevision.csv
    data/processed/meteo_france_nationale_journaliere.csv

Sortie :
    models/modele_xgboost_meteo.joblib
    results/predictions_xgboost.csv
    results/importances_xgboost.csv
    results/resultat_xgboost.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PREVISION = DATA_DIR / "dataset_prevision.csv"
DATASET_METEO = DATA_DIR / "meteo_france_nationale_journaliere.csv"

MODEL_FILE = MODELS_DIR / "modele_xgboost_meteo.joblib"
RAPPORT_FILE = RESULTS_DIR / "resultat_xgboost.txt"
PREDICTIONS_FILE = RESULTS_DIR / "predictions_xgboost.csv"
IMPORTANCES_FILE = RESULTS_DIR / "importances_xgboost.csv"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


# ============================================================
# 2. FEATURES (identiques au Random Forest v5)
# ============================================================

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

CIBLE = "consommation_cible"


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

def charger_donnees():

    print("\nChargement du dataset de prévision...")
    df = pd.read_csv(DATASET_PREVISION, sep=";", low_memory=False)
    print(f"  Lignes : {len(df):,}")

    print("\nChargement de la météo nationale...")
    meteo = pd.read_csv(DATASET_METEO, sep=";", low_memory=False)
    print(f"  Lignes : {len(meteo):,}")

    df["origine_prevision"] = pd.to_datetime(
        df["origine_prevision"], errors="coerce"
    )
    df["date_cible"] = pd.to_datetime(
        df["date_cible"], errors="coerce"
    )
    meteo["date"] = pd.to_datetime(
        meteo["date"], errors="coerce"
    )

    if df["origine_prevision"].isna().any():
        raise ValueError("Dates invalides dans origine_prevision.")

    return df, meteo


def joindre_meteo(df, meteo):

    print("\nJointure météo / dataset...")

    colonnes_meteo = [
        c for c in meteo.columns
        if c not in ["AAAAMMJJ", "date"]
    ]

    df = df.merge(
        meteo[["date"] + colonnes_meteo],
        left_on="origine_prevision",
        right_on="date",
        how="left"
    )

    df = df.drop(columns=["date"])

    print(f"  Lignes après jointure : {len(df):,}")

    return df


# ============================================================
# 4. SPLIT CHRONOLOGIQUE (identique à 06)
# ============================================================

def split_chronologique(df):

    origines = (
        df["origine_prevision"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n = len(origines)

    n_train = int(n * TRAIN_RATIO)
    n_validation = int(n * VALIDATION_RATIO)

    train_origines = origines[:n_train]
    validation_origines = origines[n_train:n_train + n_validation]
    test_origines = origines[n_train + n_validation:]

    train = df[df["origine_prevision"].isin(train_origines)].copy()
    validation = df[df["origine_prevision"].isin(validation_origines)].copy()
    test = df[df["origine_prevision"].isin(test_origines)].copy()

    print("\nDÉCOUPAGE CHRONOLOGIQUE")
    print("-" * 70)
    print(
        f"TRAIN      : {len(train_origines):,} jours | "
        f"{train_origines.min().date()} -> "
        f"{train_origines.max().date()}"
    )
    print(
        f"VALIDATION : {len(validation_origines):,} jours | "
        f"{validation_origines.min().date()} -> "
        f"{validation_origines.max().date()}"
    )
    print(
        f"TEST       : {len(test_origines):,} jours | "
        f"{test_origines.min().date()} -> "
        f"{test_origines.max().date()}"
    )

    return train, validation, test


# ============================================================
# 5. ENTRAÎNEMENT XGBOOST
# ============================================================

def entrainer_xgboost(train, validation, test):
    """
    Entraîne XGBoost avec early stopping sur la validation.
    """

    # --------------------------------------------------------
    # Suppression des NaN
    # --------------------------------------------------------

    train_clean = train.dropna(
        subset=FEATURES + [CIBLE]
    ).copy()

    validation_clean = validation.dropna(
        subset=FEATURES + [CIBLE]
    ).copy()

    test_clean = test.dropna(
        subset=FEATURES + [CIBLE]
    ).copy()

    print("\nLignes après dropna :")
    print(f"  TRAIN      : {len(train_clean):,}")
    print(f"  VALIDATION : {len(validation_clean):,}")
    print(f"  TEST       : {len(test_clean):,}")

    X_train = train_clean[FEATURES]
    y_train = train_clean[CIBLE]

    X_validation = validation_clean[FEATURES]
    y_validation = validation_clean[CIBLE]

    X_test = test_clean[FEATURES]
    y_test = test_clean[CIBLE]

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    print("\nEntraînement du XGBoost...")

    modele = XGBRegressor(
        # Nombre d'arbres et complexité
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.03,

        # Régularisation
        subsample=0.6,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,

        # Critère
        objective="reg:squarederror",
        eval_metric="mae",

        # Reproductibilité
        random_state=RANDOM_STATE,
        n_jobs=-1,

        # Early stopping
        early_stopping_rounds=100,
    )

    modele.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_validation, y_validation),
        ],
        verbose=False,
    )

    print(f"  Meilleure itération : {modele.best_iteration}")
    print(f"  Meilleur score     : {modele.best_score:.4f}")

    # --------------------------------------------------------
    # Évaluation
    # --------------------------------------------------------

    pred_val = modele.predict(X_validation)
    pred_test = modele.predict(X_test)

    mae_val = mean_absolute_error(y_validation, pred_val)
    rmse_val = np.sqrt(
        mean_squared_error(y_validation, pred_val)
    )

    mae_test = mean_absolute_error(y_test, pred_test)
    rmse_test = np.sqrt(
        mean_squared_error(y_test, pred_test)
    )

    biais_test = np.mean(pred_test - y_test.values)

    print("\nRÉSULTATS XGBOOST")
    print("-" * 70)
    print(f"Validation MAE  : {mae_val:.2f} MW")
    print(f"Validation RMSE : {rmse_val:.2f} MW")
    print(f"Test MAE        : {mae_test:.2f} MW")
    print(f"Test RMSE       : {rmse_test:.2f} MW")
    print(f"Test biais      : {biais_test:+.2f} MW")

    # --------------------------------------------------------
    # Importance des variables
    # --------------------------------------------------------

    importances = pd.DataFrame({
        "variable": FEATURES,
        "importance": modele.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nIMPORTANCE DES VARIABLES")
    print(importances.to_string(index=False))

    # --------------------------------------------------------
    # Prédictions alignées
    # --------------------------------------------------------

    predictions = test_clean[
        [
            "origine_prevision",
            "date_cible",
            "heure_cible",
            CIBLE,
        ]
    ].copy()

    predictions["prediction_avec_meteo"] = pred_test

    return {
        "modele": modele,
        "importances": importances,
        "mae_val": mae_val,
        "rmse_val": rmse_val,
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "biais_test": biais_test,
        "predictions": predictions,
    }


# ============================================================
# 6. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("MODÈLE XGBOOST AVEC MÉTÉO ENRICHIE")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    df, meteo = charger_donnees()

    df = joindre_meteo(df, meteo)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, validation, test = split_chronologique(df)

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    resultats = entrainer_xgboost(train, validation, test)

    # --------------------------------------------------------
    # Sauvegardes
    # --------------------------------------------------------

    joblib.dump(resultats["modele"], MODEL_FILE)
    print(f"\nModèle sauvegardé : {MODEL_FILE}")

    resultats["predictions"].to_csv(
        PREDICTIONS_FILE,
        sep=";",
        index=False,
        encoding="utf-8",
    )
    print(f"Prédictions sauvegardées : {PREDICTIONS_FILE}")

    resultats["importances"].to_csv(
        IMPORTANCES_FILE,
        sep=";",
        index=False,
        encoding="utf-8",
    )
    print(f"Importances sauvegardées : {IMPORTANCES_FILE}")

    # --------------------------------------------------------
    # Rapport texte
    # --------------------------------------------------------

    with open(RAPPORT_FILE, "w", encoding="utf-8") as f:

        f.write("RAPPORT DU MODÈLE XGBOOST\n")
        f.write("=" * 70 + "\n\n")

        f.write("PARAMÈTRES XGBOOST\n")
        f.write("-" * 70 + "\n")
        f.write(f"n_estimators        : 2000\n")
        f.write(f"max_depth           : 8\n")
        f.write(f"learning_rate       : 0.05\n")
        f.write(f"subsample           : 0.8\n")
        f.write(f"colsample_bytree    : 0.8\n")
        f.write(f"min_child_weight    : 5\n")
        f.write(f"gamma               : 0.1\n")
        f.write(f"reg_alpha           : 0.1\n")
        f.write(f"reg_lambda          : 1.0\n")
        f.write(f"early_stopping      : 50\n\n")

        f.write(f"Meilleure itération : "
                f"{resultats['modele'].best_iteration}\n\n")

        f.write("FEATURES\n")
        f.write("-" * 70 + "\n")
        for feat in FEATURES:
            f.write(f"  - {feat}\n")

        f.write("\nRÉSULTATS\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Validation MAE  : {resultats['mae_val']:.2f} MW\n"
        )
        f.write(
            f"Validation RMSE : {resultats['rmse_val']:.2f} MW\n"
        )
        f.write(
            f"Test MAE        : {resultats['mae_test']:.2f} MW\n"
        )
        f.write(
            f"Test RMSE       : {resultats['rmse_test']:.2f} MW\n"
        )
        f.write(
            f"Test biais      : {resultats['biais_test']:+.2f} MW\n"
        )

        f.write("\nIMPORTANCE DES VARIABLES\n")
        f.write("-" * 70 + "\n")
        f.write(resultats["importances"].to_string(index=False))

    print(f"Rapport sauvegardé : {RAPPORT_FILE}")

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()
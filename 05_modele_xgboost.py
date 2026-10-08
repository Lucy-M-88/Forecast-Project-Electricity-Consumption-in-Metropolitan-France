# -*- coding: utf-8 -*-

"""
06_modele_meteo.py

Objectif
--------
Entraîner un modèle XGBoost de prévision de consommation
électrique avec les variables météo enrichies.

Le modèle utilise les hyperparamètres optimaux trouvés
lors de l'étude d'optimisation (09_optimisation_xgboost.py).

Configuration optimale :
    learning_rate     = 0.03
    max_depth         = 8
    min_child_weight  = 5
    subsample         = 0.6
    colsample_bytree  = 0.8
    gamma             = 0.1
    reg_alpha         = 0.1
    reg_lambda        = 1.0

Entrée :
    data/processed/dataset_prevision.csv
    data/processed/meteo_france_nationale_journaliere.csv

Sortie :
    models/modele_xgboost_meteo.joblib
    results/resultat_modele_meteo.txt
    results/predictions_meteo.csv
    results/importances_meteo.csv
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
RAPPORT_FILE = RESULTS_DIR / "resultat_modele_meteo.txt"
PREDICTIONS_FILE = RESULTS_DIR / "predictions_meteo.csv"
IMPORTANCES_FILE = RESULTS_DIR / "importances_meteo.csv"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

# ------------------------------------------------------------
# Hyperparamètres optimaux XGBoost
# ------------------------------------------------------------

XGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.6,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "early_stopping_rounds": 100,
}


# ============================================================
# 2. FEATURES
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
    """
    Charge le dataset de prévision et la météo nationale.
    """

    print("\nChargement du dataset de prévision...")

    df = pd.read_csv(
        DATASET_PREVISION,
        sep=";",
        low_memory=False,
    )

    print(f"  Lignes : {len(df):,}")

    print("\nChargement de la météo nationale...")

    meteo = pd.read_csv(
        DATASET_METEO,
        sep=";",
        low_memory=False,
    )

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


# ============================================================
# 4. JOINTURE MÉTÉO
# ============================================================

def joindre_meteo(df, meteo):
    """
    Joint la météo du jour J au dataset de prévision.
    """

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
# 5. SPLIT CHRONOLOGIQUE
# ============================================================

def split_chronologique(df):
    """
    Split 70/15/15 sur les origines de prévision.
    """

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
# 6. VÉRIFICATION DES FEATURES
# ============================================================

def verifier_features(df):

    print("\nVÉRIFICATION DES FEATURES")
    print("-" * 70)

    features_manquantes = [
        f for f in FEATURES
        if f not in df.columns
    ]

    if features_manquantes:
        raise ValueError(
            f"Features manquantes : {features_manquantes}"
        )

    print(f"Nombre de features : {len(FEATURES)}")

    for f in FEATURES:
        taux_nan = df[f].isna().mean()
        marqueur = "OK" if taux_nan < 0.05 else "⚠"
        print(f"  [{marqueur}] {f} ({taux_nan:.2%} NaN)")


# ============================================================
# 7. ENTRAÎNEMENT
# ============================================================

def entrainer(train, validation, test):
    """
    Entraîne le XGBoost optimal et évalue.
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
    # Entraînement XGBoost
    # --------------------------------------------------------

    print("\nEntraînement XGBoost (config optimale)...")

    modele = XGBRegressor(**XGB_PARAMS)

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
# 8. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("MODÈLE XGBOOST OPTIMAL")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    df, meteo = charger_donnees()

    df = joindre_meteo(df, meteo)

    verifier_features(df)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, validation, test = split_chronologique(df)

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    resultats = entrainer(train, validation, test)

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
    # Rapport
    # --------------------------------------------------------

    with open(RAPPORT_FILE, "w", encoding="utf-8") as f:

        f.write("RAPPORT DU MODÈLE XGBOOST OPTIMAL\n")
        f.write("=" * 70 + "\n\n")

        f.write("HYPERPARAMÈTRES\n")
        f.write("-" * 70 + "\n")
        for k, v in XGB_PARAMS.items():
            f.write(f"  {k} : {v}\n")

        f.write(f"\nMeilleure itération : "
                f"{resultats['modele'].best_iteration}\n\n")

        f.write("FEATURES\n")
        f.write("-" * 70 + "\n")
        for feat in FEATURES:
            f.write(f"  - {feat}\n")

        f.write("\nDÉCOUPAGE\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"TRAIN      : "
            f"{train['origine_prevision'].min().date()} -> "
            f"{train['origine_prevision'].max().date()}\n"
        )
        f.write(
            f"VALIDATION : "
            f"{validation['origine_prevision'].min().date()} -> "
            f"{validation['origine_prevision'].max().date()}\n"
        )
        f.write(
            f"TEST       : "
            f"{test['origine_prevision'].min().date()} -> "
            f"{test['origine_prevision'].max().date()}\n"
        )

        f.write("\nRÉSULTATS\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Validation MAE  : "
            f"{resultats['mae_val']:.2f} MW\n"
        )
        f.write(
            f"Validation RMSE : "
            f"{resultats['rmse_val']:.2f} MW\n"
        )
        f.write(
            f"Test MAE        : "
            f"{resultats['mae_test']:.2f} MW\n"
        )
        f.write(
            f"Test RMSE       : "
            f"{resultats['rmse_test']:.2f} MW\n"
        )
        f.write(
            f"Test biais      : "
            f"{resultats['biais_test']:+.2f} MW\n"
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
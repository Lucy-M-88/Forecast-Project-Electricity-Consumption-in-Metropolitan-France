# -*- coding: utf-8 -*-

"""
10_ensemble_rf_xgboost.py

Objectif
--------
Tester un ensemble (moyenne pondérée) des prédictions
Random Forest et XGBoost.

Méthodologie :
    1. Entraîner un Random Forest sur le train.
    2. Entraîner un XGBoost sur le train.
    3. Évaluer différentes pondérations alpha sur la validation.
    4. Sélectionner le meilleur alpha.
    5. Évaluer sur le test.

Entrée :
    data/processed/dataset_prevision.csv
    data/processed/meteo_france_nationale_journaliere.csv

Sortie :
    results/ensemble_ponderation.csv
    results/resultat_ensemble.txt
    results/predictions_ensemble.csv
    models/modele_ensemble_rf_xgb.joblib
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
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

MODEL_FILE = MODELS_DIR / "modele_ensemble_rf_xgb.joblib"
RAPPORT_FILE = RESULTS_DIR / "resultat_ensemble.txt"
PREDICTIONS_FILE = RESULTS_DIR / "predictions_ensemble.csv"
PONDERATION_FILE = RESULTS_DIR / "ensemble_ponderation.csv"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

# ------------------------------------------------------------
# Pondérations testées (alpha = poids du RF)
# ------------------------------------------------------------
#
#   alpha = 0.0  -> XGBoost seul
#   alpha = 1.0  -> Random Forest seul
#
# ------------------------------------------------------------

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ============================================================
# 2. FEATURES (v8 = 19 features)
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
# 4. SPLIT CHRONOLOGIQUE
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
# 5. ENTRAÎNEMENT DES DEUX MODÈLES
# ============================================================

def entrainer_modeles(train, validation, test):
    """
    Entraîne les deux modèles sur le train.
    """

    # --------------------------------------------------------
    # Suppression des NaN
    # --------------------------------------------------------

    train_clean = train.dropna(subset=FEATURES + [CIBLE]).copy()
    validation_clean = validation.dropna(subset=FEATURES + [CIBLE]).copy()
    test_clean = test.dropna(subset=FEATURES + [CIBLE]).copy()

    print("\nLignes après dropna :")
    print(f"  TRAIN      : {len(train_clean):,}")
    print(f"  VALIDATION : {len(validation_clean):,}")
    print(f"  TEST       : {len(test_clean):,}")

    X_train = train_clean[FEATURES]
    y_train = train_clean[CIBLE]

    X_val = validation_clean[FEATURES]
    y_val = validation_clean[CIBLE]

    X_test = test_clean[FEATURES]
    y_test = test_clean[CIBLE]

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\nEntraînement Random Forest...")

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    # --------------------------------------------------------
    # XGBoost optimal
    # --------------------------------------------------------

    print("Entraînement XGBoost...")

    xgb = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=5,
        subsample=0.6,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=100,
    )

    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"  Meilleure itération XGBoost : {xgb.best_iteration}")

    # --------------------------------------------------------
    # Prédictions
    # --------------------------------------------------------

    pred_rf_val = rf.predict(X_val)
    pred_xgb_val = xgb.predict(X_val)

    pred_rf_test = rf.predict(X_test)
    pred_xgb_test = xgb.predict(X_test)

    return {
        "rf": rf,
        "xgb": xgb,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "pred_rf_val": pred_rf_val,
        "pred_xgb_val": pred_xgb_val,
        "pred_rf_test": pred_rf_test,
        "pred_xgb_test": pred_xgb_test,
        "test_clean": test_clean,
    }


# ============================================================
# 6. ÉVALUATION DES PONDÉRATIONS
# ============================================================

def evaluer_ponderation(alpha, pred_rf, pred_xgb, y_true):
    """
    Évalue une pondération alpha.
    """

    pred_ensemble = alpha * pred_rf + (1 - alpha) * pred_xgb

    mae = mean_absolute_error(y_true, pred_ensemble)
    rmse = np.sqrt(mean_squared_error(y_true, pred_ensemble))
    biais = np.mean(pred_ensemble - y_true.values)

    return {
        "alpha": alpha,
        "MAE": mae,
        "RMSE": rmse,
        "biais": biais,
    }


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("ENSEMBLE RANDOM FOREST + XGBOOST")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement et préparation
    # --------------------------------------------------------

    df, meteo = charger_donnees()
    df = joindre_meteo(df, meteo)
    train, validation, test = split_chronologique(df)

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    resultats = entrainer_modeles(train, validation, test)

    # ========================================================
    # 1. ÉVALUATION DES PONDÉRATIONS SUR VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("ÉVALUATION DES PONDÉRATIONS SUR LA VALIDATION")
    print("=" * 70)

    resultats_val = []

    for alpha in ALPHAS:

        res = evaluer_ponderation(
            alpha,
            resultats["pred_rf_val"],
            resultats["pred_xgb_val"],
            resultats["y_val"],
        )

        resultats_val.append(res)

    df_val = pd.DataFrame(resultats_val)

    print("\n" + df_val.to_string(index=False))

    # --------------------------------------------------------
    # Meilleure pondération
    # --------------------------------------------------------

    meilleure = df_val.sort_values("MAE").iloc[0]
    alpha_optimal = meilleure["alpha"]

    print(f"\nMeilleure pondération (sur validation) :")
    print(f"  alpha = {alpha_optimal:.2f}")
    print(f"  MAE   = {meilleure['MAE']:.2f} MW")
    print(f"  RMSE  = {meilleure['RMSE']:.2f} MW")

    # ========================================================
    # 2. ÉVALUATION DES PONDÉRATIONS SUR TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("ÉVALUATION DES PONDÉRATIONS SUR LE TEST")
    print("=" * 70)

    resultats_test = []

    for alpha in ALPHAS:

        res = evaluer_ponderation(
            alpha,
            resultats["pred_rf_test"],
            resultats["pred_xgb_test"],
            resultats["y_test"],
        )

        resultats_test.append(res)

    df_test = pd.DataFrame(resultats_test)

    print("\n" + df_test.to_string(index=False))

    # ========================================================
    # 3. COMPARAISON FINALE
    # ========================================================

    print("\n" + "=" * 70)
    print("COMPARAISON FINALE")
    print("=" * 70)

    # XGBoost seul
    mae_xgb = df_test[df_test["alpha"] == 0.0]["MAE"].values[0]
    rmse_xgb = df_test[df_test["alpha"] == 0.0]["RMSE"].values[0]

    # RF seul
    mae_rf = df_test[df_test["alpha"] == 1.0]["MAE"].values[0]
    rmse_rf = df_test[df_test["alpha"] == 1.0]["RMSE"].values[0]

    # Ensemble optimal
    mae_ens = df_test[df_test["alpha"] == alpha_optimal]["MAE"].values[0]
    rmse_ens = df_test[df_test["alpha"] == alpha_optimal]["RMSE"].values[0]

    print(f"\n{'Modèle':<30s} {'MAE (MW)':>12s} {'RMSE (MW)':>12s}")
    print("-" * 70)
    print(f"{'XGBoost seul':<30s} {mae_xgb:>12.2f} {rmse_xgb:>12.2f}")
    print(f"{'Random Forest seul':<30s} {mae_rf:>12.2f} {rmse_rf:>12.2f}")
    print(
        f"{'Ensemble (alpha=' + str(alpha_optimal) + ')':<30s} "
        f"{mae_ens:>12.2f} {rmse_ens:>12.2f}"
    )

    # --------------------------------------------------------
    # Gain de l'ensemble vs XGBoost seul
    # --------------------------------------------------------

    gain_mae = (mae_xgb - mae_ens) / mae_xgb * 100
    gain_rmse = (rmse_xgb - rmse_ens) / rmse_xgb * 100

    print(f"\nGain de l'ensemble vs XGBoost seul :")
    print(f"  MAE  : {gain_mae:+.2f} %")
    print(f"  RMSE : {gain_rmse:+.2f} %")

    # ========================================================
    # 4. SAUVEGARDES
    # ========================================================

    # Tableau des pondérations
    df_val["jeu"] = "validation"
    df_test["jeu"] = "test"

    df_complet = pd.concat([df_val, df_test], ignore_index=True)

    df_complet.to_csv(
        PONDERATION_FILE,
        sep=";",
        index=False,
        encoding="utf-8",
    )
    print(f"\nTableau des pondérations sauvegardé : {PONDERATION_FILE}")

    # Modèle d'ensemble (dict contenant les deux modèles et alpha)
    modele_ensemble = {
        "rf": resultats["rf"],
        "xgb": resultats["xgb"],
        "alpha": alpha_optimal,
        "features": FEATURES,
    }

    joblib.dump(modele_ensemble, MODEL_FILE)
    print(f"Modèle d'ensemble sauvegardé : {MODEL_FILE}")

    # Prédictions sur le test
    predictions = resultats["test_clean"][
        ["origine_prevision", "date_cible", "heure_cible", CIBLE]
    ].copy()

    predictions["prediction_rf"] = resultats["pred_rf_test"]
    predictions["prediction_xgb"] = resultats["pred_xgb_test"]
    predictions["prediction_ensemble"] = (
        alpha_optimal * resultats["pred_rf_test"]
        + (1 - alpha_optimal) * resultats["pred_xgb_test"]
    )
    predictions["prediction_avec_meteo"] = predictions["prediction_ensemble"]

    predictions.to_csv(
        PREDICTIONS_FILE,
        sep=";",
        index=False,
        encoding="utf-8",
    )
    print(f"Prédictions sauvegardées : {PREDICTIONS_FILE}")

    # ========================================================
    # 5. RAPPORT TEXTE
    # ========================================================

    with open(RAPPORT_FILE, "w", encoding="utf-8") as f:

        f.write("RAPPORT ENSEMBLE RF + XGBOOST\n")
        f.write("=" * 70 + "\n\n")

        f.write("FEATURES UTILISÉES\n")
        f.write("-" * 70 + "\n")
        for feat in FEATURES:
            f.write(f"  - {feat}\n")

        f.write("\nPONDÉRATIONS TESTÉES\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {ALPHAS}\n")
        f.write(
            "  (alpha = poids du Random Forest, "
            "1 - alpha = poids du XGBoost)\n\n"
        )

        f.write("RÉSULTATS SUR VALIDATION\n")
        f.write("-" * 70 + "\n")
        f.write(df_val.to_string(index=False))
        f.write("\n\n")

        f.write("RÉSULTATS SUR TEST\n")
        f.write("-" * 70 + "\n")
        f.write(df_test.to_string(index=False))
        f.write("\n\n")

        f.write("COMPARAISON FINALE\n")
        f.write("-" * 70 + "\n")
        f.write(f"XGBoost seul           MAE : {mae_xgb:.2f} MW\n")
        f.write(f"Random Forest seul     MAE : {mae_rf:.2f} MW\n")
        f.write(
            f"Ensemble (alpha={alpha_optimal})   "
            f"MAE : {mae_ens:.2f} MW\n"
        )
        f.write(f"\nGain MAE  : {gain_mae:+.2f} %\n")
        f.write(f"Gain RMSE : {gain_rmse:+.2f} %\n")

    print(f"Rapport sauvegardé : {RAPPORT_FILE}")

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()
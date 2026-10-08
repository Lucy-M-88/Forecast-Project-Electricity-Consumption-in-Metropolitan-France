# -*- coding: utf-8 -*-

"""
09_optimisation_xgboost.py

Objectif
--------
Tester plusieurs configurations d'hyperparamètres XGBoost
pour trouver la meilleure sur la validation.

IMPORTANT :
    - Le test final n'est PAS utilisé pour choisir la config.
    - Le choix se fait uniquement sur la validation.
    - Le test final est évalué uniquement pour la config gagnante.

Sortie :
    results/optimisation_xgboost.csv
    results/meilleure_config_xgboost.txt
"""

from pathlib import Path
import time
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

FICHIER_RESULTATS = RESULTS_DIR / "optimisation_xgboost.csv"
FICHIER_MEILLEURE = RESULTS_DIR / "meilleure_config_xgboost.txt"
MODEL_FILE = MODELS_DIR / "modele_xgboost_optimal.joblib"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


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
# 3. GRILLE D'HYPERPARAMÈTRES
# ============================================================
#
# On teste plusieurs configurations autour de la meilleure
# actuelle (learning_rate=0.03, max_depth=8).
#
# Stratégie :
#   - On explore d'abord l'effet du learning_rate et de
#     max_depth (les plus impactants).
#   - Puis on affine sur d'autres paramètres.
# ============================================================

CONFIGS = [
    # --------------------------------------------------------
    # Référence (v7b)
    # --------------------------------------------------------
    {
        "nom": "ref_lr003_d8",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    # --------------------------------------------------------
    # Variation du learning_rate
    # --------------------------------------------------------
    {
        "nom": "lr001_d8",
        "learning_rate": 0.01,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr002_d8",
        "learning_rate": 0.02,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr005_d8",
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    # --------------------------------------------------------
    # Variation de max_depth (à lr=0.03)
    # --------------------------------------------------------
    {
        "nom": "lr003_d6",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr003_d7",
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr003_d10",
        "learning_rate": 0.03,
        "max_depth": 10,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr003_d12",
        "learning_rate": 0.03,
        "max_depth": 12,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    # --------------------------------------------------------
    # Variation de min_child_weight
    # --------------------------------------------------------
    {
        "nom": "lr003_d8_mcw1",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr003_d8_mcw10",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr003_d8_mcw20",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    # --------------------------------------------------------
    # Variation de subsample / colsample
    # --------------------------------------------------------
    {
        "nom": "lr003_d8_sub06",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr003_d8_sub10",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr003_d8_col06",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr003_d8_col10",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    # --------------------------------------------------------
    # Variation de gamma
    # --------------------------------------------------------
    {
        "nom": "lr003_d8_g0",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "nom": "lr003_d8_g05",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    # --------------------------------------------------------
    # Variation de la régularisation L2
    # --------------------------------------------------------
    {
        "nom": "lr003_d8_l2_5",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
    },
    {
        "nom": "lr003_d8_l1_1",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
    },
]


# ============================================================
# 4. CHARGEMENT DES DONNÉES
# ============================================================

def charger_et_preparer():

    print("\nChargement des données...")

    df = pd.read_csv(DATASET_PREVISION, sep=";", low_memory=False)
    meteo = pd.read_csv(DATASET_METEO, sep=";", low_memory=False)

    df["origine_prevision"] = pd.to_datetime(
        df["origine_prevision"], errors="coerce"
    )
    df["date_cible"] = pd.to_datetime(
        df["date_cible"], errors="coerce"
    )
    meteo["date"] = pd.to_datetime(
        meteo["date"], errors="coerce"
    )

    colonnes_meteo = [
        c for c in meteo.columns
        if c not in ["AAAAMMJJ", "date"]
    ]

    df = df.merge(
        meteo[["date"] + colonnes_meteo],
        left_on="origine_prevision",
        right_on="date",
        how="left"
    ).drop(columns=["date"])

    print(f"  Lignes après jointure : {len(df):,}")

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Suppression des NaN
    # --------------------------------------------------------

    train = train.dropna(subset=FEATURES + [CIBLE]).copy()
    validation = validation.dropna(subset=FEATURES + [CIBLE]).copy()
    test = test.dropna(subset=FEATURES + [CIBLE]).copy()

    print(f"  TRAIN      : {len(train):,}")
    print(f"  VALIDATION : {len(validation):,}")
    print(f"  TEST       : {len(test):,}")

    return train, validation, test


# ============================================================
# 5. ENTRAÎNEMENT ET ÉVALUATION D'UNE CONFIG
# ============================================================

def evaluer_config(config, train, validation):
    """
    Entraîne XGBoost avec la config donnée et évalue sur
    la validation.
    """

    X_train = train[FEATURES]
    y_train = train[CIBLE]

    X_val = validation[FEATURES]
    y_val = validation[CIBLE]

    modele = XGBRegressor(
        n_estimators=5000,
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        gamma=config["gamma"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=100,
    )

    t0 = time.time()

    modele.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    duree = time.time() - t0

    pred_val = modele.predict(X_val)

    mae_val = mean_absolute_error(y_val, pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))

    return {
        "nom": config["nom"],
        "learning_rate": config["learning_rate"],
        "max_depth": config["max_depth"],
        "min_child_weight": config["min_child_weight"],
        "subsample": config["subsample"],
        "colsample_bytree": config["colsample_bytree"],
        "gamma": config["gamma"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "meilleure_iteration": modele.best_iteration,
        "MAE_validation": mae_val,
        "RMSE_validation": rmse_val,
        "duree_s": duree,
        "_modele": modele,
    }


# ============================================================
# 6. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("OPTIMISATION DES HYPERPARAMÈTRES XGBOOST")
    print("=" * 70)

    train, validation, test = charger_et_preparer()

    print(f"\nNombre de configurations à tester : {len(CONFIGS)}")

    # --------------------------------------------------------
    # Évaluation de chaque config
    # --------------------------------------------------------

    resultats = []

    for i, config in enumerate(CONFIGS, start=1):

        print(f"\n[{i}/{len(CONFIGS)}] {config['nom']}")

        resultat = evaluer_config(config, train, validation)

        resultats.append(resultat)

        print(
            f"  MAE val  : {resultat['MAE_validation']:.2f} MW | "
            f"RMSE val : {resultat['RMSE_validation']:.2f} MW | "
            f"iter : {resultat['meilleure_iteration']} | "
            f"temps : {resultat['duree_s']:.1f}s"
        )

    # --------------------------------------------------------
    # Tableau comparatif (sans le modèle)
    # --------------------------------------------------------

    df_resultats = pd.DataFrame(resultats)

    df_affichage = df_resultats.drop(columns=["_modele"])

    df_affichage = df_affichage.sort_values("MAE_validation")

    print("\n" + "=" * 70)
    print("CLASSEMENT DES CONFIGURATIONS (sur validation)")
    print("=" * 70)

    print(df_affichage.to_string(index=False))

    df_affichage.to_csv(
        FICHIER_RESULTATS,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    print(f"\nRésultats sauvegardés : {FICHIER_RESULTATS}")

    # --------------------------------------------------------
    # Sélection de la meilleure config
    # --------------------------------------------------------

    meilleure = df_resultats.sort_values(
        "MAE_validation"
    ).iloc[0]

    print("\n" + "=" * 70)
    print("MEILLEURE CONFIGURATION")
    print("=" * 70)

    for col in df_affichage.columns:
        print(f"  {col} : {meilleure[col]}")

    # --------------------------------------------------------
    # Évaluation finale sur le test
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ÉVALUATION FINALE SUR LE TEST")
    print("=" * 70)

    modele_final = meilleure["_modele"]

    X_test = test[FEATURES]
    y_test = test[CIBLE]

    pred_test = modele_final.predict(X_test)

    mae_test = mean_absolute_error(y_test, pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    biais_test = np.mean(pred_test - y_test.values)

    print(f"Test MAE   : {mae_test:.2f} MW")
    print(f"Test RMSE  : {rmse_test:.2f} MW")
    print(f"Test biais : {biais_test:+.2f} MW")

    # --------------------------------------------------------
    # Sauvegarde du meilleur modèle
    # --------------------------------------------------------

    joblib.dump(modele_final, MODEL_FILE)
    print(f"\nModèle sauvegardé : {MODEL_FILE}")

    # --------------------------------------------------------
    # Rapport
    # --------------------------------------------------------

    with open(FICHIER_MEILLEURE, "w", encoding="utf-8") as f:

        f.write("MEILLEURE CONFIGURATION XGBOOST\n")
        f.write("=" * 70 + "\n\n")

        f.write("Sélection basée sur la MAE de validation.\n\n")

        f.write("HYPERPARAMÈTRES\n")
        f.write("-" * 70 + "\n")
        for col in [
            "nom",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "meilleure_iteration",
        ]:
            f.write(f"  {col} : {meilleure[col]}\n")

        f.write("\nRÉSULTATS VALIDATION\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"  MAE  : {meilleure['MAE_validation']:.2f} MW\n"
        )
        f.write(
            f"  RMSE : {meilleure['RMSE_validation']:.2f} MW\n"
        )

        f.write("\nRÉSULTATS TEST\n")
        f.write("-" * 70 + "\n")
        f.write(f"  MAE   : {mae_test:.2f} MW\n")
        f.write(f"  RMSE  : {rmse_test:.2f} MW\n")
        f.write(f"  Biais : {biais_test:+.2f} MW\n")

    print(f"Rapport sauvegardé : {FICHIER_MEILLEURE}")

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

"""
05_modelisation.py

Entraînement et évaluation des modèles de prévision de
consommation électrique.

Ce script effectue deux étapes :

    ÉTAPE 1 : Entraînement du modèle XGBoost seul
        - Utilise les hyperparamètres optimaux (déjà trouvés).
        - Entraîne sur le train, évalue sur validation et test.
        - Sauvegarde le modèle et les métriques.

    ÉTAPE 2 : Entraînement de l'ensemble RF + XGBoost
        - Entraîne un Random Forest.
        - Combine avec le XGBoost (moyenne pondérée).
        - Optimise alpha sur la validation.
        - Évalue sur le test.
        - Sauvegarde le modèle d'ensemble.

L'évaluation produit :
    - Métriques horaires, journalières, pics.
    - Erreurs par heure, saison, type de jour.
    - Pires journées.
    - Comparaison avec les benchmarks (J-7, J-1 an, moyenne).

Entrée :
    data/processed/dataset_prevision.csv
    data/processed/meteo_france_nationale_journaliere.csv

Sortie :
    models/modele_xgboost_meteo.joblib
    models/modele_ensemble_rf_xgb.joblib
    results/predictions.csv
    results/importances.csv
    results/ensemble_ponderation.csv
    results/evaluation_*.csv
    results/pires_journees.csv
    results/rapport.txt
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

MODEL_XGBOOST = MODELS_DIR / "modele_xgboost_meteo.joblib"
MODEL_ENSEMBLE = MODELS_DIR / "modele_ensemble_rf_xgb.joblib"

FICHIER_PREDICTIONS = RESULTS_DIR / "predictions.csv"
FICHIER_IMPORTANCES = RESULTS_DIR / "importances.csv"
FICHIER_PONDERATION = RESULTS_DIR / "ensemble_ponderation.csv"

FICHIER_HORAIRE = RESULTS_DIR / "evaluation_horaire.csv"
FICHIER_JOURNALIER = RESULTS_DIR / "evaluation_journaliere.csv"
FICHIER_PICS = RESULTS_DIR / "evaluation_pics.csv"
FICHIER_PAR_HEURE = RESULTS_DIR / "evaluation_par_heure.csv"
FICHIER_PAR_SAISON = RESULTS_DIR / "evaluation_par_saison.csv"
FICHIER_PAR_TYPE_JOUR = RESULTS_DIR / "evaluation_par_type_jour.csv"
FICHIER_PIRES_JOURS = RESULTS_DIR / "pires_journees.csv"

FICHIER_RAPPORT = RESULTS_DIR / "rapport.txt"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Hyperparamètres XGBoost (optimaux)
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


# ------------------------------------------------------------
# Pondérations testées pour l'ensemble (alpha = poids RF)
# ------------------------------------------------------------

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ------------------------------------------------------------
# Jours fériés (pour l'analyse par type de jour)
# ------------------------------------------------------------

JOURS_FERIES = {
    "2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08",
    "2024-05-09", "2024-05-20", "2024-07-14", "2024-08-15",
    "2024-11-01", "2024-11-11", "2024-12-25",
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-05-08",
    "2025-05-29", "2025-06-09", "2025-07-14", "2025-08-15",
    "2025-11-01", "2025-11-11", "2025-12-25",
    "2026-01-01", "2026-04-06", "2026-05-01", "2026-05-08",
    "2026-05-14", "2026-05-25", "2026-07-14", "2026-08-15",
    "2026-11-01", "2026-11-11", "2026-12-25",
}


# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def biais(y_true, y_pred):
    return np.mean(y_pred - y_true)


def saison(date):
    mois = date.month
    if mois in [12, 1, 2]:
        return "Hiver"
    if mois in [3, 4, 5]:
        return "Printemps"
    if mois in [6, 7, 8]:
        return "Été"
    return "Automne"


def type_jour(date):
    date_str = date.strftime("%Y-%m-%d")
    if date_str in JOURS_FERIES:
        return "Férié"
    if date.weekday() >= 5:
        return "Weekend"
    return "Ouvré"


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

def charger_donnees():
    """Charge le dataset de prévision et la météo nationale."""

    print("\nChargement du dataset de prévision...")

    df = pd.read_csv(DATASET_PREVISION, sep=";", low_memory=False)
    print(f"  {len(df):,} lignes")

    print("\nChargement de la météo nationale...")

    meteo = pd.read_csv(DATASET_METEO, sep=";", low_memory=False)
    print(f"  {len(meteo):,} lignes")

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
    """Joint la météo du jour J au dataset de prévision."""

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

    print(f"  {len(df):,} lignes après jointure")

    return df


def split_chronologique(df):
    """Split chronologique 70/15/15 sur les origines."""

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


def verifier_features(df):

    print("\nVÉRIFICATION DES FEATURES")
    print("-" * 70)

    features_manquantes = [f for f in FEATURES if f not in df.columns]

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
# 4. ÉTAPE 1 : ENTRAÎNEMENT DU XGBOOST SEUL
# ============================================================

def entrainer_xgboost(train, validation, test):
    """Entraîne XGBoost et évalue."""

    print("\n" + "=" * 70)
    print("ÉTAPE 1 : ENTRAÎNEMENT DU XGBOOST")
    print("=" * 70)

    # Nettoyage
    train_clean = train.dropna(subset=FEATURES + [CIBLE]).copy()
    validation_clean = validation.dropna(subset=FEATURES + [CIBLE]).copy()
    test_clean = test.dropna(subset=FEATURES + [CIBLE]).copy()

    print(f"\nLignes après dropna :")
    print(f"  TRAIN      : {len(train_clean):,}")
    print(f"  VALIDATION : {len(validation_clean):,}")
    print(f"  TEST       : {len(test_clean):,}")

    X_train = train_clean[FEATURES]
    y_train = train_clean[CIBLE]

    X_val = validation_clean[FEATURES]
    y_val = validation_clean[CIBLE]

    X_test = test_clean[FEATURES]
    y_test = test_clean[CIBLE]

    # Entraînement
    print("\nEntraînement XGBoost (config optimale)...")

    modele = XGBRegressor(**XGB_PARAMS)

    modele.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    print(f"  Meilleure itération : {modele.best_iteration}")

    # Évaluation
    pred_val = modele.predict(X_val)
    pred_test = modele.predict(X_test)

    mae_val = mean_absolute_error(y_val, pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))

    mae_test = mean_absolute_error(y_test, pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    biais_test = np.mean(pred_test - y_test.values)

    print("\nRÉSULTATS XGBOOST")
    print("-" * 70)
    print(f"Validation MAE  : {mae_val:.2f} MW")
    print(f"Validation RMSE : {rmse_val:.2f} MW")
    print(f"Test MAE        : {mae_test:.2f} MW")
    print(f"Test RMSE       : {rmse_test:.2f} MW")
    print(f"Test biais      : {biais_test:+.2f} MW")

    # Importances
    importances = pd.DataFrame({
        "variable": FEATURES,
        "importance_xgb": modele.feature_importances_,
    }).sort_values("importance_xgb", ascending=False)

    print("\nIMPORTANCE DES VARIABLES (top 10)")
    print(importances.head(10).to_string(index=False))

    # Prédictions
    predictions = test_clean[
        ["origine_prevision", "date_cible", "heure_cible", CIBLE]
    ].copy()
    predictions["prediction_xgb"] = pred_test

    # Sauvegarde
    joblib.dump(modele, MODEL_XGBOOST)
    print(f"\nModèle sauvegardé : {MODEL_XGBOOST}")

    return {
        "modele": modele,
        "importances": importances,
        "mae_val": mae_val,
        "rmse_val": rmse_val,
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "biais_test": biais_test,
        "predictions": predictions,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "test_clean": test_clean,
    }


# ============================================================
# 5. ÉTAPE 2 : ENSEMBLE RF + XGBOOST
# ============================================================

def entrainer_ensemble(train, validation, resultats_xgb):
    """Entraîne l'ensemble RF + XGBoost et évalue."""

    print("\n" + "=" * 70)
    print("ÉTAPE 2 : ENSEMBLE RF + XGBOOST")
    print("=" * 70)

    # Nettoyage
    train_clean = train.dropna(subset=FEATURES + [CIBLE]).copy()
    validation_clean = validation.dropna(subset=FEATURES + [CIBLE]).copy()

    X_train = train_clean[FEATURES]
    y_train = train_clean[CIBLE]

    X_val = validation_clean[FEATURES]
    y_val = validation_clean[CIBLE]

    # Le test est déjà préparé dans resultats_xgb
    X_test = resultats_xgb["X_test"]
    y_test = resultats_xgb["y_test"]

    # Entraînement Random Forest
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

    # XGBoost est déjà entraîné
    xgb = resultats_xgb["modele"]

    # Prédictions
    pred_rf_val = rf.predict(X_val)
    pred_xgb_val = resultats_xgb["pred_val"]

    pred_rf_test = rf.predict(X_test)
    pred_xgb_test = resultats_xgb["pred_test"]

    # Optimisation de alpha sur validation
    print("\nOptimisation de alpha (validation)...")

    resultats_val = []

    for alpha in ALPHAS:
        pred_ensemble = alpha * pred_rf_val + (1 - alpha) * pred_xgb_val

        resultats_val.append({
            "alpha": alpha,
            "MAE": mean_absolute_error(y_val, pred_ensemble),
            "RMSE": np.sqrt(mean_squared_error(y_val, pred_ensemble)),
            "biais": np.mean(pred_ensemble - y_val.values),
        })

    df_val = pd.DataFrame(resultats_val)
    print("\n" + df_val.to_string(index=False))

    meilleure = df_val.sort_values("MAE").iloc[0]
    alpha_optimal = meilleure["alpha"]

    print(f"\nMeilleur alpha (validation) : {alpha_optimal:.2f}")
    print(f"  MAE  : {meilleure['MAE']:.2f} MW")
    print(f"  RMSE : {meilleure['RMSE']:.2f} MW")

    # Évaluation sur test
    print("\nÉvaluation sur test...")

    resultats_test = []

    for alpha in ALPHAS:
        pred_ensemble = alpha * pred_rf_test + (1 - alpha) * pred_xgb_test

        resultats_test.append({
            "alpha": alpha,
            "MAE": mean_absolute_error(y_test, pred_ensemble),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred_ensemble)),
            "biais": np.mean(pred_ensemble - y_test.values),
        })

    df_test = pd.DataFrame(resultats_test)
    print("\n" + df_test.to_string(index=False))

    # Comparaison finale
    mae_xgb = df_test[df_test["alpha"] == 0.0]["MAE"].values[0]
    rmse_xgb = df_test[df_test["alpha"] == 0.0]["RMSE"].values[0]

    mae_rf = df_test[df_test["alpha"] == 1.0]["MAE"].values[0]
    rmse_rf = df_test[df_test["alpha"] == 1.0]["RMSE"].values[0]

    mae_ens = df_test[df_test["alpha"] == alpha_optimal]["MAE"].values[0]
    rmse_ens = df_test[df_test["alpha"] == alpha_optimal]["RMSE"].values[0]

    print("\n" + "=" * 70)
    print("COMPARAISON FINALE")
    print("=" * 70)
    print(f"\n{'Modèle':<30s} {'MAE (MW)':>12s} {'RMSE (MW)':>12s}")
    print("-" * 70)
    print(f"{'XGBoost seul':<30s} {mae_xgb:>12.2f} {rmse_xgb:>12.2f}")
    print(f"{'Random Forest seul':<30s} {mae_rf:>12.2f} {rmse_rf:>12.2f}")
    print(
        f"{'Ensemble (alpha=' + str(alpha_optimal) + ')':<30s} "
        f"{mae_ens:>12.2f} {rmse_ens:>12.2f}"
    )

    gain_mae = (mae_xgb - mae_ens) / mae_xgb * 100
    gain_rmse = (rmse_xgb - rmse_ens) / rmse_xgb * 100

    print(f"\nGain de l'ensemble vs XGBoost seul :")
    print(f"  MAE  : {gain_mae:+.2f} %")
    print(f"  RMSE : {gain_rmse:+.2f} %")

    # Importances RF (pour compléter le fichier importances)
    importances_rf = pd.DataFrame({
        "variable": FEATURES,
        "importance_rf": rf.feature_importances_,
    })

    # Modèle d'ensemble
    modele_ensemble = {
        "rf": rf,
        "xgb": xgb,
        "alpha": alpha_optimal,
        "features": FEATURES,
    }

    joblib.dump(modele_ensemble, MODEL_ENSEMBLE)
    print(f"\nModèle d'ensemble sauvegardé : {MODEL_ENSEMBLE}")

    # Pondération
    df_val["jeu"] = "validation"
    df_test["jeu"] = "test"
    df_complet = pd.concat([df_val, df_test], ignore_index=True)

    df_complet.to_csv(
        FICHIER_PONDERATION,
        sep=";",
        index=False,
        encoding="utf-8",
    )
    print(f"Tableau des pondérations : {FICHIER_PONDERATION}")

    return {
        "rf": rf,
        "xgb": xgb,
        "alpha_optimal": alpha_optimal,
        "importances_rf": importances_rf,
        "mae_xgb": mae_xgb,
        "mae_rf": mae_rf,
        "mae_ens": mae_ens,
        "rmse_xgb": rmse_xgb,
        "rmse_rf": rmse_rf,
        "rmse_ens": rmse_ens,
        "gain_mae": gain_mae,
        "gain_rmse": gain_rmse,
        "pred_rf_test": pred_rf_test,
        "pred_xgb_test": pred_xgb_test,
    }


# ============================================================
# 6. ÉVALUATION DÉTAILLÉE (sur le modèle d'ensemble)
# ============================================================

def evaluer_modele_final(resultats_xgb, resultats_ensemble):
    """
    Évalue le modèle d'ensemble sur le test et produit toutes
    les métriques et analyses.
    """

    print("\n" + "=" * 70)
    print("ÉVALUATION DÉTAILLÉE DU MODÈLE FINAL")
    print("=" * 70)

    test_clean = resultats_xgb["test_clean"].copy()
    alpha = resultats_ensemble["alpha_optimal"]

    pred_ensemble = (
        alpha * resultats_ensemble["pred_rf_test"]
        + (1 - alpha) * resultats_ensemble["pred_xgb_test"]
    )

    # Ajout des prédictions
    test_clean["prediction_avec_meteo"] = pred_ensemble

    # Benchmarks
    test_clean["benchmark_j7"] = test_clean["consommation_j7"]
    test_clean["benchmark_j1an"] = test_clean["consommation_j1an"]
    test_clean["benchmark_moyenne"] = (
        test_clean["consommation_j7"] + test_clean["consommation_j1an"]
    ) / 2

    # --------------------------------------------------------
    # 1. Métriques horaires
    # --------------------------------------------------------

    print("\n1. MÉTRIQUES HORAIRES")
    print("-" * 70)

    modeles = [
        "prediction_avec_meteo",
        "benchmark_j7",
        "benchmark_j1an",
        "benchmark_moyenne",
    ]

    lignes = []
    for col in modeles:
        sous = test_clean[["consommation_cible", col]].dropna()
        y_true = sous["consommation_cible"].values
        y_pred = sous[col].values
        lignes.append({
            "modele": col,
            "n_heures": len(sous),
            "MAE_MW": mae(y_true, y_pred),
            "RMSE_MW": rmse(y_true, y_pred),
            "biais_MW": biais(y_true, y_pred),
        })

    metriques_h = pd.DataFrame(lignes).sort_values("MAE_MW")
    print(metriques_h.to_string(index=False))

    metriques_h.to_csv(FICHIER_HORAIRE, sep=";", index=False)

    # --------------------------------------------------------
    # 2. Métriques journalières
    # --------------------------------------------------------

    print("\n2. MÉTRIQUES JOURNALIÈRES")
    print("-" * 70)

    agg = {"consommation_cible": "sum"}
    for col in modeles:
        agg[col] = "sum"

    journalier = (
        test_clean.groupby("date_cible").agg(agg).reset_index()
    )

    lignes = []
    for col in modeles:
        sous = journalier[["consommation_cible", col]].dropna()
        erreur = sous[col] - sous["consommation_cible"]
        lignes.append({
            "modele": col,
            "n_jours": len(sous),
            "MAE_total_MWh": np.mean(np.abs(erreur)),
            "RMSE_total_MWh": np.sqrt(np.mean(erreur ** 2)),
            "biais_total_MWh": np.mean(erreur),
        })

    metriques_j = pd.DataFrame(lignes).sort_values("MAE_total_MWh")
    print(metriques_j.to_string(index=False))

    metriques_j.to_csv(FICHIER_JOURNALIER, sep=";", index=False)

    # --------------------------------------------------------
    # 3. Métriques sur les pics
    # --------------------------------------------------------

    print("\n3. MÉTRIQUES SUR LES PICS")
    print("-" * 70)

    lignes = []

    for date_c, groupe in test_clean.groupby("date_cible"):

        groupe_reel = groupe.dropna(subset=["consommation_cible"])
        if len(groupe_reel) != 24:
            continue

        idx_reel = groupe_reel["consommation_cible"].idxmax()
        pic_reel = groupe_reel.loc[idx_reel, "consommation_cible"]
        heure_reel = int(groupe_reel.loc[idx_reel, "heure_cible"])

        for col in modeles:
            groupe_modele = groupe.dropna(subset=["consommation_cible", col])
            if len(groupe_modele) != 24:
                continue

            idx_prevu = groupe_modele[col].idxmax()
            pic_prevu = groupe_modele.loc[idx_prevu, col]
            heure_prevu = int(groupe_modele.loc[idx_prevu, "heure_cible"])

            lignes.append({
                "date_cible": date_c,
                "modele": col,
                "pic_reel_MW": pic_reel,
                "pic_prevu_MW": pic_prevu,
                "erreur_pic_MW": pic_prevu - pic_reel,
                "erreur_pic_abs_MW": abs(pic_prevu - pic_reel),
                "heure_pic_reel": heure_reel,
                "heure_pic_prevu": heure_prevu,
                "ecart_heure_pic": heure_prevu - heure_reel,
                "pic_correct": int(heure_prevu == heure_reel),
            })

    details_pics = pd.DataFrame(lignes)

    if len(details_pics) > 0:
        metriques_p = (
            details_pics.groupby("modele").agg(
                n_jours=("date_cible", "count"),
                MAE_pic_MW=("erreur_pic_abs_MW", "mean"),
                biais_pic_MW=("erreur_pic_MW", "mean"),
                ecart_moyen_heure_pic=(
                    "ecart_heure_pic",
                    lambda x: np.mean(np.abs(x))
                ),
                taux_pic_correct=("pic_correct", "mean"),
            ).reset_index()
        )
        metriques_p["taux_pic_correct"] *= 100
        metriques_p = metriques_p.sort_values("MAE_pic_MW")

        print(metriques_p.to_string(index=False))
        metriques_p.to_csv(FICHIER_PICS, sep=";", index=False)

    # --------------------------------------------------------
    # 4. Analyse par heure
    # --------------------------------------------------------

    print("\n4. ERREUR PAR HEURE")
    print("-" * 70)

    test_clean["erreur"] = (
        test_clean["prediction_avec_meteo"] - test_clean["consommation_cible"]
    )
    test_clean["erreur_abs"] = test_clean["erreur"].abs()

    par_heure = (
        test_clean.groupby("heure_cible").agg(
            MAE_MW=("erreur_abs", "mean"),
            biais_MW=("erreur", "mean"),
            n=("erreur", "count"),
        ).reset_index()
    )

    print(par_heure.to_string(index=False))
    par_heure.to_csv(FICHIER_PAR_HEURE, sep=";", index=False)

    # --------------------------------------------------------
    # 5. Analyse par saison
    # --------------------------------------------------------

    print("\n5. ERREUR PAR SAISON")
    print("-" * 70)

    test_clean["saison"] = test_clean["date_cible"].apply(saison)

    par_saison = (
        test_clean.groupby("saison").agg(
            MAE_MW=("erreur_abs", "mean"),
            RMSE_MW=("erreur", lambda x: np.sqrt(np.mean(x ** 2))),
            biais_MW=("erreur", "mean"),
            n=("erreur", "count"),
        ).reset_index()
    )

    print(par_saison.to_string(index=False))
    par_saison.to_csv(FICHIER_PAR_SAISON, sep=";", index=False)

    # --------------------------------------------------------
    # 6. Analyse par type de jour
    # --------------------------------------------------------

    print("\n6. ERREUR PAR TYPE DE JOUR")
    print("-" * 70)

    test_clean["type_jour"] = test_clean["date_cible"].apply(type_jour)

    par_type = (
        test_clean.groupby("type_jour").agg(
            MAE_MW=("erreur_abs", "mean"),
            RMSE_MW=("erreur", lambda x: np.sqrt(np.mean(x ** 2))),
            biais_MW=("erreur", "mean"),
            n=("erreur", "count"),
        ).reset_index()
    )

    print(par_type.to_string(index=False))
    par_type.to_csv(FICHIER_PAR_TYPE_JOUR, sep=";", index=False)

    # --------------------------------------------------------
    # 7. Pires journées
    # --------------------------------------------------------

    print("\n7. PIRES JOURNÉES")
    print("-" * 70)

    journalier = (
        test_clean.groupby("date_cible").agg(
            MAE_jour_MW=("erreur_abs", "mean"),
            RMSE_jour_MW=("erreur", lambda x: np.sqrt(np.mean(x ** 2))),
            erreur_totale_MWh=("erreur", "sum"),
            erreur_totale_abs_MWh=("erreur", lambda x: abs(x.sum())),
            pic_reel_MW=("consommation_cible", "max"),
            pic_prevu_MW=("prediction_avec_meteo", "max"),
        ).reset_index()
    )

    # Enrichissement météo
    meteo_sub = (
        test_clean[[
            "date_cible", "temperature_moyenne", "temperature_min",
            "temperature_max", "HDD_moyen", "precipitations_totales_mm"
        ]]
        .drop_duplicates(subset=["date_cible"])
    )

    journalier = journalier.merge(meteo_sub, on="date_cible", how="left")

    journalier["type_jour"] = journalier["date_cible"].apply(type_jour)
    journalier["saison"] = journalier["date_cible"].apply(saison)

    pires = (
        journalier
        .sort_values("MAE_jour_MW", ascending=False)
        .head(15)
    )

    print(pires.to_string(index=False))
    pires.to_csv(FICHIER_PIRES_JOURS, sep=";", index=False)

    return {
        "metriques_h": metriques_h,
        "metriques_j": metriques_j,
        "metriques_p": metriques_p if len(details_pics) > 0 else None,
        "par_heure": par_heure,
        "par_saison": par_saison,
        "par_type": par_type,
        "pires": pires,
        "test_clean": test_clean,
    }


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MODÉLISATION")
    print("Projet de prévision - France métropolitaine")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    df, meteo = charger_donnees()
    df = joindre_meteo(df, meteo)
    verifier_features(df)

    train, validation, test = split_chronologique(df)

    # --------------------------------------------------------
    # Étape 1 : XGBoost
    # --------------------------------------------------------

    resultats_xgb = entrainer_xgboost(train, validation, test)

    # --------------------------------------------------------
    # Étape 2 : Ensemble
    # --------------------------------------------------------

    resultats_ensemble = entrainer_ensemble(
        train, validation, resultats_xgb
    )

    # --------------------------------------------------------
    # Évaluation détaillée
    # --------------------------------------------------------

    resultats_eval = evaluer_modele_final(
        resultats_xgb, resultats_ensemble
    )

    # --------------------------------------------------------
    # Sauvegarde des prédictions et importances
    # --------------------------------------------------------

    test_clean = resultats_eval["test_clean"]

    # Prédictions
    predictions = test_clean[[
        "origine_prevision", "date_cible", "heure_cible",
        "consommation_cible",
        "prediction_avec_meteo",
    ]].copy()

    predictions = predictions.rename(columns={
        "consommation_cible": "consommation_reelle",
        "prediction_avec_meteo": "consommation_prevue",
    })

    predictions.to_csv(
        FICHIER_PREDICTIONS,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    print(f"\nPrédictions sauvegardées : {FICHIER_PREDICTIONS}")

    # Importances (fusion XGB + RF)
    importances = resultats_xgb["importances"].merge(
        resultats_ensemble["importances_rf"],
        on="variable",
        how="left",
    )

    importances.to_csv(
        FICHIER_IMPORTANCES,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    print(f"Importances sauvegardées : {FICHIER_IMPORTANCES}")

    # --------------------------------------------------------
    # Rapport consolidé
    # --------------------------------------------------------

    with open(FICHIER_RAPPORT, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("RAPPORT DE MODÉLISATION\n")
        f.write("=" * 70 + "\n\n")

        # Section 1 : XGBoost
        f.write("=" * 70 + "\n")
        f.write("ÉTAPE 1 : XGBOOST SEUL\n")
        f.write("=" * 70 + "\n\n")

        f.write("Hyperparamètres\n")
        f.write("-" * 70 + "\n")
        for k, v in XGB_PARAMS.items():
            f.write(f"  {k} : {v}\n")

        f.write(f"\nMeilleure itération : "
                f"{resultats_xgb['modele'].best_iteration}\n\n")

        f.write("Résultats\n")
        f.write("-" * 70 + "\n")
        f.write(f"Validation MAE  : {resultats_xgb['mae_val']:.2f} MW\n")
        f.write(f"Validation RMSE : {resultats_xgb['rmse_val']:.2f} MW\n")
        f.write(f"Test MAE        : {resultats_xgb['mae_test']:.2f} MW\n")
        f.write(f"Test RMSE       : {resultats_xgb['rmse_test']:.2f} MW\n")
        f.write(f"Test biais      : {resultats_xgb['biais_test']:+.2f} MW\n\n")

        # Section 2 : Ensemble
        f.write("=" * 70 + "\n")
        f.write("ÉTAPE 2 : ENSEMBLE RF + XGBOOST\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Alpha optimal (validation) : "
                f"{resultats_ensemble['alpha_optimal']}\n\n")

        f.write("Comparaison finale (test)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Modèle':<30s} {'MAE (MW)':>12s} {'RMSE (MW)':>12s}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'XGBoost seul':<30s} "
                f"{resultats_ensemble['mae_xgb']:>12.2f} "
                f"{resultats_ensemble['rmse_xgb']:>12.2f}\n")
        f.write(f"{'Random Forest seul':<30s} "
                f"{resultats_ensemble['mae_rf']:>12.2f} "
                f"{resultats_ensemble['rmse_rf']:>12.2f}\n")
        f.write(
            f"{'Ensemble (alpha=' + str(resultats_ensemble['alpha_optimal']) + ')':<30s} "
            f"{resultats_ensemble['mae_ens']:>12.2f} "
            f"{resultats_ensemble['rmse_ens']:>12.2f}\n\n"
        )
        f.write(f"Gain MAE  : {resultats_ensemble['gain_mae']:+.2f} %\n")
        f.write(f"Gain RMSE : {resultats_ensemble['gain_rmse']:+.2f} %\n\n")

        # Section 3 : Évaluation détaillée
        f.write("=" * 70 + "\n")
        f.write("ÉVALUATION DÉTAILLÉE (modèle d'ensemble)\n")
        f.write("=" * 70 + "\n\n")

        f.write("Métriques horaires\n")
        f.write("-" * 70 + "\n")
        f.write(resultats_eval["metriques_h"].to_string(index=False))
        f.write("\n\n")

        f.write("Métriques journalières\n")
        f.write("-" * 70 + "\n")
        f.write(resultats_eval["metriques_j"].to_string(index=False))
        f.write("\n\n")

        if resultats_eval["metriques_p"] is not None:
            f.write("Métriques sur les pics\n")
            f.write("-" * 70 + "\n")
            f.write(resultats_eval["metriques_p"].to_string(index=False))
            f.write("\n\n")

        f.write("Erreur par heure\n")
        f.write("-" * 70 + "\n")
        f.write(resultats_eval["par_heure"].to_string(index=False))
        f.write("\n\n")

        f.write("Erreur par saison\n")
        f.write("-" * 70 + "\n")
        f.write(resultats_eval["par_saison"].to_string(index=False))
        f.write("\n\n")

        f.write("Erreur par type de jour\n")
        f.write("-" * 70 + "\n")
        f.write(resultats_eval["par_type"].to_string(index=False))
        f.write("\n\n")

        f.write("Pires journées\n")
        f.write("-" * 70 + "\n")
        f.write(resultats_eval["pires"].to_string(index=False))
        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("FIN DU RAPPORT\n")
        f.write("=" * 70 + "\n")

    print(f"Rapport consolidé : {FICHIER_RAPPORT}")

    print()
    print("=" * 70)
    print("MODÉLISATION TERMINÉE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
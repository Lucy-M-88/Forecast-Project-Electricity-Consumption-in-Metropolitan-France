# -*- coding: utf-8 -*-

"""
11_modelisation_regionale.py

Entraînement de modèles régionaux de prévision de consommation.

Même structure temporelle que 05_modelisation.py :
    Train      : jusqu'au 2023-12-31
    Validation : 2024-01-01 -> 2024-12-31
    Test       : après 2024-12-31

Lancement :
    python 11_modelisation_regionale.py
"""

from pathlib import Path
from datetime import timedelta

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
MODELS_DIR = BASE_DIR / "models" / "regional"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONSO_REGIONALE = DATA_DIR / "eco2mix_regional_horaire.csv"
METEO_REGIONALE = DATA_DIR / "meteo_regionale_journaliere.csv"

FICHIER_PREVISIONS = RESULTS_DIR / "previsions_regionales.csv"
FICHIER_RAPPORT = RESULTS_DIR / "regional_rapport.txt"

DATE_FIN_TRAIN = pd.Timestamp("2023-12-31")
DATE_FIN_VALIDATION = pd.Timestamp("2024-12-31")

RANDOM_STATE = 42

FEATURES = [
    "consommation_j7", "consommation_j1an",
    "heure", "jour_semaine", "mois", "weekend", "est_ferie",
    "temperature_moyenne", "temperature_min", "temperature_max",
    "amplitude_thermique",
    "HDD_moyen", "HDD_min", "HDD_max",
    "TM_3j", "TM_7j", "HDD_3j", "HDD_7j",
    "duree_gel_min",
]

CIBLE = "consommation_cible"

XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 6,
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

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
    "2027-01-01", "2027-03-29", "2027-05-01", "2027-05-08",
    "2027-05-06", "2027-05-17", "2027-07-14", "2027-08-15",
    "2027-11-01", "2027-11-11", "2027-12-25",
}


def est_ferie(d):
    return int(d.strftime("%Y-%m-%d") in JOURS_FERIES)


def normalize_code(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 1 and s.isdigit():
        s = "0" + s
    return s


# ============================================================
# 2. CHARGEMENT
# ============================================================

def charger_conso_regionale():

    print(f"\nChargement conso régionale : {CONSO_REGIONALE.name}")

    df = pd.read_csv(CONSO_REGIONALE, sep=";", low_memory=False)

    df["date_heure_utc"] = pd.to_datetime(
        df["date_heure_utc"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["date_heure_utc"]).copy()

    df["timestamp"] = (
        df["date_heure_utc"].dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    )
    df["date_locale"] = df["timestamp"].dt.date
    df["heure_locale"] = df["timestamp"].dt.hour

    df["code_region"] = df["code_region"].apply(normalize_code)
    df["consommation_horaire"] = pd.to_numeric(
        df["consommation_horaire"], errors="coerce"
    )

    df = df.dropna(subset=["consommation_horaire"])

    df = df.sort_values(
        ["code_region", "timestamp"]
    ).reset_index(drop=True)

    print(f"  {len(df):,} lignes, {df['code_region'].nunique()} régions")
    print(f"    Codes : {sorted(df['code_region'].unique())}")

    return df


def charger_meteo_regionale():

    print(f"\nChargement météo régionale : {METEO_REGIONALE.name}")

    df = pd.read_csv(METEO_REGIONALE, sep=";", low_memory=False)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code_region"] = df["code_region"].apply(normalize_code)

    print(f"  {len(df):,} lignes, {df['code_region'].nunique()} régions")

    return df


# ============================================================
# 3. CONSTRUCTION DU DATASET RÉGIONAL
# ============================================================

def construire_dataset_region(conso_region, meteo_region, code_region):

    print(f"  Construction dataset région {code_region}...")

    lookup = {}
    for row in conso_region.itertuples(index=False):
        lookup[(row.date_locale, int(row.heure_locale))] = float(
            row.consommation_horaire
        )

    meteo_dict = {}
    for row in meteo_region.itertuples(index=False):
        meteo_dict[row.date.date()] = {
            "temperature_moyenne": row.temperature_moyenne,
            "temperature_min": row.temperature_min,
            "temperature_max": row.temperature_max,
            "amplitude_thermique": row.amplitude_thermique,
            "HDD_moyen": row.HDD_moyen,
            "HDD_min": row.HDD_min,
            "HDD_max": row.HDD_max,
            "TM_3j": row.TM_3j,
            "TM_7j": row.TM_7j,
            "HDD_3j": row.HDD_3j,
            "HDD_7j": row.HDD_7j,
            "duree_gel_min": row.duree_gel_min,
        }

    jours_conso = {}
    for (d, _) in lookup.keys():
        jours_conso.setdefault(d, set())

    jours_24h = {
        d for d in jours_conso.keys()
        if len([h for h in range(24) if (d, h) in lookup]) == 24
    }

    rows = []

    for date_cible in sorted(jours_24h):

        date_j7 = date_cible - timedelta(days=7)
        try:
            date_j1an = date_cible.replace(year=date_cible.year - 1)
        except ValueError:
            continue

        if date_j7 not in jours_24h or date_j1an not in jours_24h:
            continue

        met = meteo_dict.get(date_cible)
        if met is None or any(pd.isna(v) for v in met.values()):
            continue

        for heure in range(24):

            c7 = lookup.get((date_j7, heure))
            c1 = lookup.get((date_j1an, heure))
            y = lookup.get((date_cible, heure))

            if c7 is None or c1 is None or y is None:
                continue

            js = date_cible.weekday()

            rows.append({
                "origine_prevision": pd.Timestamp(date_cible) - pd.Timedelta(days=1),
                "date_cible": pd.Timestamp(date_cible),
                "heure_cible": heure,
                "timestamp": pd.Timestamp(date_cible) + pd.Timedelta(hours=heure),
                "code_region": code_region,
                "consommation_j7": c7,
                "consommation_j1an": c1,
                "heure": heure,
                "jour_semaine": js,
                "mois": date_cible.month,
                "weekend": int(js >= 5),
                "est_ferie": est_ferie(date_cible),
                "temperature_moyenne": met["temperature_moyenne"],
                "temperature_min": met["temperature_min"],
                "temperature_max": met["temperature_max"],
                "amplitude_thermique": met["amplitude_thermique"],
                "HDD_moyen": met["HDD_moyen"],
                "HDD_min": met["HDD_min"],
                "HDD_max": met["HDD_max"],
                "TM_3j": met["TM_3j"],
                "TM_7j": met["TM_7j"],
                "HDD_3j": met["HDD_3j"],
                "HDD_7j": met["HDD_7j"],
                "duree_gel_min": met["duree_gel_min"],
                "consommation_cible": y,
            })

    if not rows:
        print(f"    ⚠ Aucun exemple pour {code_region}")
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"    {len(df):,} lignes, "
          f"{df['date_cible'].min().date()} -> "
          f"{df['date_cible'].max().date()}")

    return df


# ============================================================
# 4. ENTRAÎNEMENT
# ============================================================

def entrainer_region(df, code_region, nom_region):

    print(f"\n  Région {code_region} ({nom_region})")

    df = df.copy()
    df["origine_prevision"] = pd.to_datetime(df["origine_prevision"])

    train = df[df["origine_prevision"] <= DATE_FIN_TRAIN].copy()
    validation = df[
        (df["origine_prevision"] > DATE_FIN_TRAIN)
        & (df["origine_prevision"] <= DATE_FIN_VALIDATION)
    ].copy()
    test = df[df["origine_prevision"] > DATE_FIN_VALIDATION].copy()

    train = train.dropna(subset=FEATURES + [CIBLE])
    validation = validation.dropna(subset=FEATURES + [CIBLE])
    test = test.dropna(subset=FEATURES + [CIBLE])

    print(f"    Train      : {len(train):,}")
    print(f"    Validation : {len(validation):,}")
    print(f"    Test       : {len(test):,}")

    if len(train) < 1000 or len(validation) < 100 or len(test) < 100:
        print(f"    ⚠ Trop peu de données, on saute")
        return None

    X_train, y_train = train[FEATURES], train[CIBLE]
    X_val, y_val = validation[FEATURES], validation[CIBLE]
    X_test, y_test = test[FEATURES], test[CIBLE]

    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    pred_xgb_val = xgb.predict(X_val)
    pred_xgb_test = xgb.predict(X_test)

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    pred_rf_val = rf.predict(X_val)
    pred_rf_test = rf.predict(X_test)

    resultats_val = []
    for a in ALPHAS:
        p = a * pred_rf_val + (1 - a) * pred_xgb_val
        resultats_val.append({"alpha": a, "MAE": mean_absolute_error(y_val, p)})

    df_val = pd.DataFrame(resultats_val)
    alpha_optimal = df_val.sort_values("MAE").iloc[0]["alpha"]

    pred_test = alpha_optimal * pred_rf_test + (1 - alpha_optimal) * pred_xgb_test

    mae_test = mean_absolute_error(y_test, pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    biais_test = np.mean(pred_test - y_test.values)

    print(f"    Alpha optimal : {alpha_optimal}")
    print(f"    Test MAE      : {mae_test:.2f} MW")
    print(f"    Test RMSE     : {rmse_test:.2f} MW")
    print(f"    Test biais    : {biais_test:+.2f} MW")

    chemin_modele = MODELS_DIR / f"{code_region}_ensemble.joblib"
    joblib.dump({
        "rf": rf, "xgb": xgb, "alpha": alpha_optimal,
        "features": FEATURES, "code_region": code_region,
        "nom_region": nom_region,
        "mae_test": mae_test, "rmse_test": rmse_test, "biais_test": biais_test,
    }, chemin_modele)

    test = test.copy()
    test["consommation_prevue"] = pred_test
    test["nom_region"] = nom_region

    return {
        "predictions": test[[
            "timestamp", "code_region", "nom_region",
            "consommation_prevue", "consommation_cible",
        ]].rename(columns={"consommation_cible": "consommation_reelle"}),
        "mae_test": mae_test, "rmse_test": rmse_test, "biais_test": biais_test,
        "alpha": alpha_optimal,
        "n_train": len(train), "n_val": len(validation), "n_test": len(test),
    }


# ============================================================
# 5. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MODÉLISATION RÉGIONALE")
    print("=" * 70)

    conso = charger_conso_regionale()
    meteo = charger_meteo_regionale()

    regions_conso = set(conso["code_region"].unique())
    regions_meteo = set(meteo["code_region"].unique())

    print(f"\nRégions avec conso : {sorted(regions_conso)}")
    print(f"Régions avec météo : {sorted(regions_meteo)}")

    regions_communes = sorted(regions_conso & regions_meteo)
    regions_manquantes = sorted(regions_conso - regions_meteo)

    if regions_manquantes:
        print(f"\n⚠ Régions avec conso mais SANS météo : {regions_manquantes}")

    print(f"\n{len(regions_communes)} régions à traiter")

    resultats = []
    rapports = []

    for code_region in regions_communes:

        conso_r = conso[conso["code_region"] == code_region].copy()
        meteo_r = meteo[meteo["code_region"] == code_region].copy()

        nom_region = conso_r["nom_region"].iloc[0]

        df = construire_dataset_region(conso_r, meteo_r, code_region)

        if df is None or len(df) < 1000:
            print(f"    ⚠ Skip {code_region} : dataset trop petit")
            continue

        res = entrainer_region(df, code_region, nom_region)

        if res is None:
            continue

        resultats.append(res["predictions"])
        rapports.append({
            "code_region": code_region,
            "nom_region": nom_region,
            "n_train": res["n_train"],
            "n_val": res["n_val"],
            "n_test": res["n_test"],
            "alpha": res["alpha"],
            "MAE": res["mae_test"],
            "RMSE": res["rmse_test"],
            "biais": res["biais_test"],
        })

    if not resultats:
        raise ValueError("Aucun modèle régional entraîné.")

    df_pred = pd.concat(resultats, ignore_index=True)
    df_pred = df_pred.sort_values(
        ["timestamp", "code_region"]
    ).reset_index(drop=True)

    df_pred.to_csv(FICHIER_PREVISIONS, sep=";", index=False, encoding="utf-8")

    print(f"\nPrédictions régionales : {FICHIER_PREVISIONS}")
    print(f"  Lignes  : {len(df_pred):,}")
    print(f"  Régions : {df_pred['code_region'].nunique()}")
    print(f"    Codes : {sorted(df_pred['code_region'].unique())}")

    df_rap = pd.DataFrame(rapports)

    with open(FICHIER_RAPPORT, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RAPPORT MODÉLISATION RÉGIONALE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Train      : jusqu'au {DATE_FIN_TRAIN.date()}\n")
        f.write(f"Validation : jusqu'au {DATE_FIN_VALIDATION.date()}\n")
        f.write(f"Test       : après {DATE_FIN_VALIDATION.date()}\n\n")
        f.write("Performances par région (test)\n")
        f.write("-" * 70 + "\n")
        f.write(df_rap.to_string(index=False))
        f.write("\n\n")
        f.write(f"MAE moyenne   : {df_rap['MAE'].mean():.2f} MW\n")
        f.write(f"RMSE moyenne  : {df_rap['RMSE'].mean():.2f} MW\n")
        f.write(f"Biais moyen   : {df_rap['biais'].mean():+.2f} MW\n")
        f.write(f"Nb régions    : {len(df_rap)}\n")

    print(f"Rapport : {FICHIER_RAPPORT}")

    print("\n" + "=" * 70)
    print("PERFORMANCES (test)")
    print("=" * 70)
    print(df_rap[["nom_region", "n_test", "MAE", "RMSE", "biais"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

"""
07_evaluation_meteo.py

Objectif
--------
Évaluation détaillée du modèle Random Forest avec météo enrichie.

Produit :
    results/evaluation_horaire.csv
    results/evaluation_journaliere.csv
    results/evaluation_pics.csv
    results/evaluation_par_heure.csv
    results/evaluation_par_saison.csv
    results/evaluation_par_type_jour.csv
    results/pires_journees_meteo.csv
    results/comparaison_finale_meteo.csv
    results/rapport_evaluation_meteo.txt

IMPORTANT :
    - On évalue uniquement le modèle avec météo.
    - Les benchmarks (J-7, J-1an, moyenne, RTE) sont conservés
      comme référence externe.
    - Le modèle sans météo n'est plus comparé.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PREVISION = DATA_DIR / "dataset_prevision.csv"
DATASET_METEO = DATA_DIR / "meteo_france_nationale_journaliere.csv"
PREDICTIONS_METEO = RESULTS_DIR / "predictions_meteo.csv"

FICHIER_HORAIRE = RESULTS_DIR / "evaluation_horaire.csv"
FICHIER_JOURNALIER = RESULTS_DIR / "evaluation_journaliere.csv"
FICHIER_PICS = RESULTS_DIR / "evaluation_pics.csv"
FICHIER_PAR_HEURE = RESULTS_DIR / "evaluation_par_heure.csv"
FICHIER_PAR_SAISON = RESULTS_DIR / "evaluation_par_saison.csv"
FICHIER_PAR_TYPE_JOUR = RESULTS_DIR / "evaluation_par_type_jour.csv"
FICHIER_PIRES_JOURS = RESULTS_DIR / "pires_journees_meteo.csv"
FICHIER_COMPARAISON = RESULTS_DIR / "comparaison_finale_meteo.csv"
FICHIER_RAPPORT = RESULTS_DIR / "rapport_evaluation_meteo.txt"


# ------------------------------------------------------------
# Jours fériés français
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

    print("\nChargement du dataset de prévision...")
    df = pd.read_csv(DATASET_PREVISION, sep=";", low_memory=False)
    print(f"  Lignes : {len(df):,}")

    print("\nChargement de la météo nationale...")
    meteo = pd.read_csv(DATASET_METEO, sep=";", low_memory=False)
    print(f"  Lignes : {len(meteo):,}")

    print("\nChargement des prédictions météo...")
    if not PREDICTIONS_METEO.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PREDICTIONS_METEO}\n"
            f"Lance d'abord 06_modele_meteo.py."
        )
    predictions = pd.read_csv(PREDICTIONS_METEO, sep=";", low_memory=False)
    print(f"  Lignes : {len(predictions):,}")

    df["origine_prevision"] = pd.to_datetime(
        df["origine_prevision"], errors="coerce"
    )
    df["date_cible"] = pd.to_datetime(
        df["date_cible"], errors="coerce"
    )
    meteo["date"] = pd.to_datetime(
        meteo["date"], errors="coerce"
    )
    predictions["origine_prevision"] = pd.to_datetime(
        predictions["origine_prevision"], errors="coerce"
    )
    predictions["date_cible"] = pd.to_datetime(
        predictions["date_cible"], errors="coerce"
    )

    return df, meteo, predictions


def reconstruire_test(df, predictions):
    """
    Reconstitue le sous-ensemble de test utilisé par
    06_modele_meteo.py.
    """

    origines_test = predictions["origine_prevision"].unique()
    test = df[df["origine_prevision"].isin(origines_test)].copy()

    return test


def ajouter_benchmarks(test):
    """
    Ajoute les benchmarks J-7, J-1 an et moyenne.
    """

    test["benchmark_j7"] = test["consommation_j7"]
    test["benchmark_j1an"] = test["consommation_j1an"]
    test["benchmark_moyenne"] = (
        test["consommation_j7"] + test["consommation_j1an"]
    ) / 2

    return test


# ============================================================
# 4. MÉTRIQUES HORAIRES
# ============================================================

def metriques_horaires(test, colonnes_modeles):

    lignes = []

    for col in colonnes_modeles:

        if col not in test.columns:
            continue

        sous = test[["consommation_cible", col]].dropna()

        if len(sous) == 0:
            continue

        y_true = sous["consommation_cible"].values
        y_pred = sous[col].values

        lignes.append({
            "modele": col,
            "n_heures": len(sous),
            "MAE_MW": mae(y_true, y_pred),
            "RMSE_MW": rmse(y_true, y_pred),
            "biais_MW": biais(y_true, y_pred),
        })

    return pd.DataFrame(lignes).sort_values("MAE_MW")


# ============================================================
# 5. MÉTRIQUES JOURNALIÈRES
# ============================================================

def metriques_journalieres(test, colonnes_modeles):

    agg = {"consommation_cible": "sum"}
    for col in colonnes_modeles:
        if col in test.columns:
            agg[col] = "sum"

    journalier = (
        test
        .groupby("date_cible")
        .agg(agg)
        .reset_index()
    )

    lignes = []

    for col in colonnes_modeles:

        if col not in journalier.columns:
            continue

        sous = journalier[["consommation_cible", col]].dropna()

        if len(sous) == 0:
            continue

        erreur = sous[col] - sous["consommation_cible"]

        lignes.append({
            "modele": col,
            "n_jours": len(sous),
            "MAE_total_MWh": np.mean(np.abs(erreur)),
            "RMSE_total_MWh": np.sqrt(np.mean(erreur ** 2)),
            "biais_total_MWh": np.mean(erreur),
        })

    return pd.DataFrame(lignes).sort_values("MAE_total_MWh"), journalier


# ============================================================
# 6. MÉTRIQUES SUR LES PICS
# ============================================================

def metriques_pics(test, colonnes_modeles):

    lignes = []

    for date, groupe in test.groupby("date_cible"):

        groupe_reel = groupe.dropna(subset=["consommation_cible"])

        if len(groupe_reel) != 24:
            continue

        idx_reel = groupe_reel["consommation_cible"].idxmax()
        pic_reel = groupe_reel.loc[idx_reel, "consommation_cible"]
        heure_pic_reel = int(groupe_reel.loc[idx_reel, "heure_cible"])

        for col in colonnes_modeles:

            if col not in groupe.columns:
                continue

            groupe_modele = groupe.dropna(
                subset=["consommation_cible", col]
            )

            if len(groupe_modele) != 24:
                continue

            idx_prevu = groupe_modele[col].idxmax()
            pic_prevu = groupe_modele.loc[idx_prevu, col]
            heure_pic_prevu = int(
                groupe_modele.loc[idx_prevu, "heure_cible"]
            )

            lignes.append({
                "date_cible": date,
                "modele": col,
                "pic_reel_MW": pic_reel,
                "pic_prevu_MW": pic_prevu,
                "erreur_pic_MW": pic_prevu - pic_reel,
                "erreur_pic_abs_MW": abs(pic_prevu - pic_reel),
                "heure_pic_reel": heure_pic_reel,
                "heure_pic_prevu": heure_pic_prevu,
                "ecart_heure_pic": heure_pic_prevu - heure_pic_reel,
                "pic_correct": int(heure_pic_prevu == heure_pic_reel),
            })

    details = pd.DataFrame(lignes)

    if len(details) == 0:
        return pd.DataFrame(), details

    resume = (
        details
        .groupby("modele")
        .agg(
            n_jours=("date_cible", "count"),
            MAE_pic_MW=("erreur_pic_abs_MW", "mean"),
            biais_pic_MW=("erreur_pic_MW", "mean"),
            ecart_moyen_heure_pic=("ecart_heure_pic", lambda x: np.mean(np.abs(x))),
            taux_pic_correct=("pic_correct", "mean"),
        )
        .reset_index()
    )

    resume["taux_pic_correct"] *= 100

    return resume.sort_values("MAE_pic_MW"), details


# ============================================================
# 7. ANALYSE PAR HEURE, SAISON, TYPE DE JOUR
# ============================================================

def analyse_par_heure(test, col_modele):

    test = test.copy()
    test["erreur"] = test[col_modele] - test["consommation_cible"]
    test["erreur_abs"] = test["erreur"].abs()

    agg = (
        test
        .groupby("heure_cible")
        .agg(
            MAE_MW=("erreur_abs", "mean"),
            biais_MW=("erreur", "mean"),
            n=("erreur", "count"),
        )
        .reset_index()
    )

    return agg


def analyse_par_saison(test, col_modele):

    test = test.copy()
    test["saison"] = test["date_cible"].apply(saison)
    test["erreur"] = test[col_modele] - test["consommation_cible"]
    test["erreur_abs"] = test["erreur"].abs()

    agg = (
        test
        .groupby("saison")
        .agg(
            MAE_MW=("erreur_abs", "mean"),
            RMSE_MW=("erreur", lambda x: np.sqrt(np.mean(x ** 2))),
            biais_MW=("erreur", "mean"),
            n=("erreur", "count"),
        )
        .reset_index()
    )

    return agg


def analyse_par_type_jour(test, col_modele):

    test = test.copy()
    test["type_jour"] = test["date_cible"].apply(type_jour)
    test["erreur"] = test[col_modele] - test["consommation_cible"]
    test["erreur_abs"] = test["erreur"].abs()

    agg = (
        test
        .groupby("type_jour")
        .agg(
            MAE_MW=("erreur_abs", "mean"),
            RMSE_MW=("erreur", lambda x: np.sqrt(np.mean(x ** 2))),
            biais_MW=("erreur", "mean"),
            n=("erreur", "count"),
        )
        .reset_index()
    )

    return agg


# ============================================================
# 8. PIRES JOURNÉES
# ============================================================

def pires_journees(test, col_modele, meteo, n=15):

    test = test.copy()
    test["erreur"] = test[col_modele] - test["consommation_cible"]
    test["erreur_abs"] = test["erreur"].abs()

    journalier = (
        test
        .groupby("date_cible")
        .agg(
            MAE_jour_MW=("erreur_abs", "mean"),
            RMSE_jour_MW=("erreur", lambda x: np.sqrt(np.mean(x ** 2))),
            erreur_totale_MWh=("erreur", "sum"),
            erreur_totale_abs_MWh=("erreur", lambda x: abs(x.sum())),
            pic_reel_MW=("consommation_cible", "max"),
            pic_prevu_MW=(col_modele, "max"),
        )
        .reset_index()
    )

    meteo_sub = meteo[[
        "date", "temperature_moyenne", "temperature_min",
        "temperature_max", "HDD_moyen", "precipitations_totales_mm"
    ]].rename(columns={"date": "date_cible"})

    journalier = journalier.merge(
        meteo_sub, on="date_cible", how="left"
    )

    journalier["type_jour"] = journalier["date_cible"].apply(type_jour)
    journalier["saison"] = journalier["date_cible"].apply(saison)

    journalier = journalier.sort_values(
        "MAE_jour_MW", ascending=False
    ).head(n)

    return journalier


# ============================================================
# 9. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("ÉVALUATION DÉTAILLÉE DU MODÈLE SIMPLIFIÉ")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    df, meteo, predictions = charger_donnees()

    test = reconstruire_test(df, predictions)

    print(f"\nLignes de test reconstruites : {len(test):,}")

    test = test.merge(
        predictions,
        on=["origine_prevision", "heure_cible"],
        how="inner",
        suffixes=("", "_pred")
    )

    print(f"Lignes après jointure prédictions : {len(test):,}")

    test = ajouter_benchmarks(test)

    # --------------------------------------------------------
    # Modèles comparés : notre modèle + benchmarks
    # --------------------------------------------------------

    modeles = [
        "prediction_avec_meteo",
        "benchmark_j7",
        "benchmark_j1an",
        "benchmark_moyenne",
    ]

    modeles = [m for m in modeles if m in test.columns]

    print("\nModèles comparés :")
    for m in modeles:
        print(f"  - {m}")

    # ========================================================
    # 1. MÉTRIQUES HORAIRES
    # ========================================================

    print("\n" + "=" * 70)
    print("1. MÉTRIQUES HORAIRES")
    print("=" * 70)

    metriques_h = metriques_horaires(test, modeles)
    print(metriques_h.to_string(index=False))
    metriques_h.to_csv(FICHIER_HORAIRE, sep=";", index=False)

    # ========================================================
    # 2. MÉTRIQUES JOURNALIÈRES
    # ========================================================

    print("\n" + "=" * 70)
    print("2. MÉTRIQUES JOURNALIÈRES")
    print("=" * 70)

    metriques_j, journalier = metriques_journalieres(test, modeles)
    print(metriques_j.to_string(index=False))
    metriques_j.to_csv(FICHIER_JOURNALIER, sep=";", index=False)

    # ========================================================
    # 3. MÉTRIQUES SUR LES PICS
    # ========================================================

    print("\n" + "=" * 70)
    print("3. MÉTRIQUES SUR LES PICS")
    print("=" * 70)

    metriques_p, details_p = metriques_pics(test, modeles)

    if len(metriques_p) > 0:
        print(metriques_p.to_string(index=False))
        metriques_p.to_csv(FICHIER_PICS, sep=";", index=False)

    # ========================================================
    # 4. ANALYSE PAR HEURE
    # ========================================================

    print("\n" + "=" * 70)
    print("4. ERREUR PAR HEURE (modèle simplifié)")
    print("=" * 70)

    par_heure = analyse_par_heure(test, "prediction_avec_meteo")
    print(par_heure.to_string(index=False))
    par_heure.to_csv(FICHIER_PAR_HEURE, sep=";", index=False)

    # ========================================================
    # 5. ANALYSE PAR SAISON
    # ========================================================

    print("\n" + "=" * 70)
    print("5. ERREUR PAR SAISON (modèle simplifié)")
    print("=" * 70)

    par_saison = analyse_par_saison(test, "prediction_avec_meteo")
    print(par_saison.to_string(index=False))
    par_saison.to_csv(FICHIER_PAR_SAISON, sep=";", index=False)

    # ========================================================
    # 6. ANALYSE PAR TYPE DE JOUR
    # ========================================================

    print("\n" + "=" * 70)
    print("6. ERREUR PAR TYPE DE JOUR (modèle simplifié)")
    print("=" * 70)

    par_type = analyse_par_type_jour(test, "prediction_avec_meteo")
    print(par_type.to_string(index=False))
    par_type.to_csv(FICHIER_PAR_TYPE_JOUR, sep=";", index=False)

    # ========================================================
    # 7. PIRES JOURNÉES
    # ========================================================

    print("\n" + "=" * 70)
    print("7. PIRES JOURNÉES (modèle simplifié)")
    print("=" * 70)

    pires = pires_journees(test, "prediction_avec_meteo", meteo, n=15)
    print(pires.to_string(index=False))
    pires.to_csv(FICHIER_PIRES_JOURS, sep=";", index=False)

    # ========================================================
    # 8. RAPPORT TEXTE
    # ========================================================

    print("\nRédaction du rapport...")

    with open(FICHIER_RAPPORT, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("RAPPORT D'ÉVALUATION DU MODÈLE SIMPLIFIÉ\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Période de test : "
            f"{test['date_cible'].min().date()} -> "
            f"{test['date_cible'].max().date()}\n"
        )
        f.write(f"Nombre de jours de test : {test['date_cible'].nunique()}\n")
        f.write(f"Nombre d'heures de test : {len(test)}\n\n")

        f.write("MODÈLES COMPARÉS\n")
        f.write("-" * 70 + "\n")
        for m in modeles:
            f.write(f"  - {m}\n")

        f.write("\n1. MÉTRIQUES HORAIRES\n")
        f.write("-" * 70 + "\n")
        f.write(metriques_h.to_string(index=False))
        f.write("\n\n")

        f.write("2. MÉTRIQUES JOURNALIÈRES\n")
        f.write("-" * 70 + "\n")
        f.write(metriques_j.to_string(index=False))
        f.write("\n\n")

        f.write("3. MÉTRIQUES SUR LES PICS\n")
        f.write("-" * 70 + "\n")
        if len(metriques_p) > 0:
            f.write(metriques_p.to_string(index=False))
        f.write("\n\n")

        f.write("4. ERREUR PAR HEURE\n")
        f.write("-" * 70 + "\n")
        f.write(par_heure.to_string(index=False))
        f.write("\n\n")

        f.write("5. ERREUR PAR SAISON\n")
        f.write("-" * 70 + "\n")
        f.write(par_saison.to_string(index=False))
        f.write("\n\n")

        f.write("6. ERREUR PAR TYPE DE JOUR\n")
        f.write("-" * 70 + "\n")
        f.write(par_type.to_string(index=False))
        f.write("\n\n")

        f.write("7. PIRES JOURNÉES\n")
        f.write("-" * 70 + "\n")
        f.write(pires.to_string(index=False))
        f.write("\n\n")

        f.write("8. INTERPRÉTATION\n")
        f.write("-" * 70 + "\n")

        meilleur = metriques_h.iloc[0]
        f.write(
            f"Le meilleur modèle est {meilleur['modele']} "
            f"avec une MAE de {meilleur['MAE_MW']:.2f} MW "
            f"et un RMSE de {meilleur['RMSE_MW']:.2f} MW.\n\n"
        )

        if len(par_saison) > 0:
            pire_saison = par_saison.sort_values(
                "MAE_MW", ascending=False
            ).iloc[0]
            f.write(
                f"Saison la plus difficile : "
                f"{pire_saison['saison']} "
                f"(MAE = {pire_saison['MAE_MW']:.2f} MW)\n"
            )

        if len(par_type) > 0:
            pire_type = par_type.sort_values(
                "MAE_MW", ascending=False
            ).iloc[0]
            f.write(
                f"Type de jour le plus difficile : "
                f"{pire_type['type_jour']} "
                f"(MAE = {pire_type['MAE_MW']:.2f} MW)\n"
            )

        f.write("\n" + "=" * 70 + "\n")
        f.write("FIN DU RAPPORT\n")
        f.write("=" * 70 + "\n")

    print(f"\nRapport sauvegardé : {FICHIER_RAPPORT}")

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

"""
13_completer_aura.py

Complète les données manquantes d'Auvergne-Rhône-Alpes (84) dans
le fichier eco2mix_regional_horaire.csv.

MÉTHODE
-------
La consommation nationale = somme des 12 régions métropolitaines.
Donc pour chaque timestamp où AURA manque :

    AURA(t) = National(t) − somme(11 autres régions)(t)

On ne remplace PAS les valeurs existantes d'AURA (on les garde
telles quelles). On ajoute uniquement les lignes manquantes.

Vérifications :
    - Sur les timestamps où AURA existe ET où la reconstitution est
      possible, on compare les deux valeurs. Si l'écart médian est
      faible (<5%), la méthode est validée.
    - On ne reconstitue que si les 11 autres régions sont TOUTES
      présentes à ce timestamp.

Entrées :
    data/processed/eco2mix_national_horaire_fusionne.csv
    data/processed/eco2mix_regional_horaire.csv

Sortie :
    data/processed/eco2mix_regional_horaire.csv  (mis à jour)
    data/processed/eco2mix_regional_horaire_backup.csv (sauvegarde)

Lancement :
    python 13_completer_aura.py
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

CONSO_NATIONALE = DATA_DIR / "eco2mix_national_horaire_fusionne.csv"
CONSO_REGIONALE = DATA_DIR / "eco2mix_regional_horaire.csv"
BACKUP = DATA_DIR / "eco2mix_regional_horaire_backup.csv"

CODE_AURA = "84"
NOM_AURA = "Auvergne-Rhône-Alpes"

AUTRES_REGIONS = [
    "11", "24", "27", "28", "32", "44",
    "52", "53", "75", "76", "93",
]  # Sans AURA (84) ni Corse (94)

TIMEZONE = "Europe/Paris"

SEUIL_VALIDATION_PCT = 5.0  # écart médian acceptable (en %)


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

def charger_conso_nationale():

    print(f"\nChargement conso nationale...")
    print(f"  Fichier : {CONSO_NATIONALE.name}")

    df = pd.read_csv(CONSO_NATIONALE, sep=";", low_memory=False)

    df["date_heure_utc"] = pd.to_datetime(
        df["date_heure_utc"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["date_heure_utc"]).copy()

    if "consommation_horaire" in df.columns:
        df["conso_nat"] = pd.to_numeric(
            df["consommation_horaire"], errors="coerce"
        )
    elif "consommation" in df.columns:
        df["conso_nat"] = pd.to_numeric(
            df["consommation"], errors="coerce"
        )
    else:
        raise ValueError(
            "Colonne 'consommation_horaire' ou 'consommation' absente."
        )

    df = df.dropna(subset=["conso_nat"])

    # On garde uniquement HH:00
    df["minute"] = df["date_heure_utc"].dt.minute
    df = df[df["minute"] == 0].copy()

    df = df[["date_heure_utc", "conso_nat"]].drop_duplicates("date_heure_utc")

    print(f"  {len(df):,} heures nationales valides")

    return df.sort_values("date_heure_utc").reset_index(drop=True)


def charger_conso_regionale():

    print(f"\nChargement conso régionale...")
    print(f"  Fichier : {CONSO_REGIONALE.name}")

    df = pd.read_csv(CONSO_REGIONALE, sep=";", low_memory=False)

    df["date_heure_utc"] = pd.to_datetime(
        df["date_heure_utc"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["date_heure_utc"]).copy()

    df["code_region"] = df["code_region"].apply(normalize_code)
    df["consommation_horaire"] = pd.to_numeric(
        df["consommation_horaire"], errors="coerce"
    )

    df = df.dropna(subset=["consommation_horaire"])

    print(f"  {len(df):,} lignes")
    print(f"  Régions : {sorted(df['code_region'].unique())}")

    # État initial d'AURA
    aura = df[df["code_region"] == CODE_AURA]
    print(f"\n  État actuel AURA ({CODE_AURA}) :")
    print(f"    Lignes      : {len(aura):,}")
    if len(aura) > 0:
        print(f"    Du          : {aura['date_heure_utc'].min()}")
        print(f"    Au          : {aura['date_heure_utc'].max()}")

    return df


# ============================================================
# 3. VALIDATION DE LA MÉTHODE
# ============================================================

def valider_methode(df_region, df_nat):

    print("\n" + "=" * 70)
    print("VALIDATION DE LA MÉTHODE")
    print("=" * 70)

    # Somme des autres régions par timestamp
    autres = df_region[df_region["code_region"].isin(AUTRES_REGIONS)]

    # Vérifier que les 11 autres régions sont bien présentes
    nb_regions_attendues = len(AUTRES_REGIONS)

    somme_autres = (
        autres.groupby("date_heure_utc")["consommation_horaire"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "conso_autres", "count": "nb_regions"})
    )

    # Fusion avec national
    df = df_nat.merge(somme_autres, on="date_heure_utc", how="inner")

    # Filtrer : 11 régions présentes
    df = df[df["nb_regions"] == nb_regions_attendues].copy()

    # AURA reconstruit
    df["aura_reconstruit"] = df["conso_nat"] - df["conso_autres"]

    # Comparer avec AURA observé quand il existe
    aura_obs = (
        df_region[df_region["code_region"] == CODE_AURA]
        [["date_heure_utc", "consommation_horaire"]]
        .rename(columns={"consommation_horaire": "aura_observe"})
    )

    df = df.merge(aura_obs, on="date_heure_utc", how="left")

    df_avec_obs = df.dropna(subset=["aura_observe"]).copy()

    if len(df_avec_obs) == 0:
        print("\n  ⚠ Aucun timestamp commun AURA/reconstitution")
        print("  → On utilisera la reconstitution sans validation.")
        return df[["date_heure_utc", "aura_reconstruit"]]

    df_avec_obs["ecart"] = (
        df_avec_obs["aura_reconstruit"] - df_avec_obs["aura_observe"]
    )
    df_avec_obs["ecart_abs"] = df_avec_obs["ecart"].abs()
    df_avec_obs["ecart_pct"] = (
        df_avec_obs["ecart_abs"] / df_avec_obs["aura_observe"] * 100
    )

    ecart_median = df_avec_obs["ecart_pct"].median()
    ecart_moyen = df_avec_obs["ecart_pct"].mean()
    ecart_max = df_avec_obs["ecart_pct"].max()

    print(f"\n  Timestamps communs observé/reconstruit : {len(df_avec_obs):,}")
    print(f"  Écart médian  : {ecart_median:.3f} %")
    print(f"  Écart moyen   : {ecart_moyen:.3f} %")
    print(f"  Écart max     : {ecart_max:.3f} %")

    if ecart_median > SEUIL_VALIDATION_PCT:
        print(f"\n  ⚠ Écart médian > {SEUIL_VALIDATION_PCT}%")
        print(f"  → La reconstitution est peu fiable.")
        print(f"  → On l'utilise quand même (pas d'alternative).")
    else:
        print(f"\n  ✓ Écart médian < {SEUIL_VALIDATION_PCT}%")
        print(f"  → La reconstitution est FIABLE.")

    return df[["date_heure_utc", "aura_reconstruit"]]


# ============================================================
# 4. RECONSTITUTION
# ============================================================

def reconstituer_aura(df_region, df_nat):

    print("\n" + "=" * 70)
    print("RECONSTITUTION D'AURA")
    print("=" * 70)

    df_reconstruit = valider_methode(df_region, df_nat)

    # --------------------------------------------------------
    # AURA observé (à garder tel quel)
    # --------------------------------------------------------

    aura_obs = df_region[df_region["code_region"] == CODE_AURA].copy()
    ts_observes = set(aura_obs["date_heure_utc"])

    print(f"\n  Lignes AURA observées : {len(aura_obs):,}")

    # --------------------------------------------------------
    # AURA reconstruit (uniquement là où il manque)
    # --------------------------------------------------------

    df_rec_filtre = df_reconstruit[
        ~df_reconstruit["date_heure_utc"].isin(ts_observes)
    ].copy()

    print(f"  Timestamps reconstruits : {len(df_rec_filtre):,}")

    # --------------------------------------------------------
    # Filtrer les valeurs aberrantes
    # --------------------------------------------------------

    # Consommation négative = suspect
    masque_negatif = df_rec_filtre["aura_reconstruit"] <= 0
    nb_negatifs = masque_negatif.sum()

    if nb_negatifs > 0:
        print(f"  ⚠ {nb_negatifs} valeurs reconstruites ≤ 0 → rejetées")
        df_rec_filtre = df_rec_filtre[~masque_negatif].copy()

    if len(df_rec_filtre) == 0:
        print("\n  Aucune ligne à ajouter.")
        return aura_obs, pd.DataFrame()

    # --------------------------------------------------------
    # Construire les lignes AURA complètes
    # --------------------------------------------------------

    # Reconstruction des colonnes
    df_new = df_rec_filtre.copy()

    # Heure locale
    df_new["date_heure"] = (
        df_new["date_heure_utc"].dt.tz_convert(TIMEZONE)
    )
    df_new["date"] = df_new["date_heure"].dt.date
    df_new["heure"] = df_new["date_heure"].dt.hour
    df_new["minute"] = 0
    df_new["code_region"] = CODE_AURA
    df_new["nom_region"] = NOM_AURA
    df_new["consommation"] = df_new["aura_reconstruit"]
    df_new["consommation_horaire"] = df_new["aura_reconstruit"]
    df_new["nature"] = "Reconstitué"

    # Colonnes dans le même ordre que le fichier original
    colonnes_cibles = [
        "date_heure_utc", "date_heure", "date", "heure", "minute",
        "code_region", "nom_region",
        "consommation", "consommation_horaire", "nature",
    ]

    df_new = df_new[colonnes_cibles].copy()

    print(f"\n  Lignes AURA reconstruites à ajouter : {len(df_new):,}")

    return aura_obs, df_new


# ============================================================
# 5. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("COMPLÉTION DES DONNÉES D'AUVERGNE-RHÔNE-ALPES")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    df_nat = charger_conso_nationale()
    df_region = charger_conso_regionale()

    # --------------------------------------------------------
    # Sauvegarde de l'original
    # --------------------------------------------------------

    if not BACKUP.exists():
        print(f"\nSauvegarde de l'original :")
        shutil.copy(CONSO_REGIONALE, BACKUP)
        print(f"  → {BACKUP.name}")
    else:
        print(f"\nBackup existant, non écrasé : {BACKUP.name}")

    # --------------------------------------------------------
    # Reconstitution
    # --------------------------------------------------------

    aura_obs, aura_new = reconstituer_aura(df_region, df_nat)

    if len(aura_new) == 0:
        print("\n✓ Aucune complétion nécessaire.")
        return

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUSION")
    print("=" * 70)

    # Régions autres qu'AURA (inchangées)
    df_autres = df_region[df_region["code_region"] != CODE_AURA].copy()

    # AURA complété
    df_aura = pd.concat([aura_obs, aura_new], ignore_index=True)

    df_aura = df_aura.sort_values("date_heure_utc").reset_index(drop=True)

    # Fusion finale
    df_final = pd.concat([df_autres, df_aura], ignore_index=True)

    df_final = df_final.sort_values(
        ["date_heure_utc", "code_region"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Contrôle : on ne doit pas avoir de doublons
    # --------------------------------------------------------

    nb_doublons = df_final.duplicated(
        subset=["date_heure_utc", "code_region"], keep=False
    ).sum()

    if nb_doublons > 0:
        print(f"\n  ⚠ {nb_doublons} doublons détectés, déduplication")
        df_final = df_final.drop_duplicates(
            subset=["date_heure_utc", "code_region"], keep="first"
        )

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    df_final.to_csv(CONSO_REGIONALE, sep=";", index=False, encoding="utf-8")

    # --------------------------------------------------------
    # Bilan final
    # --------------------------------------------------------

    print(f"\n  Fichier mis à jour : {CONSO_REGIONALE.name}")
    print(f"  Total lignes       : {len(df_final):,}")

    print(f"\n  État final AURA :")
    aura_final = df_final[df_final["code_region"] == CODE_AURA]
    print(f"    Lignes      : {len(aura_final):,}")
    print(f"    Du          : {aura_final['date_heure_utc'].min()}")
    print(f"    Au          : {aura_final['date_heure_utc'].max()}")
    print(f"    Observées   : {len(aura_obs):,}")
    print(f"    Reconstruites: {len(aura_new):,}")

    # Comparaison avec les autres régions
    print(f"\n  Comparaison avec les autres régions :")

    stats = df_final.groupby("code_region").agg(
        n=("date_heure_utc", "count"),
        min_ts=("date_heure_utc", "min"),
        max_ts=("date_heure_utc", "max"),
    )

    print(f"\n  {'Code':<6} {'Lignes':>10} {'Du':>12} {'Au':>12}")
    print("  " + "-" * 50)

    for code, row in stats.sort_index().iterrows():
        print(f"  {code:<6} {row['n']:>10,} "
              f"{str(row['min_ts'])[:10]:>12} "
              f"{str(row['max_ts'])[:10]:>12}")

    print()
    print("=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    print()
    print("Étapes suivantes :")
    print("  1. python 11_modelisation_regionale.py")
    print("  2. streamlit run 07_dashboard.py")
    print()


if __name__ == "__main__":
    main()
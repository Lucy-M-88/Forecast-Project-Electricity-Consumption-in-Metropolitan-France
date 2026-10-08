# -*- coding: utf-8 -*-

"""
02b_fusion_horaire.py

Objectif
--------
Fusionner les données éCO2mix consolidées et temps réel
pour produire un historique horaire complet.

Stratégie :
    1. Charger les données consolidées préparées (eco2mix_national_horaire.csv).
    2. Charger les données temps réel brutes.
    3. Appliquer les mêmes transformations aux données temps réel.
    4. Fusionner les deux sources.
    5. En cas de conflit, privilégier les données consolidées
       (plus fiables).
    6. Sauvegarder le fichier fusionné.

Entrée :
    data/processed/eco2mix_national_horaire.csv
    data/raw/eco2mix_national_temps_reel.csv

Sortie :
    data/processed/eco2mix_national_horaire_fusionne.csv
    data/processed/rapport_fusion.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PARAMETRES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

HORAIRE_CONSOLIDE = BASE_DIR / "data" / "processed" / "eco2mix_national_horaire.csv"
TEMPS_REEL_BRUT = BASE_DIR / "data" / "raw" / "eco2mix_national_temps_reel.csv"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "eco2mix_national_horaire_fusionne.csv"
REPORT_FILE = BASE_DIR / "data" / "processed" / "rapport_fusion.txt"

TIMEZONE = "Europe/Paris"


# ============================================================
# 2. CHARGEMENT
# ============================================================

print("=" * 70)
print("FUSION DES DONNÉES éCO2mix")
print("=" * 70)

# ------------------------------------------------------------
# Données consolidées
# ------------------------------------------------------------

print("\nChargement des données consolidées...")

if not HORAIRE_CONSOLIDE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {HORAIRE_CONSOLIDE}\n"
        "Lance d'abord 02_preparation_horaire.py."
    )

df_consolide = pd.read_csv(
    HORAIRE_CONSOLIDE,
    sep=";",
    low_memory=False,
)

print(f"  {len(df_consolide):,} lignes")

df_consolide["date_heure_utc"] = pd.to_datetime(
    df_consolide["date_heure_utc"],
    errors="coerce",
    utc=True,
)

print(
    f"  Période : "
    f"{df_consolide['date_heure_utc'].min()} -> "
    f"{df_consolide['date_heure_utc'].max()}"
)

# ------------------------------------------------------------
# Données temps réel
# ------------------------------------------------------------

print("\nChargement des données temps réel...")

if not TEMPS_REEL_BRUT.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {TEMPS_REEL_BRUT}\n"
        "Lance d'abord 01b_telechargement_temps_reel.py."
    )

df_tr = pd.read_csv(
    TEMPS_REEL_BRUT,
    sep=";",
    low_memory=False,
)

print(f"  {len(df_tr):,} lignes")

df_tr["date_heure_utc"] = pd.to_datetime(
    df_tr["date_heure"],
    errors="coerce",
    utc=True,
)

print(
    f"  Période : "
    f"{df_tr['date_heure_utc'].min()} -> "
    f"{df_tr['date_heure_utc'].max()}"
)


# ============================================================
# 3. TRANSFORMATION DES DONNEES TEMPS REEL
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMATION DES DONNÉES TEMPS RÉEL")
print("=" * 70)

# ------------------------------------------------------------
# Suppression des NaN
# ------------------------------------------------------------

df_tr = df_tr.dropna(subset=["date_heure_utc"]).copy()

# ------------------------------------------------------------
# Consommation numérique
# ------------------------------------------------------------

df_tr["consommation"] = pd.to_numeric(
    df_tr["consommation"],
    errors="coerce",
)

df_tr = df_tr.dropna(subset=["consommation"]).copy()

# ------------------------------------------------------------
# Tri chronologique
# ------------------------------------------------------------

df_tr = df_tr.sort_values("date_heure_utc").reset_index(drop=True)

# ------------------------------------------------------------
# Sélection des heures entières (HH:00)
# ------------------------------------------------------------

df_tr["minute"] = df_tr["date_heure_utc"].dt.minute

nb_avant = len(df_tr)

df_tr = df_tr[df_tr["minute"] == 0].copy()

print(f"  Lignes à HH:00 : {len(df_tr):,} / {nb_avant:,}")

# ------------------------------------------------------------
# Doublons
# ------------------------------------------------------------

doublons = df_tr["date_heure_utc"].duplicated().sum()

if doublons > 0:
    print(f"  {doublons} doublons supprimés")
    df_tr = df_tr.drop_duplicates(
        subset=["date_heure_utc"],
        keep="first",
    )

# ------------------------------------------------------------
# Reconstruction des colonnes locales
# ------------------------------------------------------------

df_tr["date_heure_locale"] = df_tr["date_heure_utc"].dt.tz_convert(TIMEZONE)
df_tr["date"] = df_tr["date_heure_locale"].dt.date
df_tr["heure"] = df_tr["date_heure_locale"].dt.hour
df_tr["jour_semaine"] = df_tr["date_heure_locale"].dt.dayofweek
df_tr["mois"] = df_tr["date_heure_locale"].dt.month
df_tr["annee"] = df_tr["date_heure_locale"].dt.year
df_tr["weekend"] = (df_tr["jour_semaine"] >= 5).astype(int)

# ------------------------------------------------------------
# Consommation horaire
# ------------------------------------------------------------

df_tr["consommation_horaire"] = df_tr["consommation"]

# ------------------------------------------------------------
# Jour complet (calculé après fusion)
# ------------------------------------------------------------

df_tr["jour_complet"] = True  # temporaire

# ------------------------------------------------------------
# Colonnes finales (identiques à eco2mix_national_horaire.csv)
# ------------------------------------------------------------

df_tr["perimetre"] = "France"
df_tr["nature"] = "Temps réel"

colonnes_finales = [
    "date_heure",
    "date_heure_utc",
    "date",
    "heure",
    "minute",
    "jour_semaine",
    "weekend",
    "mois",
    "annee",
    "perimetre",
    "nature",
    "consommation",
    "consommation_horaire",
    "jour_complet",
]

# On reconstruit date_heure locale pour cohérence
df_tr["date_heure"] = df_tr["date_heure_locale"]

df_tr = df_tr[colonnes_finales].copy()


# ============================================================
# 4. FUSION
# ============================================================

print("\n" + "=" * 70)
print("FUSION DES DEUX SOURCES")
print("=" * 70)

# ------------------------------------------------------------
# Éviter les doublons entre les deux datasets
# ------------------------------------------------------------
#
# On garde les données consolidées (plus fiables) et on
# supprime les données temps réel pour les dates déjà
# présentes.
# ------------------------------------------------------------

dates_consolidees = set(df_consolide["date_heure_utc"])

df_tr_filtre = df_tr[
    ~df_tr["date_heure_utc"].isin(dates_consolidees)
].copy()

print(f"  Données consolidées : {len(df_consolide):,}")
print(f"  Données temps réel : {len(df_tr):,}")
print(f"  Temps réel après suppression des doublons : {len(df_tr_filtre):,}")

# ------------------------------------------------------------
# Concaténation
# ------------------------------------------------------------

df_fusion = pd.concat(
    [df_consolide, df_tr_filtre],
    ignore_index=True,
)

df_fusion = df_fusion.sort_values("date_heure_utc").reset_index(drop=True)

# ------------------------------------------------------------
# Recalcul du jour complet
# ------------------------------------------------------------

heures_par_jour = (
    df_fusion
    .groupby("date")["heure"]
    .nunique()
)

jours_24h = set(heures_par_jour[heures_par_jour == 24].index)

df_fusion["jour_complet"] = df_fusion["date"].isin(jours_24h)

print(f"\n  Total après fusion : {len(df_fusion):,}")


# ============================================================
# 5. VERIFICATIONS
# ============================================================

print("\n" + "=" * 70)
print("VÉRIFICATIONS FINALES")
print("=" * 70)

# Pas de doublons temporels
if df_fusion["date_heure_utc"].duplicated().any():
    raise ValueError("Doublons temporels dans la fusion !")

# Tri chronologique
assert df_fusion["date_heure_utc"].is_monotonic_increasing

print(f"  Période finale :")
print(f"    {df_fusion['date_heure_utc'].min()} -> {df_fusion['date_heure_utc'].max()}")

print(f"  Nombre de jours : {df_fusion['date'].nunique():,}")
print(f"  Jours avec 24h : {len(jours_24h):,}")


# ============================================================
# 6. SAUVEGARDE
# ============================================================

df_fusion.to_csv(
    OUTPUT_FILE,
    sep=";",
    index=False,
)

print(f"\nFichier fusionné sauvegardé : {OUTPUT_FILE}")


# ============================================================
# 7. RAPPORT
# ============================================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("RAPPORT DE FUSION DES DONNÉES éCO2mix\n")
    f.write("=" * 70 + "\n\n")

    f.write("SOURCES\n")
    f.write("-" * 70 + "\n")
    f.write(f"Consolidé : {HORAIRE_CONSOLIDE.name}\n")
    f.write(f"Temps réel : {TEMPS_REEL_BRUT.name}\n\n")

    f.write("RÉSULTAT\n")
    f.write("-" * 70 + "\n")
    f.write(f"Lignes consolidées : {len(df_consolide):,}\n")
    f.write(f"Lignes temps réel  : {len(df_tr):,}\n")
    f.write(f"Lignes temps réel conservées : {len(df_tr_filtre):,}\n")
    f.write(f"Total après fusion : {len(df_fusion):,}\n\n")

    f.write(f"Première date : {df_fusion['date_heure_utc'].min()}\n")
    f.write(f"Dernière date : {df_fusion['date_heure_utc'].max()}\n")
    f.write(f"Nombre de jours : {df_fusion['date'].nunique()}\n")
    f.write(f"Jours avec 24h : {len(jours_24h)}\n\n")

    f.write("STRATÉGIE\n")
    f.write("-" * 70 + "\n")
    f.write("En cas de conflit sur un timestamp UTC, les données\n")
    f.write("consolidées sont prioritaires car plus fiables.\n")

print(f"Rapport sauvegardé : {REPORT_FILE}")

print("\n" + "=" * 70)
print("FUSION TERMINÉE")
print("=" * 70)
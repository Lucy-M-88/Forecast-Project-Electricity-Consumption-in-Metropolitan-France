# -*- coding: utf-8 -*-

"""
08_consommation_regionale.py

Télécharge et prépare la consommation électrique régionale éCO2mix.

Lancement :
    python 08_consommation_regionale.py
"""

from pathlib import Path
import io
import time
import unicodedata

import numpy as np
import pandas as pd
import requests


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FICHIER_SORTIE = PROCESSED_DIR / "eco2mix_regional_horaire.csv"

API_BASE = (
    "https://odre.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/"
)

URL_CONSOLIDE = f"{API_BASE}eco2mix-regional-cons-def/exports/csv"
URL_TEMPS_REEL = f"{API_BASE}eco2mix-regional-tr/exports/csv"

REGIONS_METROPOLE = {
    "11": "Île-de-France",
    "24": "Centre-Val de Loire",
    "27": "Bourgogne-Franche-Comté",
    "28": "Normandie",
    "32": "Hauts-de-France",
    "44": "Grand Est",
    "52": "Pays de la Loire",
    "53": "Bretagne",
    "75": "Nouvelle-Aquitaine",
    "76": "Occitanie",
    "84": "Auvergne-Rhône-Alpes",
    "93": "Provence-Alpes-Côte d'Azur",
}

TIMEZONE = "Europe/Paris"
REQUEST_TIMEOUT = 240


# ============================================================
# 2. UTILITAIRES
# ============================================================

def normalize_code(v):
    """
    Normalise un code région quel que soit son format :
        - int 84        -> "84"
        - float 84.0    -> "84"
        - str "84"      -> "84"
        - str "84.0"    -> "84"
        - int 1         -> "01"
    """

    if pd.isna(v):
        return ""

    s = str(v).strip()

    # Enlever le .0 des floats
    if s.endswith(".0"):
        s = s[:-2]

    # Padding à 2 caractères
    if len(s) == 1 and s.isdigit():
        s = "0" + s

    return s


def normaliser_texte(s):
    """Normalise un texte : minuscules, sans accents, sans tirets."""
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("-", " ").replace("'", " ")
    s = " ".join(s.split())
    return s


NOMS_NORMALISES = {
    normaliser_texte(v): k for k, v in REGIONS_METROPOLE.items()
}


# ============================================================
# 3. TÉLÉCHARGEMENT
# ============================================================

def telecharger(url, nom):

    print(f"\nTéléchargement {nom}...")

    params = {
        "delimiter": ";",
        "timezone": "Europe/Paris",
        "use_labels": "false",
    }

    try:
        t0 = time.time()
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERREUR : {e}")
        return None

    print(f"  {len(r.content) / 1024**2:.1f} Mo en {time.time() - t0:.0f}s")

    df = pd.read_csv(io.BytesIO(r.content), sep=";", low_memory=False)
    print(f"  {len(df):,} lignes, {len(df.columns)} colonnes")

    return df


# ============================================================
# 4. DÉTECTION DES COLONNES
# ============================================================

def detecter_colonne_code(df):
    for c in [
        "code_insee_region", "code_region", "code_insee_de_la_region",
        "region", "code",
    ]:
        if c in df.columns:
            return c
    return None


def detecter_colonne_nom(df):
    for c in ["libelle_region", "nom_region", "libelle_de_la_region"]:
        if c in df.columns:
            return c
    return None


# ============================================================
# 5. PRÉPARATION
# ============================================================

def preparer_dataset(df, nom):

    if df is None or df.empty:
        return None

    col_code = detecter_colonne_code(df)
    col_nom = detecter_colonne_nom(df)

    if col_code is None and col_nom is None:
        print(f"  ATTENTION {nom} : ni code ni nom de région")
        print(f"    Colonnes : {list(df.columns)}")
        return None

    # --------------------------------------------------------
    # Code région (normalisation robuste)
    # --------------------------------------------------------

    if col_code is not None:
        df["code_region"] = df[col_code].apply(normalize_code)
    else:
        df["code_region"] = ""

    # --------------------------------------------------------
    # Nom région
    # --------------------------------------------------------

    if col_nom is not None:
        df["nom_region"] = df[col_nom].astype(str)
    else:
        df["nom_region"] = ""

    # --------------------------------------------------------
    # Diagnostic : codes trouvés
    # --------------------------------------------------------

    codes_presents = sorted(df["code_region"].dropna().unique())
    codes_presents = [c for c in codes_presents if c]

    print(f"  Codes région trouvés (avant filtre) : {codes_presents[:30]}")

    # --------------------------------------------------------
    # Filtre par code OU nom
    # --------------------------------------------------------

    codes_valides = set(REGIONS_METROPOLE.keys())

    masque_code = df["code_region"].isin(codes_valides)

    noms_norm = df["nom_region"].apply(normaliser_texte)
    masque_nom = noms_norm.isin(NOMS_NORMALISES.keys())

    df = df[masque_code | masque_nom].copy()

    if df.empty:
        print(f"  ATTENTION {nom} : aucune région métropole")
        return None

    # --------------------------------------------------------
    # Re-codage à partir du nom
    # --------------------------------------------------------

    noms_norm = df["nom_region"].apply(normaliser_texte)
    code_depuis_nom = noms_norm.map(NOMS_NORMALISES)

    df["code_region"] = df["code_region"].where(
        df["code_region"].isin(codes_valides),
        code_depuis_nom,
    )

    df["nom_region"] = df["code_region"].map(REGIONS_METROPOLE)

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    if "date_heure" not in df.columns:
        return None

    df["date_heure_utc"] = pd.to_datetime(
        df["date_heure"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["date_heure_utc"]).copy()

    df["date_heure"] = df["date_heure_utc"].dt.tz_convert(TIMEZONE)
    df["date"] = df["date_heure"].dt.date
    df["heure"] = df["date_heure"].dt.hour
    df["minute"] = df["date_heure"].dt.minute

    df = df[df["minute"] == 0].copy()

    # --------------------------------------------------------
    # Consommation
    # --------------------------------------------------------

    if "consommation" not in df.columns:
        return None

    df["consommation"] = pd.to_numeric(df["consommation"], errors="coerce")
    df = df.dropna(subset=["consommation"]).copy()
    df["consommation_horaire"] = df["consommation"]

    if "nature" not in df.columns:
        df["nature"] = nom

    # --------------------------------------------------------
    # Doublons
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["date_heure_utc", "code_region"], keep="first"
    )

    df = df.sort_values(
        ["date_heure_utc", "code_region"]
    ).reset_index(drop=True)

    df = df[[
        "date_heure_utc", "date_heure", "date", "heure", "minute",
        "code_region", "nom_region",
        "consommation", "consommation_horaire", "nature",
    ]].copy()

    print(
        f"  {len(df):,} lignes, "
        f"{df['code_region'].nunique()} régions, "
        f"{df['date'].nunique()} jours"
    )
    print(f"    Codes conservés : {sorted(df['code_region'].unique())}")

    return df


# ============================================================
# 6. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PRÉPARATION DE LA CONSOMMATION RÉGIONALE éCO2mix")
    print("=" * 70)

    df_cons = preparer_dataset(
        telecharger(URL_CONSOLIDE, "consolidé"), "consolidé"
    )

    df_tr = preparer_dataset(
        telecharger(URL_TEMPS_REEL, "temps réel"), "temps réel"
    )

    frames = []

    if df_cons is not None and len(df_cons) > 0:
        frames.append(df_cons)

    if df_tr is not None and len(df_tr) > 0:
        if df_cons is not None and len(df_cons) > 0:
            cles_cons = set(
                zip(df_cons["date_heure_utc"], df_cons["code_region"])
            )
            masque = [
                (t, r) not in cles_cons
                for t, r in zip(df_tr["date_heure_utc"], df_tr["code_region"])
            ]
            df_tr = df_tr[masque].copy()
        if len(df_tr) > 0:
            frames.append(df_tr)

    if not frames:
        raise ValueError("Aucune donnée récupérée.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(
        ["date_heure_utc", "code_region"]
    ).reset_index(drop=True)

    df.to_csv(FICHIER_SORTIE, sep=";", index=False, encoding="utf-8")

    print()
    print("=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    print(f"Fichier : {FICHIER_SORTIE}")
    print(f"Lignes  : {len(df):,}")
    print(f"Régions : {df['code_region'].nunique()}")

    codes_attendus = set(REGIONS_METROPOLE.keys())
    codes_presents = set(df["code_region"].unique())
    manquantes = codes_attendus - codes_presents

    if manquantes:
        print(f"\n⚠ Régions MANQUANTES dans les données :")
        for c in sorted(manquantes):
            print(f"    {c} : {REGIONS_METROPOLE[c]}")
    else:
        print(f"\n✓ Les 13 régions métropole sont présentes.")

    print()


if __name__ == "__main__":
    main()
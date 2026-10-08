# -*- coding: utf-8 -*-

"""
04a_reduire_meteofrance.py

Objectif
--------
Alléger les fichiers Météo-France "Données climatologiques de base"
en ne conservant que les données à partir du 1er janvier 2010.

Entrée :
    data/raw/meteofrance/
        Q_{dept}_previous-1950-2024_RR-T-Vent.csv
        Q_{dept}_latest-2025-2026_RR-T-Vent.csv

Sortie :
    data/processed/meteofrance_reduit/
        Q_{dept}_reduit.csv

Le script :
    - parcourt tous les fichiers du dossier d'entrée ;
    - lit uniquement les colonnes nécessaires ;
    - filtre les lignes avec AAAAMMJJ >= 20100101 ;
    - supprime les stations qui n'ont pas assez de données ;
    - sauvegarde dans un dossier séparé.

IMPORTANT :
    - Les fichiers originaux ne sont PAS modifiés.
    - Les codes qualité sont conservés.
    - Aucune conversion d'unité n'est faite ici.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "data" / "raw" / "meteofrance"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "meteofrance_reduit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Date minimale conservée
# ------------------------------------------------------------
#
# On garde à partir du 1er janvier 2010.
# C'est largement suffisant pour couvrir la période du dataset
# de prévision (2012-2026) avec un peu de marge pour les
# indicateurs de tendance (J-1, J-2, moyennes glissantes).

DATE_MIN = 20100101

# ------------------------------------------------------------
# Colonnes à conserver
# ------------------------------------------------------------
#
# On ne garde que les variables utiles au projet et leurs
# codes qualité associés.

COLONNES_IDENTIFICATION = [
    "NUM_POSTE",
    "NOM_USUEL",
    "LAT",
    "LON",
    "ALTI",
    "AAAAMMJJ",
]

COLONNES_MESURES = [
    # Précipitations
    "RR", "QRR",
    # Températures
    "TN", "QTN",
    "TX", "QTX",
    "TM", "QTM",
    "TAMPLI", "QTAMPLI",
    "TNSOL", "QTNSOL",
    # Gel
    "DG", "QDG",
    # Vent
    "FFM", "QFFM",
    "FXY", "QFXY",
    # Durée des précipitations
    "DRR", "QDRR",
]

COLONNES_A_GARDER = COLONNES_IDENTIFICATION + COLONNES_MESURES

# ------------------------------------------------------------
# Seuil de remplissage par station
# ------------------------------------------------------------
#
# Une station est conservée seulement si elle a un taux
# de remplissage suffisant sur la variable TM (température
# moyenne), qui est notre variable la plus importante.

TAUX_REMPLISSAGE_MIN = 0.80  # 80 %


# ============================================================
# 2. FONCTIONS
# ============================================================

def lire_fichier_mf(chemin):
    """
    Lit un fichier Météo-France en ne gardant que les colonnes
    utiles.

    Parameters
    ----------
    chemin : Path
        Chemin vers le fichier CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame filtré sur les colonnes utiles.
    """

    print(f"\nLecture : {chemin.name}")

    df = pd.read_csv(
        chemin,
        sep=";",
        encoding="latin-1",
        low_memory=False,
    )

    print(f"  Lignes brutes : {len(df):,}")
    print(f"  Colonnes brutes : {len(df.columns)}")

    # --------------------------------------------------------
    # Vérification des colonnes
    # --------------------------------------------------------

    colonnes_presentes = [
        c for c in COLONNES_A_GARDER
        if c in df.columns
    ]

    colonnes_manquantes = [
        c for c in COLONNES_A_GARDER
        if c not in df.columns
    ]

    if colonnes_manquantes:
        print(
            f"  ATTENTION : colonnes manquantes : "
            f"{colonnes_manquantes}"
        )

    df = df[colonnes_presentes].copy()

    # --------------------------------------------------------
    # Conversion de la date
    # --------------------------------------------------------

    df["AAAAMMJJ"] = pd.to_numeric(
        df["AAAAMMJJ"],
        errors="coerce"
    )

    df = df.dropna(subset=["AAAAMMJJ"]).copy()
    df["AAAAMMJJ"] = df["AAAAMMJJ"].astype(int)

    return df


def filtrer_apres_2010(df):
    """
    Ne conserve que les lignes avec AAAAMMJJ >= DATE_MIN.
    """

    masque = df["AAAAMMJJ"] >= DATE_MIN

    df_filtre = df.loc[masque].copy()

    print(
        f"  Lignes après filtre {DATE_MIN} : "
        f"{len(df_filtre):,}"
    )

    return df_filtre


def supprimer_stations_incompletes(df):
    """
    Supprime les stations qui n'ont pas assez de données.

    Critère : taux de remplissage de TM >= TAUX_REMPLISSAGE_MIN.
    """

    # --------------------------------------------------------
    # Calcul du taux de remplissage par station
    # --------------------------------------------------------

    remplissage = (
        df
        .groupby("NUM_POSTE")["TM"]
        .apply(lambda x: x.notna().mean())
    )

    stations_ok = remplissage[
        remplissage >= TAUX_REMPLISSAGE_MIN
    ].index

    nb_stations_avant = df["NUM_POSTE"].nunique()
    nb_stations_apres = len(stations_ok)

    print(
        f"  Stations avant filtre : {nb_stations_avant}"
    )

    print(
        f"  Stations après filtre : {nb_stations_apres} "
        f"(seuil {TAUX_REMPLISSAGE_MIN:.0%} sur TM)"
    )

    df_filtre = df[
        df["NUM_POSTE"].isin(stations_ok)
    ].copy()

    return df_filtre


def traiter_fichier(chemin):
    """
    Pipeline complet sur un fichier : lecture, filtre temporel,
    filtre des stations incomplètes, sauvegarde.
    """

    df = lire_fichier_mf(chemin)

    df = filtrer_apres_2010(df)

    if len(df) == 0:
        print("  Aucune donnée après 2010. Fichier ignoré.")
        return None

    df = supprimer_stations_incompletes(df)

    if len(df) == 0:
        print("  Aucune station valide. Fichier ignoré.")
        return None

    # --------------------------------------------------------
    # Extraction du département depuis le nom du fichier
    # --------------------------------------------------------
    #
    # Nom typique :
    #   Q_75_previous-1950-2024_RR-T-Vent.csv
    #   Q_75_latest-2025-2026_RR-T-Vent.csv
    #
    # On extrait le numéro entre "Q_" et "_previous" ou "_latest".
    # --------------------------------------------------------

    nom = chemin.stem  # sans extension

    parties = nom.split("_")

    if len(parties) < 3:
        print(f"  Nom de fichier inattendu : {nom}")
        return None

    dept = parties[1]

    # --------------------------------------------------------
    # Nom du fichier de sortie
    # --------------------------------------------------------

    chemin_sortie = (
        OUTPUT_DIR
        / f"Q_{dept}_reduit.csv"
    )

    # --------------------------------------------------------
    # Si le fichier de sortie existe déjà pour ce département,
    # on concatène au lieu d'écraser (pour gérer previous + latest).
    # --------------------------------------------------------

    if chemin_sortie.exists():

        print(
            f"  Fichier existant détecté : "
            f"{chemin_sortie.name}"
        )

        df_existant = pd.read_csv(
            chemin_sortie,
            sep=";",
            encoding="latin-1",
            low_memory=False,
        )

        df = pd.concat(
            [df_existant, df],
            ignore_index=True
        )

        df = df.drop_duplicates(
            subset=["NUM_POSTE", "AAAAMMJJ"],
            keep="first"
        )

    # --------------------------------------------------------
    # Tri
    # --------------------------------------------------------

    df = df.sort_values(
        ["NUM_POSTE", "AAAAMMJJ"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    df.to_csv(
        chemin_sortie,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    print(
        f"  Sauvegardé : {chemin_sortie.name} "
        f"({len(df):,} lignes, "
        f"{df['NUM_POSTE'].nunique()} stations)"
    )

    return chemin_sortie


# ============================================================
# 3. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("RÉDUCTION DES FICHIERS MÉTÉO-FRANCE")
    print("=" * 70)

    print(f"\nDossier d'entrée : {INPUT_DIR}")
    print(f"Dossier de sortie : {OUTPUT_DIR}")
    print(f"Date minimale : {DATE_MIN}")
    print(f"Taux de remplissage minimum : {TAUX_REMPLISSAGE_MIN:.0%}")

    # --------------------------------------------------------
    # Vérification du dossier d'entrée
    # --------------------------------------------------------

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {INPUT_DIR}\n"
            f"Vérifie que les fichiers Météo-France sont bien "
            f"placés dans {INPUT_DIR}."
        )

    # --------------------------------------------------------
    # Liste des fichiers
    # --------------------------------------------------------

    fichiers = sorted(
        INPUT_DIR.glob("*.csv")
    )

    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans {INPUT_DIR}"
        )

    print(f"\nNombre de fichiers à traiter : {len(fichiers)}")

    # --------------------------------------------------------
    # Traitement
    # --------------------------------------------------------

    fichiers_traites = []

    for chemin in fichiers:

        try:
            resultat = traiter_fichier(chemin)

            if resultat is not None:
                fichiers_traites.append(resultat)

        except Exception as e:

            print(
                f"\nERREUR lors du traitement de "
                f"{chemin.name} : {e}"
            )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    print(
        f"Fichiers traités avec succès : "
        f"{len(fichiers_traites)} / {len(fichiers)}"
    )

    print("\nFichiers produits :")

    for chemin in sorted(OUTPUT_DIR.glob("*.csv")):
        taille_mo = chemin.stat().st_size / 1024**2
        print(
            f"  - {chemin.name} "
            f"({taille_mo:.1f} Mo)"
        )

    # --------------------------------------------------------
    # Taille totale
    # --------------------------------------------------------

    taille_totale = sum(
        f.stat().st_size
        for f in OUTPUT_DIR.glob("*.csv")
    )

    print(
        f"\nTaille totale du dossier réduit : "
        f"{taille_totale / 1024**2:.1f} Mo"
    )

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()
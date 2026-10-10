# -*- coding: utf-8 -*-

"""
04b_indicateurs_externes.py

Objectif
--------
Intégrer les indicateurs économiques externes dans le dataset.

Sources :
    - INSEE : Indice de la production industrielle (IPI) manufacturière
      Série : 010768688
      URL : https://www.insee.fr/fr/statistiques/serie/010768688
      Périodicité : mensuelle
      Unité : variation mensuelle (%)

IMPORTANT
---------
- L'IPI est publié avec un délai d'environ 40-45 jours après le mois
  concerné. À 14h le jour J, seul l'IPI du mois M-1 ou M-2 est
  disponible.
- Pour éviter toute fuite d'information, on utilise pour chaque jour J
  le dernier IPI PUBLIÉ à 14h J (donc M-1 ou M-2).
- Cet indicateur est mensuel ; il est propagé à tous les jours du mois
  suivant selon la date de publication.

Entrée :
    data/raw/insee/valeurs_mensuelles.csv
    data/raw/insee/caracteristiques.csv

Sortie :
    data/processed/ipi_mensuel.csv
    data/processed/ipi_operationnel.csv
    data/processed/rapport_indicateurs_externes.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw" / "insee"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

VALEURS_FILE = RAW_DIR / "valeurs_mensuelles.csv"
CARACTERISTIQUES_FILE = RAW_DIR / "caracteristiques.csv"

OUTPUT_IPI_MENSUEL = PROCESSED_DIR / "ipi_mensuel.csv"
OUTPUT_IPI_OPERATIONNEL = PROCESSED_DIR / "ipi_operationnel.csv"
RAPPORT = PROCESSED_DIR / "rapport_indicateurs_externes.txt"

# Délai de publication de l'IPI en jours
# (délai INSEE ~40 jours après la fin du mois)
DELAI_PUBLICATION_JOURS = 40


# ============================================================
# 2. CHARGEMENT DE L'IPI
# ============================================================

def lire_valeurs_mensuelles_insee(chemin):
    """
    Lit le fichier INSEE de valeurs mensuelles.

    Le fichier a une structure particulière :
        - Ligne 0 : libellé
        - Ligne 1 : idBank
        - Ligne 2 : mises à jour
        - Ligne 3 : "Période" (en-tête des données)
        - Ligne 4+ : données (Période, Valeur, Code qualité)

    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes 'mois' et 'ipi_variation'.
    """

    print(f"\nLecture : {chemin.name}")

    # Lecture brute sans header
    df = pd.read_csv(
        chemin,
        sep=";",
        header=None,
        dtype=str,
        encoding="utf-8-sig",
    )

    print(f"  Lignes brutes : {len(df)}")
    print(f"  Colonnes brutes : {len(df.columns)}")

    # --------------------------------------------------------
    # Extraction des données
    # --------------------------------------------------------
    #
    # On part de la ligne 4 (index 4) qui contient les données.
    # Les colonnes 0 (période) et 1 (valeur) sont utilisées.
    # --------------------------------------------------------

    if len(df) < 5:
        raise ValueError(
            f"Fichier trop court : {len(df)} lignes. "
            "Format inattendu."
        )

    # On garde les lignes à partir de l'index 4
    # On garde les colonnes 0 (période) et 1 (valeur)
    df_data = df.iloc[4:, [0, 1]].copy()
    df_data.columns = ["mois", "ipi_variation"]

    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------

    # Suppression des lignes vides
    df_data = df_data.dropna(subset=["mois"])
    df_data = df_data[df_data["mois"].str.strip() != ""]

    # Filtrage sur le format YYYY-MM
    masque_format = df_data["mois"].str.match(
        r"^\d{4}-\d{2}$", na=False
    )
    df_data = df_data[masque_format].copy()

    # Conversion numérique
    df_data["ipi_variation"] = pd.to_numeric(
        df_data["ipi_variation"],
        errors="coerce",
    )

    # Suppression des valeurs manquantes
    df_data = df_data.dropna(subset=["ipi_variation"]).copy()

    # Conversion de la période en datetime (premier jour du mois)
    df_data["mois"] = pd.to_datetime(
        df_data["mois"] + "-01",
        format="%Y-%m-%d",
        errors="coerce",
    )

    df_data = df_data.dropna(subset=["mois"]).copy()

    df_data = df_data.sort_values("mois").reset_index(drop=True)

    print(f"  Données extraites : {len(df_data)} mois")
    print(f"  Période : {df_data['mois'].min().date()} -> "
          f"{df_data['mois'].max().date()}")

    return df_data


# ============================================================
# 3. LECTURE DES CARACTÉRISTIQUES (pour le rapport)
# ============================================================

def lire_caracteristiques(chemin):
    """
    Lit le fichier de caractéristiques pour vérification.
    """

    print(f"\nLecture : {chemin.name}")

    df = pd.read_csv(
        chemin,
        sep=";",
        header=None,
        dtype=str,
        encoding="utf-8-sig",
    )

    # La première ligne contient les noms de colonnes
    if len(df) > 0:
        noms = df.iloc[0].tolist()
        valeurs = df.iloc[1].tolist() if len(df) > 1 else []

        caracteristiques = dict(zip(noms, valeurs))

    else:
        caracteristiques = {}

    return caracteristiques


# ============================================================
# 4. CONSTRUCTION DE L'IPI OPÉRATIONNEL
# ============================================================

def construire_ipi_operationnel(df_ipi_mensuel):
    """
    Construit un IPI journalier "opérationnel" qui reflète
    ce qui serait réellement disponible à 14h chaque jour.

    Principe :
        Pour chaque jour J, on utilise le dernier IPI publié
        à 14h J, c'est-à-dire l'IPI du mois M-2 (délai de
        publication d'environ 40-45 jours).

    Exemple :
        - 15 octobre 2026 à 14h → IPI d'août 2026 disponible
        - IPI de septembre 2026 publié ~10 octobre, donc
          disponible aussi

    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes 'date' et 'ipi_variation'.
    """

    print("\nConstruction de l'IPI opérationnel (journalier)...")

    # --------------------------------------------------------
    # Période : du premier au dernier jour des IPI mensuels
    # --------------------------------------------------------

    date_min = df_ipi_mensuel["mois"].min()
    date_max = df_ipi_mensuel["mois"].max()

    # On génère toutes les dates journalières
    dates_journalieres = pd.date_range(
        start=date_min,
        end=date_max + pd.offsets.MonthEnd(0),
        freq="D",
    )

    df_dates = pd.DataFrame({"date": dates_journalieres})

    # --------------------------------------------------------
    # Pour chaque jour, trouver le dernier IPI publié
    # --------------------------------------------------------

    def trouver_ipi_disponible(date_j):
        """
        Trouve l'IPI du mois M-2 par rapport à date_j.

        Returns
        -------
        float or None
        """
        # Mois cible : M-2
        mois_cible = (
            pd.Timestamp(date_j).replace(day=1)
            - pd.DateOffset(months=2)
        )

        # Recherche dans le dataset IPI
        row = df_ipi_mensuel[
            df_ipi_mensuel["mois"] == mois_cible
        ]

        if len(row) == 0:
            return np.nan

        return row.iloc[0]["ipi_variation"]

    df_dates["ipi_variation"] = df_dates["date"].apply(
        trouver_ipi_disponible
    )

    print(f"  Dates : {len(df_dates):,}")
    print(f"  Valeurs disponibles : "
          f"{df_dates['ipi_variation'].notna().sum():,}")
    print(f"  Valeurs manquantes : "
          f"{df_dates['ipi_variation'].isna().sum():,}")

    return df_dates


# ============================================================
# 5. RAPPORT
# ============================================================

def ecrire_rapport(
    df_ipi_mensuel,
    df_ipi_operationnel,
    caracteristiques,
):

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAPPORT, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("RAPPORT - INDICATEURS ÉCONOMIQUES EXTERNES\n")
        f.write("=" * 70 + "\n\n")

        # ----------------------------------------------------
        # Indicateur
        # ----------------------------------------------------

        f.write("INDICATEUR UTILISÉ\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Indice de la production industrielle (IPI)\n"
        )
        f.write(
            "Industrie manufacturière (NAF A10-CZ)\n"
        )
        f.write(
            "Série INSEE : 010768688\n"
        )
        f.write(
            "URL : https://www.insee.fr/fr/statistiques/serie/010768688\n"
        )
        f.write(
            "Périodicité : mensuelle\n"
        )
        f.write(
            "Unité : variation mensuelle (%)\n"
        )
        f.write(
            "Correction : CVS-CJO\n"
        )
        f.write(
            "Base : 2021\n\n"
        )

        # ----------------------------------------------------
        # Caractéristiques INSEE
        # ----------------------------------------------------

        if caracteristiques:
            f.write("CARACTÉRISTIQUES INSEE\n")
            f.write("-" * 70 + "\n")
            for nom, valeur in caracteristiques.items():
                if pd.notna(valeur) and str(valeur).strip() != "":
                    f.write(f"  {nom} : {valeur}\n")
            f.write("\n")

        # ----------------------------------------------------
        # Statistiques
        # ----------------------------------------------------

        f.write("STATISTIQUES IPI MENSUEL\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Nombre de mois : {len(df_ipi_mensuel)}\n"
        )
        f.write(
            f"Première date : {df_ipi_mensuel['mois'].min().date()}\n"
        )
        f.write(
            f"Dernière date : {df_ipi_mensuel['mois'].max().date()}\n"
        )
        f.write(
            f"Variation moyenne : "
            f"{df_ipi_mensuel['ipi_variation'].mean():.2f} %\n"
        )
        f.write(
            f"Variation min : "
            f"{df_ipi_mensuel['ipi_variation'].min():.2f} %\n"
        )
        f.write(
            f"Variation max : "
            f"{df_ipi_mensuel['ipi_variation'].max():.2f} %\n"
        )
        f.write(
            f"Écart-type : "
            f"{df_ipi_mensuel['ipi_variation'].std():.2f} %\n\n"
        )

        # ----------------------------------------------------
        # IPI opérationnel
        # ----------------------------------------------------

        f.write("IPI OPÉRATIONNEL (JOURNALIER)\n")
        f.write("-" * 70 + "\n")
        f.write(
            "L'IPI est propagé à tous les jours du mois suivant\n"
            "la publication, avec un décalage de 2 mois.\n\n"
        )
        f.write(
            f"Nombre de jours : {len(df_ipi_operationnel)}\n"
        )
        f.write(
            f"Valeurs disponibles : "
            f"{df_ipi_operationnel['ipi_variation'].notna().sum()}\n"
        )
        f.write(
            f"Valeurs manquantes : "
            f"{df_ipi_operationnel['ipi_variation'].isna().sum()}\n\n"
        )

        # ----------------------------------------------------
        # Fuite d'information
        # ----------------------------------------------------

        f.write("AUDIT - FUITE D'INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Délai de publication de l'IPI : environ 40 à 45 jours\n"
            "après la fin du mois concerné.\n\n"
        )
        f.write(
            "Règle appliquée : pour chaque jour J, l'IPI utilisé\n"
            "est celui du mois M-2 (pour garantir qu'il soit\n"
            "publié à 14h J).\n\n"
        )
        f.write(
            "Cela garantit l'absence de fuite d'information.\n\n"
        )

        f.write("=" * 70 + "\n")
        f.write("FIN DU RAPPORT\n")
        f.write("=" * 70 + "\n")


# ============================================================
# 6. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("INDICATEURS ÉCONOMIQUES EXTERNES")
    print("Projet de prévision - France métropolitaine")
    print("=" * 70)
    print()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Vérification
    # --------------------------------------------------------

    if not VALEURS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {VALEURS_FILE}\n"
            "Place les fichiers INSEE dans data/raw/insee/"
        )

    # --------------------------------------------------------
    # Lecture
    # --------------------------------------------------------

    df_ipi = lire_valeurs_mensuelles_insee(VALEURS_FILE)

    caracteristiques = {}
    if CARACTERISTIQUES_FILE.exists():
        caracteristiques = lire_caracteristiques(
            CARACTERISTIQUES_FILE
        )

    # --------------------------------------------------------
    # Sauvegarde IPI mensuel
    # --------------------------------------------------------

    df_ipi.to_csv(
        OUTPUT_IPI_MENSUEL,
        sep=";",
        index=False,
    )

    print(f"\nIPI mensuel sauvegardé : {OUTPUT_IPI_MENSUEL}")

    # --------------------------------------------------------
    # Construction de l'IPI opérationnel
    # --------------------------------------------------------

    df_ipi_op = construire_ipi_operationnel(df_ipi)

    df_ipi_op.to_csv(
        OUTPUT_IPI_OPERATIONNEL,
        sep=";",
        index=False,
    )

    print(f"IPI opérationnel sauvegardé : {OUTPUT_IPI_OPERATIONNEL}")

    # --------------------------------------------------------
    # Rapport
    # --------------------------------------------------------

    ecrire_rapport(df_ipi, df_ipi_op, caracteristiques)

    print(f"Rapport sauvegardé : {RAPPORT}")

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INDICATEURS EXTERNES - TERMINÉ")
    print("=" * 70)
    print()
    print(f"Mois couverts : {df_ipi['mois'].min().date()} -> "
          f"{df_ipi['mois'].max().date()}")
    print(f"Nombre de mois : {len(df_ipi)}")
    print(f"Jours couverts : {df_ipi_op['date'].min().date()} -> "
          f"{df_ipi_op['date'].max().date()}")
    print()


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

"""
02_preparation.py

Préparation des données éCO2mix pour la prévision horaire.

Ce script effectue deux étapes :
    ÉTAPE 1 : Préparation des données consolidées
        - Passage au pas horaire (HH:00 uniquement)
        - Aucune moyenne entre demi-heures
        - Aucune interpolation
        - Reconstruction des colonnes calendaires

    ÉTAPE 2 : Fusion avec les données temps réel
        - Application des mêmes transformations
        - Fusion avec priorité aux données consolidées
        - Sauvegarde du fichier fusionné

Entrée :
    data/raw/eco2mix_national_brut.csv
    data/raw/eco2mix_national_temps_reel.csv

Sortie :
    data/processed/eco2mix_national_horaire.csv
    data/processed/eco2mix_national_horaire_fusionne.csv
    data/processed/rapport_preparation.txt

IMPORTANT :
    - AUCUNE moyenne entre 2 demi-heures.
    - Nous utilisons uniquement les observations exactement à HH:00.
    - Aucune interpolation.
    - Aucune valeur artificielle.
    - Les données restent dans le périmètre France.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PARAMÈTRES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Entrées
RAW_CONSOLIDE = BASE_DIR / "data" / "raw" / "eco2mix_national_brut.csv"
RAW_TEMPS_REEL = BASE_DIR / "data" / "raw" / "eco2mix_national_temps_reel.csv"

# Sorties
HORAIRE_CONSOLIDE = BASE_DIR / "data" / "processed" / "eco2mix_national_horaire.csv"
HORAIRE_FUSIONNE = BASE_DIR / "data" / "processed" / "eco2mix_national_horaire_fusionne.csv"

# Rapport unique
RAPPORT_UNIQUE = BASE_DIR / "data" / "processed" / "rapport_preparation.txt"

TIMEZONE = "Europe/Paris"

REQUIRED_COLUMNS = [
    "perimetre",
    "date_heure",
    "consommation",
    "nature",
]


# ============================================================
# 2. ÉTAPE 1 : PRÉPARATION DES DONNÉES CONSOLIDÉES
# ============================================================

def preparer_consolide():
    """
    Prépare les données consolidées au pas horaire.

    Returns
    -------
    dict
        Dictionnaire contenant :
            - df : DataFrame préparé
            - stats : dictionnaire de statistiques
    """

    print("=" * 70)
    print("ÉTAPE 1 : PRÉPARATION DES DONNÉES CONSOLIDÉES")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    print("\nChargement des données brutes consolidées...")

    if not RAW_CONSOLIDE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {RAW_CONSOLIDE}\n"
            "Lance d'abord 01_telechargement.py."
        )

    df = pd.read_csv(
        RAW_CONSOLIDE,
        sep=";",
        low_memory=False,
    )

    print(f"  {len(df):,} lignes chargées")

    # --------------------------------------------------------
    # Vérification des colonnes
    # --------------------------------------------------------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires manquantes : "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Vérification du périmètre
    # --------------------------------------------------------

    print("\nVérification du périmètre...")

    if not (df["perimetre"].astype(str).str.strip() == "France").all():
        raise ValueError(
            "Le fichier contient un périmètre différent de 'France'."
        )

    print("  OK : toutes les observations sont sur France")

    # --------------------------------------------------------
    # Conversion des dates via UTC
    # --------------------------------------------------------

    print("\nConversion des dates...")

    df["date_heure_utc"] = pd.to_datetime(
        df["date_heure"],
        errors="coerce",
        utc=True,
    )

    nb_dates_invalides = df["date_heure_utc"].isna().sum()

    if nb_dates_invalides > 0:
        print(
            f"  Attention : {nb_dates_invalides:,} dates invalides supprimées."
        )

    df = df.dropna(subset=["date_heure_utc"]).copy()

    df["date_heure"] = (
        df["date_heure_utc"]
        .dt.tz_convert(TIMEZONE)
    )

    # --------------------------------------------------------
    # Conversion de la consommation
    # --------------------------------------------------------

    df["consommation"] = pd.to_numeric(
        df["consommation"],
        errors="coerce",
    )

    nb_conso_invalides = df["consommation"].isna().sum()

    print(
        f"  Consommations invalides : {nb_conso_invalides:,}"
    )

    # --------------------------------------------------------
    # Tri chronologique
    # --------------------------------------------------------

    df = df.sort_values("date_heure_utc").reset_index(drop=True)

    # --------------------------------------------------------
    # Doublons physiques (UTC)
    # --------------------------------------------------------

    nb_avant = len(df)

    df = df.drop_duplicates(
        subset=["date_heure_utc"],
        keep="first",
    ).copy()

    nb_doublons = nb_avant - len(df)

    print(f"  Doublons physiques supprimés : {nb_doublons:,}")

    # --------------------------------------------------------
    # Informations calendaires
    # --------------------------------------------------------

    df["date"] = df["date_heure"].dt.date
    df["heure"] = df["date_heure"].dt.hour
    df["minute"] = df["date_heure"].dt.minute

    df["jour_semaine"] = df["date_heure"].dt.dayofweek
    df["mois"] = df["date_heure"].dt.month
    df["annee"] = df["date_heure"].dt.year

    df["weekend"] = (
        df["jour_semaine"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Conservation uniquement des HH:00
    # --------------------------------------------------------

    print("\nSélection des heures entières (HH:00)...")

    nb_total = len(df)
    nb_heures_entieres = (df["minute"] == 0).sum()
    nb_demi_heures = (df["minute"] == 30).sum()

    print(f"  Observations totales : {nb_total:,}")
    print(f"  Observations à HH:00 : {nb_heures_entieres:,}")
    print(f"  Observations à HH:30 : {nb_demi_heures:,}")

    df_horaire = df[df["minute"] == 0].copy()

    df_horaire = df_horaire.sort_values(
        "date_heure_utc"
    ).reset_index(drop=True)

    df_horaire["consommation_horaire"] = df_horaire["consommation"]

    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------

    nb_negatives = (df_horaire["consommation_horaire"] < 0).sum()

    if nb_negatives > 0:
        print(f"  ATTENTION : {nb_negatives} consommations négatives")

    nb_missing = df_horaire["consommation_horaire"].isna().sum()

    print(f"  Consommations manquantes à HH:00 : {nb_missing:,}")

    # --------------------------------------------------------
    # Journées complètes (24h)
    # --------------------------------------------------------

    heures_par_jour = (
        df_horaire
        .groupby("date")["heure"]
        .nunique()
    )

    dates_completes = heures_par_jour[heures_par_jour == 24].index
    dates_incompletes = heures_par_jour[heures_par_jour != 24].index

    print(
        f"  Journées avec 24h : {len(dates_completes):,}"
    )
    print(
        f"  Journées incomplètes : {len(dates_incompletes):,}"
    )

    df_horaire["jour_complet"] = (
        df_horaire["date"].isin(dates_completes)
    )

    # --------------------------------------------------------
    # Ordre final des colonnes
    # --------------------------------------------------------

    columns_finales = [
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

    df_horaire = df_horaire[columns_finales].copy()

    # --------------------------------------------------------
    # Vérifications finales
    # --------------------------------------------------------

    assert (df_horaire["minute"] == 0).all()
    assert df_horaire["consommation_horaire"].equals(
        df_horaire["consommation"]
    )
    assert not df_horaire["date_heure_utc"].duplicated().any()
    assert df_horaire["date_heure_utc"].is_monotonic_increasing

    print("  ✓ Toutes les observations sont à HH:00")
    print("  ✓ Aucune moyenne, aucune interpolation")

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    HORAIRE_CONSOLIDE.parent.mkdir(parents=True, exist_ok=True)

    df_horaire.to_csv(
        HORAIRE_CONSOLIDE,
        sep=";",
        index=False,
    )

    print(f"\n  Fichier préparé : {HORAIRE_CONSOLIDE}")

    stats = {
        "nb_total": nb_total,
        "nb_horaire": len(df_horaire),
        "nb_demi_heures": nb_demi_heures,
        "nb_doublons": nb_doublons,
        "nb_missing": nb_missing,
        "nb_negatives": nb_negatives,
        "nb_dates_completes": len(dates_completes),
        "nb_dates_incompletes": len(dates_incompletes),
    }

    return {"df": df_horaire, "stats": stats}


# ============================================================
# 3. ÉTAPE 2 : FUSION AVEC LES DONNÉES TEMPS RÉEL
# ============================================================

def fusionner_temps_reel(df_consolide):
    """
    Fusionne les données consolidées préparées avec les
    données temps réel.

    Parameters
    ----------
    df_consolide : pd.DataFrame

    Returns
    -------
    dict
        Dictionnaire contenant :
            - df : DataFrame fusionné
            - stats : dictionnaire de statistiques
    """

    print("\n" + "=" * 70)
    print("ÉTAPE 2 : FUSION AVEC LES DONNÉES TEMPS RÉEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Cas où le fichier temps réel est absent
    # --------------------------------------------------------

    if not RAW_TEMPS_REEL.exists():

        print(
            f"\nATTENTION : fichier introuvable {RAW_TEMPS_REEL}\n"
            "La fusion est ignorée. Utilisation du fichier consolidé seul."
        )

        df_consolide.to_csv(
            HORAIRE_FUSIONNE,
            sep=";",
            index=False,
        )

        stats = {
            "tr_absent": True,
            "nb_consolide": len(df_consolide),
            "nb_tr": 0,
            "nb_tr_garde": 0,
            "nb_total": len(df_consolide),
            "nb_jours": df_consolide["date"].nunique(),
            "nb_jours_24h": int(df_consolide["jour_complet"].sum() > 0),
        }

        return {"df": df_consolide, "stats": stats}

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    print("\nChargement des données temps réel...")

    df_tr = pd.read_csv(
        RAW_TEMPS_REEL,
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

    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------

    df_tr = df_tr.dropna(subset=["date_heure_utc"]).copy()

    df_tr["consommation"] = pd.to_numeric(
        df_tr["consommation"],
        errors="coerce",
    )

    df_tr = df_tr.dropna(subset=["consommation"]).copy()

    df_tr = df_tr.sort_values("date_heure_utc").reset_index(drop=True)

    # --------------------------------------------------------
    # HH:00 uniquement
    # --------------------------------------------------------

    df_tr["minute"] = df_tr["date_heure_utc"].dt.minute

    nb_avant = len(df_tr)

    df_tr = df_tr[df_tr["minute"] == 0].copy()

    print(f"  Lignes à HH:00 : {len(df_tr):,} / {nb_avant:,}")

    # --------------------------------------------------------
    # Doublons
    # --------------------------------------------------------

    doublons = df_tr["date_heure_utc"].duplicated().sum()

    if doublons > 0:
        print(f"  {doublons} doublons supprimés")
        df_tr = df_tr.drop_duplicates(
            subset=["date_heure_utc"],
            keep="first",
        )

    # --------------------------------------------------------
    # Reconstruction des colonnes locales
    # --------------------------------------------------------

    df_tr["date_heure_locale"] = df_tr["date_heure_utc"].dt.tz_convert(TIMEZONE)
    df_tr["date"] = df_tr["date_heure_locale"].dt.date
    df_tr["heure"] = df_tr["date_heure_locale"].dt.hour
    df_tr["jour_semaine"] = df_tr["date_heure_locale"].dt.dayofweek
    df_tr["mois"] = df_tr["date_heure_locale"].dt.month
    df_tr["annee"] = df_tr["date_heure_locale"].dt.year
    df_tr["weekend"] = (df_tr["jour_semaine"] >= 5).astype(int)

    df_tr["consommation_horaire"] = df_tr["consommation"]
    df_tr["jour_complet"] = True

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

    df_tr["date_heure"] = df_tr["date_heure_locale"]

    df_tr = df_tr[colonnes_finales].copy()

    # --------------------------------------------------------
    # Fusion avec priorité au consolidé
    # --------------------------------------------------------

    dates_consolidees = set(df_consolide["date_heure_utc"])

    df_tr_filtre = df_tr[
        ~df_tr["date_heure_utc"].isin(dates_consolidees)
    ].copy()

    print(f"\n  Consolidé         : {len(df_consolide):,} lignes")
    print(f"  Temps réel        : {len(df_tr):,} lignes")
    print(f"  Temps réel gardé  : {len(df_tr_filtre):,} lignes")

    df_fusion = pd.concat(
        [df_consolide, df_tr_filtre],
        ignore_index=True,
    )

    df_fusion = df_fusion.sort_values(
        "date_heure_utc"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Recalcul du jour complet
    # --------------------------------------------------------

    heures_par_jour = (
        df_fusion
        .groupby("date")["heure"]
        .nunique()
    )

    jours_24h = set(heures_par_jour[heures_par_jour == 24].index)

    df_fusion["jour_complet"] = df_fusion["date"].isin(jours_24h)

    print(f"\n  Total après fusion : {len(df_fusion):,}")
    print(f"  Nombre de jours : {df_fusion['date'].nunique():,}")
    print(f"  Jours avec 24h : {len(jours_24h):,}")

    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------

    assert not df_fusion["date_heure_utc"].duplicated().any()
    assert df_fusion["date_heure_utc"].is_monotonic_increasing

    print(f"\n  Période finale :")
    print(f"    {df_fusion['date_heure_utc'].min()} -> "
          f"{df_fusion['date_heure_utc'].max()}")

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    df_fusion.to_csv(
        HORAIRE_FUSIONNE,
        sep=";",
        index=False,
    )

    print(f"\n  Fichier fusionné : {HORAIRE_FUSIONNE}")

    stats = {
        "tr_absent": False,
        "nb_consolide": len(df_consolide),
        "nb_tr": len(df_tr),
        "nb_tr_garde": len(df_tr_filtre),
        "nb_total": len(df_fusion),
        "nb_jours": df_fusion["date"].nunique(),
        "nb_jours_24h": len(jours_24h),
    }

    return {"df": df_fusion, "stats": stats}


# ============================================================
# 4. RAPPORT UNIQUE
# ============================================================

def ecrire_rapport_unifie(stats_prep, stats_fusion):
    """
    Écrit un rapport unique consolidant les informations des
    deux étapes.
    """

    RAPPORT_UNIQUE.parent.mkdir(parents=True, exist_ok=True)

    with open(RAPPORT_UNIQUE, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("RAPPORT DE PRÉPARATION DES DONNÉES éCO2mix\n")
        f.write("=" * 70 + "\n\n")

        # ----------------------------------------------------
        # Principe fondamental
        # ----------------------------------------------------

        f.write("PRINCIPE FONDAMENTAL\n")
        f.write("-" * 70 + "\n")
        f.write(
            "La consommation horaire correspond EXACTEMENT à "
            "l'observation RTE située à HH:00.\n"
        )
        f.write("Aucune moyenne entre HH:00 et HH:30 n'est calculée.\n")
        f.write("Aucune interpolation n'est réalisée.\n")
        f.write("Aucune valeur artificielle n'est créée.\n\n")

        # ----------------------------------------------------
        # Étape 1 : Préparation du consolidé
        # ----------------------------------------------------

        f.write("=" * 70 + "\n")
        f.write("ÉTAPE 1 : PRÉPARATION DES DONNÉES CONSOLIDÉES\n")
        f.write("=" * 70 + "\n\n")

        f.write("Entrée\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {RAW_CONSOLIDE.name}\n\n")

        f.write("Statistiques\n")
        f.write("-" * 70 + "\n")
        f.write(f"Observations initiales          : {stats_prep['nb_total']:,}\n")
        f.write(f"Observations à HH:00 conservées : {stats_prep['nb_horaire']:,}\n")
        f.write(f"Observations à HH:30 écartées   : {stats_prep['nb_demi_heures']:,}\n")
        f.write(f"Doublons physiques supprimés    : {stats_prep['nb_doublons']:,}\n")
        f.write(f"Consommations manquantes à HH:00: {stats_prep['nb_missing']:,}\n")
        f.write(f"Consommations négatives         : {stats_prep['nb_negatives']:,}\n")
        f.write(f"Journées complètes (24h)        : {stats_prep['nb_dates_completes']:,}\n")
        f.write(f"Journées incomplètes            : {stats_prep['nb_dates_incompletes']:,}\n\n")

        f.write("Sortie\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {HORAIRE_CONSOLIDE.name}\n\n")

        # ----------------------------------------------------
        # Étape 2 : Fusion
        # ----------------------------------------------------

        f.write("=" * 70 + "\n")
        f.write("ÉTAPE 2 : FUSION AVEC LES DONNÉES TEMPS RÉEL\n")
        f.write("=" * 70 + "\n\n")

        if stats_fusion["tr_absent"]:

            f.write(
                "ATTENTION : fichier temps réel absent.\n"
                "La fusion est ignorée.\n"
                "Le fichier consolidé a été copié tel quel.\n\n"
            )

        else:

            f.write("Entrée\n")
            f.write("-" * 70 + "\n")
            f.write(f"  {RAW_TEMPS_REEL.name}\n\n")

            f.write("Résultat de la fusion\n")
            f.write("-" * 70 + "\n")
            f.write(f"Lignes consolidées              : {stats_fusion['nb_consolide']:,}\n")
            f.write(f"Lignes temps réel               : {stats_fusion['nb_tr']:,}\n")
            f.write(f"Lignes temps réel conservées    : {stats_fusion['nb_tr_garde']:,}\n")
            f.write(f"Total après fusion              : {stats_fusion['nb_total']:,}\n")
            f.write(f"Nombre de jours                 : {stats_fusion['nb_jours']:,}\n")
            f.write(f"Jours avec 24 heures            : {stats_fusion['nb_jours_24h']:,}\n\n")

            f.write("Stratégie\n")
            f.write("-" * 70 + "\n")
            f.write(
                "En cas de conflit sur un timestamp UTC, les données\n"
                "consolidées sont prioritaires car plus fiables.\n"
                "Les données temps réel ne comblent que les jours\n"
                "absents du consolidé.\n\n")

        f.write("Sortie\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {HORAIRE_FUSIONNE.name}\n\n")

        # ----------------------------------------------------
        # Période finale
        # ----------------------------------------------------

        f.write("=" * 70 + "\n")
        f.write("PÉRIODE FINALE\n")
        f.write("=" * 70 + "\n\n")

        if not stats_fusion["tr_absent"]:
            f.write(f"Nombre de lignes totales : {stats_fusion['nb_total']:,}\n")
            f.write(f"Nombre de jours          : {stats_fusion['nb_jours']:,}\n\n")

        f.write("=" * 70 + "\n")
        f.write("FIN DU RAPPORT\n")
        f.write("=" * 70 + "\n")


# ============================================================
# 5. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PRÉPARATION DES DONNÉES éCO2mix")
    print("Projet de prévision - France métropolitaine")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Étape 1 : Préparation du consolidé
    # --------------------------------------------------------

    resultat_prep = preparer_consolide()
    df_consolide = resultat_prep["df"]
    stats_prep = resultat_prep["stats"]

    # --------------------------------------------------------
    # Étape 2 : Fusion avec le temps réel
    # --------------------------------------------------------

    resultat_fusion = fusionner_temps_reel(df_consolide)
    df_fusion = resultat_fusion["df"]
    stats_fusion = resultat_fusion["stats"]

    # --------------------------------------------------------
    # Rapport unique
    # --------------------------------------------------------

    ecrire_rapport_unifie(stats_prep, stats_fusion)

    print(f"\nRapport unique : {RAPPORT_UNIQUE}")

    # --------------------------------------------------------
    # Résumé final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PRÉPARATION TERMINÉE")
    print("=" * 70)
    print()
    print(f"Fichier préparé (consolidé) : {HORAIRE_CONSOLIDE}")
    print(f"Fichier fusionné (final)    : {HORAIRE_FUSIONNE}")
    print(f"Rapport                     : {RAPPORT_UNIQUE}")
    print()
    print(
        f"Période finale : "
        f"{df_fusion['date_heure_utc'].min()} -> "
        f"{df_fusion['date_heure_utc'].max()}"
    )
    print(f"Nombre de lignes : {len(df_fusion):,}")
    print()


if __name__ == "__main__":
    main()
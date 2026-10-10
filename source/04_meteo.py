# -*- coding: utf-8 -*-

"""
04_meteo.py

Préparation de la météo nationale à partir des données
Météo-France "Données climatologiques de base".

Ce script effectue deux étapes :

    ÉTAPE 1 : Réduction des fichiers Météo-France (optionnelle)
        - Allège les fichiers bruts (2.6 Go) en ne conservant
          que les données à partir du 1er janvier 2010.
        - Filtre les stations avec un remplissage suffisant.
        - Sauvegarde dans data/processed/meteofrance_reduit/.

    ÉTAPE 2 : Agrégation nationale
        - Filtre les codes qualité (garder 0, 1, 9).
        - Agrège les stations par département.
        - Agrège les départements par population (INSEE).
        - Crée les indicateurs dérivés (HDD, CDD, tendances,
          moyennes glissantes, cumuls).

Entrée :
    data/raw/meteofrance/
        Q_{dept}_previous-1950-2024_RR-T-Vent.csv
        Q_{dept}_latest-2025-2026_RR-T-Vent.csv

Sortie :
    data/processed/meteofrance_reduit/Q_{dept}_reduit.csv
    data/processed/meteo_france_nationale_journaliere.csv
    data/processed/rapport_meteo.txt

IMPORTANT :
    - Les valeurs sont déjà en °C et en m/s dans les fichiers
      Météo-France téléchargés (pas de division par 10).
    - Les codes qualité sont filtrés : on ne garde que 0, 1, 9.
    - La pondération nationale est faite par population (INSEE).
    - Toutes les opérations sont reproductibles.

Note :
    Si le dossier data/processed/meteofrance_reduit/ existe déjà,
    l'étape 1 est sautée. Pour forcer la réduction, supprime le
    dossier avant de relancer le script.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Entrées
RAW_METEOFRANCE_DIR = BASE_DIR / "data" / "raw" / "meteofrance"

# Sorties
REDUIT_DIR = BASE_DIR / "data" / "processed" / "meteofrance_reduit"
METEO_NATIONALE = BASE_DIR / "data" / "processed" / "meteo_france_nationale_journaliere.csv"
RAPPORT = BASE_DIR / "data" / "processed" / "rapport_meteo.txt"


# ------------------------------------------------------------
# Paramètres étape 1 : réduction
# ------------------------------------------------------------

DATE_MIN = 20100101  # 1er janvier 2010
TAUX_REMPLISSAGE_MIN = 0.80  # 80 %

COLONNES_IDENTIFICATION = [
    "NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJ",
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
# Paramètres étape 2 : agrégation
# ------------------------------------------------------------

DEPARTEMENTS_RETENUS = [
    "06", "13", "31", "33", "34", "35", "38", "44",
    "54", "59", "62", "63", "64", "67", "69",
    "75", "77", "78", "91", "92", "94", "95",
]

PONDERATIONS_POPULATION = {
    "75": 2_133_111,
    "77": 1_421_197,
    "78": 1_449_723,
    "91": 1_313_994,
    "92": 1_624_054,
    "94": 1_407_124,
    "95": 1_249_674,
    "69": 1_883_268,
    "38": 1_271_166,
    "63": 662_152,
    "59": 2_607_746,
    "62": 1_468_018,
    "13": 2_043_943,
    "06": 1_094_579,
    "31": 1_400_039,
    "34": 1_175_623,
    "33": 1_623_749,
    "64": 682_621,
    "44": 1_429_272,
    "35": 1_079_498,
    "67": 1_148_073,
    "54": 733_760,
}

CODES_QUALITE_VALIDES = {0, 1, 9}

T_REF_CHAUFFAGE = 18.0
T_REF_CLIMATISATION = 22.0

SEUIL_PLUIE_MM = 1.0
SEUIL_FORTE_PLUIE_MM = 10.0
SEUIL_GEL_C = 0.0
SEUIL_FORT_GEL_C = -5.0
SEUIL_VENT_FORT_MS = 15.0

VARIABLES_MESURES = [
    ("RR", "QRR"),
    ("TN", "QTN"),
    ("TX", "QTX"),
    ("TM", "QTM"),
    ("TAMPLI", "QTAMPLI"),
    ("TNSOL", "QTNSOL"),
    ("DG", "QDG"),
    ("FFM", "QFFM"),
    ("FXY", "QFXY"),
    ("DRR", "QDRR"),
]


# ============================================================
# 2. ÉTAPE 1 : RÉDUCTION DES FICHIERS MÉTÉO-FRANCE
# ============================================================

def lire_fichier_mf(chemin):
    """
    Lit un fichier Météo-France en ne gardant que les colonnes
    utiles.
    """

    print(f"\n  Lecture : {chemin.name}")

    df = pd.read_csv(
        chemin,
        sep=";",
        encoding="latin-1",
        low_memory=False,
    )

    print(f"    Lignes brutes : {len(df):,}")
    print(f"    Colonnes brutes : {len(df.columns)}")

    # Vérification des colonnes
    colonnes_presentes = [
        c for c in COLONNES_A_GARDER
        if c in df.columns
    ]

    colonnes_manquantes = [
        c for c in COLONNES_A_GARDER
        if c not in df.columns
    ]

    if colonnes_manquantes:
        print(f"    ATTENTION : colonnes manquantes : {colonnes_manquantes}")

    df = df[colonnes_presentes].copy()

    # Conversion de la date
    df["AAAAMMJJ"] = pd.to_numeric(df["AAAAMMJJ"], errors="coerce")
    df = df.dropna(subset=["AAAAMMJJ"]).copy()
    df["AAAAMMJJ"] = df["AAAAMMJJ"].astype(int)

    return df


def filtrer_apres_2010(df):
    """Ne conserve que les lignes avec AAAAMMJJ >= DATE_MIN."""

    masque = df["AAAAMMJJ"] >= DATE_MIN
    df_filtre = df.loc[masque].copy()

    print(f"    Lignes après filtre {DATE_MIN} : {len(df_filtre):,}")

    return df_filtre


def supprimer_stations_incompletes(df):
    """
    Supprime les stations dont le taux de remplissage de TM
    est inférieur à TAUX_REMPLISSAGE_MIN.
    """

    remplissage = (
        df
        .groupby("NUM_POSTE")["TM"]
        .apply(lambda x: x.notna().mean())
    )

    stations_ok = remplissage[
        remplissage >= TAUX_REMPLISSAGE_MIN
    ].index

    nb_avant = df["NUM_POSTE"].nunique()
    nb_apres = len(stations_ok)

    print(f"    Stations avant filtre : {nb_avant}")
    print(
        f"    Stations après filtre : {nb_apres} "
        f"(seuil {TAUX_REMPLISSAGE_MIN:.0%} sur TM)"
    )

    df_filtre = df[df["NUM_POSTE"].isin(stations_ok)].copy()

    return df_filtre


def traiter_fichier_reduction(chemin):
    """
    Pipeline de réduction sur un fichier.
    """

    df = lire_fichier_mf(chemin)

    df = filtrer_apres_2010(df)

    if len(df) == 0:
        print("    Aucune donnée après 2010. Fichier ignoré.")
        return None

    df = supprimer_stations_incompletes(df)

    if len(df) == 0:
        print("    Aucune station valide. Fichier ignoré.")
        return None

    # Extraction du département
    nom = chemin.stem
    parties = nom.split("_")

    if len(parties) < 3:
        print(f"    Nom de fichier inattendu : {nom}")
        return None

    dept = parties[1]

    chemin_sortie = REDUIT_DIR / f"Q_{dept}_reduit.csv"

    # Concaténation si le fichier existe déjà
    if chemin_sortie.exists():

        print(f"    Fichier existant : {chemin_sortie.name}")

        df_existant = pd.read_csv(
            chemin_sortie,
            sep=";",
            encoding="latin-1",
            low_memory=False,
        )

        df = pd.concat([df_existant, df], ignore_index=True)

        df = df.drop_duplicates(
            subset=["NUM_POSTE", "AAAAMMJJ"],
            keep="first",
        )

    df = df.sort_values(["NUM_POSTE", "AAAAMMJJ"]).reset_index(drop=True)

    df.to_csv(
        chemin_sortie,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    print(
        f"    Sauvegardé : {chemin_sortie.name} "
        f"({len(df):,} lignes, {df['NUM_POSTE'].nunique()} stations)"
    )

    return chemin_sortie


def etape1_reduire():
    """
    Étape 1 : réduire les fichiers Météo-France.

    Returns
    -------
    bool
        True si l'étape a été effectuée, False si elle a été sautée.
    """

    print("=" * 70)
    print("ÉTAPE 1 : RÉDUCTION DES FICHIERS MÉTÉO-FRANCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Vérification : les fichiers réduits existent-ils déjà ?
    # --------------------------------------------------------

    if REDUIT_DIR.exists() and any(REDUIT_DIR.glob("*.csv")):

        nb_fichiers = len(list(REDUIT_DIR.glob("*.csv")))

        print(
            f"\n  Le dossier {REDUIT_DIR.name}/ existe déjà "
            f"({nb_fichiers} fichiers)."
        )
        print("  Étape 1 sautée.")
        print(
            "  Pour forcer la réduction, supprime le dossier "
            f"{REDUIT_DIR}"
        )
        print("  et relance le script.")

        return False

    # --------------------------------------------------------
    # Vérification du dossier d'entrée
    # --------------------------------------------------------

    if not RAW_METEOFRANCE_DIR.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {RAW_METEOFRANCE_DIR}\n"
            f"Vérifie que les fichiers Météo-France sont bien "
            f"placés dans {RAW_METEOFRANCE_DIR}."
        )

    fichiers = sorted(RAW_METEOFRANCE_DIR.glob("*.csv"))

    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans {RAW_METEOFRANCE_DIR}"
        )

    print(f"\n  Dossier d'entrée : {RAW_METEOFRANCE_DIR}")
    print(f"  Dossier de sortie : {REDUIT_DIR}")
    print(f"  Date minimale : {DATE_MIN}")
    print(f"  Seuil de remplissage : {TAUX_REMPLISSAGE_MIN:.0%}")
    print(f"\n  Nombre de fichiers à traiter : {len(fichiers)}")

    # --------------------------------------------------------
    # Traitement
    # --------------------------------------------------------

    REDUIT_DIR.mkdir(parents=True, exist_ok=True)

    fichiers_traites = []

    for chemin in fichiers:

        try:
            resultat = traiter_fichier_reduction(chemin)
            if resultat is not None:
                fichiers_traites.append(resultat)

        except Exception as e:
            print(f"\n  ERREUR sur {chemin.name} : {e}")

    print(f"\n  Fichiers traités : {len(fichiers_traites)} / {len(fichiers)}")

    # Taille totale
    taille_totale = sum(
        f.stat().st_size for f in REDUIT_DIR.glob("*.csv")
    )

    print(f"  Taille du dossier réduit : {taille_totale / 1024**2:.1f} Mo")

    return True


# ============================================================
# 3. ÉTAPE 2 : AGRÉGATION NATIONALE
# ============================================================

def nettoyer_valeur_par_qualite(df, var, qual):
    """Remplace par NaN les valeurs avec code qualité invalide."""

    if var not in df.columns or qual not in df.columns:
        return df

    masque_invalide = ~df[qual].isin(CODES_QUALITE_VALIDES)
    df.loc[masque_invalide, var] = np.nan

    return df


def lire_fichier_departemental(chemin):
    """Lit un fichier départemental réduit et nettoie les codes qualité."""

    df = pd.read_csv(
        chemin,
        sep=";",
        encoding="utf-8",
        low_memory=False,
    )

    for var, qual in VARIABLES_MESURES:
        df = nettoyer_valeur_par_qualite(df, var, qual)

    colonnes_qualite = [
        qual for _, qual in VARIABLES_MESURES
        if qual in df.columns
    ]

    df = df.drop(columns=colonnes_qualite)

    return df


def agreger_departement(df):
    """Agrège les stations d'un département en une série journalière."""

    agg_dict = {
        "RR": ["mean"],
        "TN": ["mean", "min"],
        "TX": ["mean", "max"],
        "TM": ["mean"],
        "TAMPLI": ["mean"],
        "TNSOL": ["mean"],
        "DG": ["mean"],
        "FFM": ["mean"],
        "FXY": ["mean", "max"],
        "DRR": ["mean"],
    }

    groupe = df.groupby("AAAAMMJJ").agg(agg_dict)

    groupe.columns = [
        f"{col}_{stat}" for col, stat in groupe.columns
    ]

    groupe = groupe.reset_index()

    renommage = {
        "RR_mean": "RR",
        "TN_mean": "TN",
        "TN_min": "TN_min",
        "TX_mean": "TX",
        "TX_max": "TX_max",
        "TM_mean": "TM",
        "TAMPLI_mean": "TAMPLI",
        "TNSOL_mean": "TNSOL",
        "DG_mean": "DG",
        "FFM_mean": "FFM",
        "FXY_mean": "FXY",
        "FXY_max": "FXY_max",
        "DRR_mean": "DRR",
    }

    groupe = groupe.rename(columns=renommage)

    return groupe


def agreger_national(df_national, ponderations):
    """Agrège les séries départementales en série nationale pondérée."""

    colonnes_a_ponderer = [
        "RR", "TN", "TX", "TM", "TAMPLI",
        "TNSOL", "DG", "FFM", "FXY", "DRR",
        "TN_min", "TX_max", "FXY_max",
    ]

    df_national["poids"] = df_national["departement"].map(ponderations)

    def moyenne_ponderee(group, colonnes):
        resultat = {}
        poids = group["poids"].values

        for col in colonnes:
            if col in group.columns:
                valeurs = group[col].values
                masque = ~pd.isna(valeurs)

                if masque.sum() == 0:
                    resultat[col] = np.nan
                else:
                    resultat[col] = np.average(
                        valeurs[masque],
                        weights=poids[masque]
                    )

        return pd.Series(resultat)

    resultats = []

    for jour, groupe in df_national.groupby("AAAAMMJJ"):

        serie = moyenne_ponderee(groupe, colonnes_a_ponderer)
        serie["AAAAMMJJ"] = jour
        resultats.append(serie)

    df_final = pd.DataFrame(resultats)

    return df_final


def creer_indicateurs(meteo):
    """Crée les indicateurs dérivés à partir des agrégats nationaux."""

    meteo = meteo.copy()

    # Conversion de la date
    meteo["AAAAMMJJ"] = pd.to_numeric(
        meteo["AAAAMMJJ"], errors="coerce"
    ).astype("Int64")

    meteo["date"] = pd.to_datetime(
        meteo["AAAAMMJJ"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    meteo = meteo.dropna(subset=["date"]).copy()
    meteo = meteo.sort_values("date").reset_index(drop=True)

    # HDD
    meteo["HDD_moyen"] = np.maximum(0, T_REF_CHAUFFAGE - meteo["TM"])
    meteo["HDD_min"] = np.maximum(0, T_REF_CHAUFFAGE - meteo["TN"])
    meteo["HDD_max"] = np.maximum(0, T_REF_CHAUFFAGE - meteo["TX"])

    # CDD
    meteo["CDD_moyen"] = np.maximum(0, meteo["TM"] - T_REF_CLIMATISATION)
    meteo["CDD_max"] = np.maximum(0, meteo["TX"] - T_REF_CLIMATISATION)

    # Tendances
    meteo["delta_TM_J1"] = meteo["TM"].diff(1)
    meteo["delta_TM_J2"] = meteo["TM"].diff(2)
    meteo["delta_HDD_J1"] = meteo["HDD_moyen"].diff(1)
    meteo["delta_HDD_J2"] = meteo["HDD_moyen"].diff(2)

    # Moyennes glissantes
    meteo["TM_3j"] = meteo["TM"].rolling(3, min_periods=1).mean()
    meteo["TM_7j"] = meteo["TM"].rolling(7, min_periods=1).mean()
    meteo["HDD_3j"] = meteo["HDD_moyen"].rolling(3, min_periods=1).mean()
    meteo["HDD_7j"] = meteo["HDD_moyen"].rolling(7, min_periods=1).mean()

    # Précipitations cumulées
    meteo["RR_3j"] = meteo["RR"].rolling(3, min_periods=1).sum()
    meteo["RR_7j"] = meteo["RR"].rolling(7, min_periods=1).sum()

    # Indicateurs binaires
    meteo["a_plu"] = (meteo["RR"] > SEUIL_PLUIE_MM).astype(int)
    meteo["forte_pluie"] = (meteo["RR"] > SEUIL_FORTE_PLUIE_MM).astype(int)
    meteo["gel"] = (meteo["TN"] <= SEUIL_GEL_C).astype(int)
    meteo["fort_gel"] = (meteo["TN"] <= SEUIL_FORT_GEL_C).astype(int)
    meteo["vent_fort"] = (meteo["FXY"] > SEUIL_VENT_FORT_MS).astype(int)

    # Renommage lisible
    meteo = meteo.rename(columns={
        "TM": "temperature_moyenne",
        "TN": "temperature_min",
        "TX": "temperature_max",
        "RR": "precipitations_totales_mm",
        "TAMPLI": "amplitude_thermique",
        "TNSOL": "temperature_min_sol",
        "DG": "duree_gel_min",
        "FFM": "vent_moyen_ms",
        "FXY": "vent_max_ms",
        "DRR": "duree_precipitations_min",
    })

    # Réorganisation
    colonnes_finales = [
        "AAAAMMJJ", "date",

        "temperature_moyenne", "temperature_min", "temperature_max",
        "temperature_min_sol", "amplitude_thermique",

        "HDD_moyen", "HDD_min", "HDD_max", "CDD_moyen", "CDD_max",

        "delta_TM_J1", "delta_TM_J2", "delta_HDD_J1", "delta_HDD_J2",

        "TM_3j", "TM_7j", "HDD_3j", "HDD_7j",

        "precipitations_totales_mm", "RR_3j", "RR_7j",
        "duree_precipitations_min", "a_plu", "forte_pluie",

        "duree_gel_min", "gel", "fort_gel",

        "vent_moyen_ms", "vent_max_ms", "vent_fort",
    ]

    colonnes_presentes = [c for c in colonnes_finales if c in meteo.columns]
    meteo = meteo[colonnes_presentes].copy()

    return meteo


def etape2_agreger():
    """
    Étape 2 : agréger la météo nationale.

    Returns
    -------
    dict
        Dictionnaire de statistiques pour le rapport.
    """

    print("\n" + "=" * 70)
    print("ÉTAPE 2 : AGRÉGATION NATIONALE DE LA MÉTÉO")
    print("=" * 70)

    # Vérification
    if not REDUIT_DIR.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {REDUIT_DIR}\n"
            "L'étape 1 doit être effectuée en premier."
        )

    fichiers = sorted(REDUIT_DIR.glob("Q_*_reduit.csv"))

    print(f"\n  Fichiers trouvés : {len(fichiers)}")

    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans {REDUIT_DIR}"
        )

    # Traitement par département
    frames = []

    for chemin in fichiers:

        nom = chemin.stem
        parties = nom.split("_")

        if len(parties) < 2:
            continue

        dept = parties[1].zfill(2)

        if dept not in DEPARTEMENTS_RETENUS:
            print(f"  [SKIP] Département {dept} non retenu")
            continue

        print(f"\n  Traitement département {dept}...")

        df = lire_fichier_departemental(chemin)

        print(f"    Lignes lues : {len(df):,}")

        df_dept = agreger_departement(df)
        df_dept["departement"] = dept

        print(f"    Jours agrégés : {len(df_dept):,}")

        frames.append(df_dept)

    if not frames:
        raise ValueError("Aucun département traité.")

    # Concaténation
    print("\n  Concaténation des départements...")

    df_national = pd.concat(frames, ignore_index=True)

    print(f"    Lignes totales : {len(df_national):,}")
    print(f"    Départements : {df_national['departement'].nunique()}")
    print(f"    Jours uniques : {df_national['AAAAMMJJ'].nunique():,}")

    # Agrégation nationale
    print("\n  Agrégation nationale pondérée...")

    df_final = agreger_national(df_national, PONDERATIONS_POPULATION)

    print(f"    Lignes finales : {len(df_final):,}")

    # Indicateurs dérivés
    print("\n  Création des indicateurs dérivés...")

    meteo = creer_indicateurs(df_final)

    print(f"    Colonnes finales : {len(meteo.columns)}")

    # Vérifications
    print("\n  Vérifications finales :")
    print(f"    Première date : {meteo['date'].min().date()}")
    print(f"    Dernière date : {meteo['date'].max().date()}")
    print(f"    Nombre de jours : {len(meteo):,}")

    # Sauvegarde
    meteo.to_csv(
        METEO_NATIONALE,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    print(f"\n  Fichier sauvegardé : {METEO_NATIONALE}")

    # Statistiques pour le rapport
    valeurs_manquantes = meteo.isna().sum()
    valeurs_manquantes = valeurs_manquantes[valeurs_manquantes > 0]

    stats = {
        "nb_fichiers": len(fichiers),
        "nb_departements": df_national["departement"].nunique(),
        "nb_jours_uniques": df_national["AAAAMMJJ"].nunique(),
        "nb_lignes_finales": len(meteo),
        "date_min": meteo["date"].min(),
        "date_max": meteo["date"].max(),
        "nb_colonnes": len(meteo.columns),
        "valeurs_manquantes": valeurs_manquantes.to_dict(),
    }

    return stats


# ============================================================
# 4. RAPPORT UNIQUE
# ============================================================

def ecrire_rapport(etape1_effectuee, stats_etape1, stats_etape2):
    """
    Écrit un rapport unique pour tout le pipeline météo.
    """

    RAPPORT.parent.mkdir(parents=True, exist_ok=True)

    with open(RAPPORT, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("RAPPORT DE PRÉPARATION DE LA MÉTÉO NATIONALE\n")
        f.write("=" * 70 + "\n\n")

        # ----------------------------------------------------
        # Étape 1
        # ----------------------------------------------------

        f.write("=" * 70 + "\n")
        f.write("ÉTAPE 1 : RÉDUCTION DES FICHIERS MÉTÉO-FRANCE\n")
        f.write("=" * 70 + "\n\n")

        if etape1_effectuee:

            f.write(f"Dossier d'entrée : {RAW_METEOFRANCE_DIR.name}/\n")
            f.write(f"Dossier de sortie : {REDUIT_DIR.name}/\n")
            f.write(f"Date minimale : {DATE_MIN}\n")
            f.write(f"Seuil de remplissage : {TAUX_REMPLISSAGE_MIN:.0%}\n\n")

            if stats_etape1:
                f.write(f"Fichiers traités : {stats_etape1['nb_traites']}\n")
                f.write(f"Fichiers totaux : {stats_etape1['nb_total']}\n")
                f.write(
                    f"Taille du dossier réduit : "
                    f"{stats_etape1['taille_mo']:.1f} Mo\n\n"
                )
            else:
                f.write("Aucun fichier traité.\n\n")

        else:

            f.write(
                "ÉTAPE SAUTÉE : le dossier meteorfrance_reduit/ "
                "existe déjà.\n\n"
            )
            f.write(
                "Pour forcer la réduction, supprime le dossier "
                f"{REDUIT_DIR.name}/\n"
                "et relance le script.\n\n"
            )

        # ----------------------------------------------------
        # Étape 2
        # ----------------------------------------------------

        f.write("=" * 70 + "\n")
        f.write("ÉTAPE 2 : AGRÉGATION NATIONALE\n")
        f.write("=" * 70 + "\n\n")

        f.write("Sources\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Dossier : {REDUIT_DIR.name}/\n")
        f.write(f"  Fichiers : {stats_etape2['nb_fichiers']}\n")
        f.write(f"  Départements retenus : {stats_etape2['nb_departements']}\n\n")

        f.write("Agrégation\n")
        f.write("-" * 70 + "\n")
        f.write(
            "  1. Filtrage des codes qualité (garder 0, 1, 9)\n"
        )
        f.write(
            "  2. Agrégation intra-département (moyenne des stations)\n"
        )
        f.write(
            "  3. Agrégation nationale pondérée par population (INSEE)\n\n"
        )

        f.write("Résultat\n")
        f.write("-" * 70 + "\n")
        f.write(f"Jours uniques : {stats_etape2['nb_jours_uniques']:,}\n")
        f.write(f"Lignes finales : {stats_etape2['nb_lignes_finales']:,}\n")
        f.write(f"Colonnes finales : {stats_etape2['nb_colonnes']}\n")
        f.write(f"Première date : {stats_etape2['date_min']}\n")
        f.write(f"Dernière date : {stats_etape2['date_max']}\n\n")

        f.write("Indicateurs créés\n")
        f.write("-" * 70 + "\n")
        f.write("  - Degrés-jours de chauffage (HDD)\n")
        f.write("  - Degrés-jours de climatisation (CDD)\n")
        f.write("  - Tendances (delta_TM_J1, delta_TM_J2, etc.)\n")
        f.write("  - Moyennes glissantes (TM_3j, TM_7j, HDD_3j, HDD_7j)\n")
        f.write("  - Précipitations cumulées (RR_3j, RR_7j)\n")
        f.write("  - Indicateurs binaires (a_plu, gel, vent_fort, etc.)\n\n")

        if stats_etape2["valeurs_manquantes"]:
            f.write("Valeurs manquantes par colonne\n")
            f.write("-" * 70 + "\n")
            for col, n in stats_etape2["valeurs_manquantes"].items():
                taux = n / stats_etape2["nb_lignes_finales"] * 100
                f.write(f"  {col} : {n} ({taux:.1f} %)\n")
            f.write("\n")

        f.write("Sortie\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {METEO_NATIONALE.name}\n\n")

        f.write("=" * 70 + "\n")
        f.write("FIN DU RAPPORT\n")
        f.write("=" * 70 + "\n")


# ============================================================
# 5. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PRÉPARATION DE LA MÉTÉO NATIONALE")
    print("Projet de prévision - France métropolitaine")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Étape 1 : Réduction
    # --------------------------------------------------------

    etape1_effectuee = etape1_reduire()

    stats_etape1 = None

    if etape1_effectuee:

        fichiers_traites = list(REDUIT_DIR.glob("Q_*_reduit.csv"))
        taille_mo = sum(
            f.stat().st_size for f in fichiers_traites
        ) / 1024**2

        stats_etape1 = {
            "nb_traites": len(fichiers_traites),
            "nb_total": len(list(RAW_METEOFRANCE_DIR.glob("*.csv"))),
            "taille_mo": taille_mo,
        }

    # --------------------------------------------------------
    # Étape 2 : Agrégation
    # --------------------------------------------------------

    stats_etape2 = etape2_agreger()

    # --------------------------------------------------------
    # Rapport unique
    # --------------------------------------------------------

    ecrire_rapport(etape1_effectuee, stats_etape1, stats_etape2)

    print(f"\nRapport unique : {RAPPORT}")

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PRÉPARATION MÉTÉO TERMINÉE")
    print("=" * 70)
    print()
    print(f"Fichier national : {METEO_NATIONALE}")
    print(f"Rapport          : {RAPPORT}")
    print()
    print(f"Période : {stats_etape2['date_min']} -> {stats_etape2['date_max']}")
    print(f"Jours   : {stats_etape2['nb_lignes_finales']:,}")
    print()


if __name__ == "__main__":
    main()
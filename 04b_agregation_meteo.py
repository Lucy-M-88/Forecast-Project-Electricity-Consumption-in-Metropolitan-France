# -*- coding: utf-8 -*-

"""
06b_agregation_meteo_mf.py

Objectif
--------
Agréger les fichiers Météo-France réduits (par département)
en une série nationale journalière enrichie.

Entrée :
    data/processed/meteofrance_reduit/Q_{dept}_reduit.csv

Sortie :
    data/processed/meteo_france_nationale_journaliere.csv

Étapes
------
1. Lecture des fichiers départementaux.
2. Filtrage des codes qualité (garder 0, 1, 9).
3. Nettoyage des valeurs aberrantes.
4. Agrégation intra-département (une ligne par jour et par dept).
5. Agrégation nationale pondérée par la population.
6. Création des indicateurs dérivés (HDD, CDD, tendances,
   moyennes glissantes, cumuls, indicateurs binaires).
7. Sauvegarde.

IMPORTANT
---------
- Les valeurs sont déjà en °C et en m/s dans les fichiers
  Météo-France téléchargés (pas de division par 10).
- Les codes qualité sont filtrés : on ne garde que 0, 1, 9.
- La pondération nationale est faite par population (INSEE).
- Toutes les opérations sont reproductibles.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "data" / "processed" / "meteofrance_reduit"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "meteo_france_nationale_journaliere.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Départements retenus (21 après exclusion du 93)
# ------------------------------------------------------------

DEPARTEMENTS_RETENUS = [
    "06", "13", "31", "33", "34", "35", "38", "44",
    "54", "59", "62", "63", "64", "67", "69",
    "75", "77", "78", "91", "92", "94", "95",
]

# Note : le 93 est exclu car aucune station ne passe le seuil
# de remplissage sur la période 2010-2024.


# ------------------------------------------------------------
# Pondérations par population (INSEE 2021)
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Codes qualité à conserver
# ------------------------------------------------------------

CODES_QUALITE_VALIDES = {0, 1, 9}


# ------------------------------------------------------------
# Constantes météo
# ------------------------------------------------------------

T_REF_CHAUFFAGE = 18.0
T_REF_CLIMATISATION = 22.0

SEUIL_PLUIE_MM = 1.0
SEUIL_FORTE_PLUIE_MM = 10.0
SEUIL_GEL_C = 0.0
SEUIL_FORT_GEL_C = -5.0
SEUIL_VENT_FORT_MS = 15.0


# ------------------------------------------------------------
# Colonnes météo et leurs codes qualité
# ------------------------------------------------------------

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
# 2. FONCTIONS UTILITAIRES
# ============================================================

def nettoyer_valeur_par_qualite(df, var, qual):
    """
    Remplace par NaN les valeurs dont le code qualité n'est
    pas dans CODES_QUALITE_VALIDES.
    """

    if var not in df.columns or qual not in df.columns:
        return df

    masque_invalide = ~df[qual].isin(CODES_QUALITE_VALIDES)

    df.loc[masque_invalide, var] = np.nan

    return df


def lire_fichier_departemental(chemin):
    """
    Lit un fichier départemental réduit et nettoie les valeurs
    selon les codes qualité.
    """

    df = pd.read_csv(
        chemin,
        sep=";",
        encoding="utf-8",
        low_memory=False,
    )

    # --------------------------------------------------------
    # Nettoyage des codes qualité
    # --------------------------------------------------------

    for var, qual in VARIABLES_MESURES:
        df = nettoyer_valeur_par_qualite(df, var, qual)

    # --------------------------------------------------------
    # Suppression des colonnes de qualité (plus utiles)
    # --------------------------------------------------------

    colonnes_qualite = [
        qual for _, qual in VARIABLES_MESURES
        if qual in df.columns
    ]

    df = df.drop(columns=colonnes_qualite)

    return df


def agreger_departement(df):
    """
    Agrège les stations d'un département en une seule série
    journalière.
    """

    # --------------------------------------------------------
    # Agrégation principale
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Aplatir les colonnes multi-index
    # --------------------------------------------------------

    groupe.columns = [
        f"{col}_{stat}" for col, stat in groupe.columns
    ]

    groupe = groupe.reset_index()

    # --------------------------------------------------------
    # Renommer les colonnes finales
    # --------------------------------------------------------

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
    """
    Agrège les séries départementales en une série nationale
    pondérée par la population.

    Chaque jour, la valeur nationale est la moyenne pondérée
    des valeurs départementales.
    """

    # --------------------------------------------------------
    # Colonnes à agréger
    # --------------------------------------------------------

    colonnes_a_ponderer = [
        "RR", "TN", "TX", "TM", "TAMPLI",
        "TNSOL", "DG", "FFM", "FXY", "DRR",
        "TN_min", "TX_max", "FXY_max",
    ]

    # --------------------------------------------------------
    # Ajout de la pondération
    # --------------------------------------------------------

    df_national["poids"] = (
        df_national["departement"]
        .map(ponderations)
    )

    # --------------------------------------------------------
    # Agrégation pondérée
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Application par jour
    # --------------------------------------------------------

    resultats = []

    for jour, groupe in df_national.groupby("AAAAMMJJ"):

        serie = moyenne_ponderee(
            groupe,
            colonnes_a_ponderer
        )

        serie["AAAAMMJJ"] = jour
        resultats.append(serie)

    df_final = pd.DataFrame(resultats)

    return df_final


# ============================================================
# 3. CRÉATION DES INDICATEURS DÉRIVÉS
# ============================================================

def creer_indicateurs(meteo):
    """
    Crée les indicateurs dérivés à partir des agrégats
    nationaux journaliers.
    """

    meteo = meteo.copy()

    # --------------------------------------------------------
    # Conversion de la date
    # --------------------------------------------------------

    # Conversion robuste : on force en int avant de convertir en str
    meteo["AAAAMMJJ"] = pd.to_numeric(
        meteo["AAAAMMJJ"],
        errors="coerce"
    ).astype("Int64")

    meteo["date"] = pd.to_datetime(
        meteo["AAAAMMJJ"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    meteo = meteo.dropna(subset=["date"]).copy()

    meteo = meteo.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------------
    # Degrés-jours de chauffage (HDD)
    # --------------------------------------------------------

    meteo["HDD_moyen"] = np.maximum(
        0, T_REF_CHAUFFAGE - meteo["TM"]
    )

    meteo["HDD_min"] = np.maximum(
        0, T_REF_CHAUFFAGE - meteo["TN"]
    )

    meteo["HDD_max"] = np.maximum(
        0, T_REF_CHAUFFAGE - meteo["TX"]
    )

    # --------------------------------------------------------
    # Degrés-jours de climatisation (CDD)
    # --------------------------------------------------------

    meteo["CDD_moyen"] = np.maximum(
        0, meteo["TM"] - T_REF_CLIMATISATION
    )

    meteo["CDD_max"] = np.maximum(
        0, meteo["TX"] - T_REF_CLIMATISATION
    )

    # --------------------------------------------------------
    # Tendances
    # --------------------------------------------------------

    meteo["delta_TM_J1"] = meteo["TM"].diff(1)
    meteo["delta_TM_J2"] = meteo["TM"].diff(2)

    meteo["delta_HDD_J1"] = meteo["HDD_moyen"].diff(1)
    meteo["delta_HDD_J2"] = meteo["HDD_moyen"].diff(2)

    # --------------------------------------------------------
    # Moyennes glissantes
    # --------------------------------------------------------

    meteo["TM_3j"] = meteo["TM"].rolling(3, min_periods=1).mean()
    meteo["TM_7j"] = meteo["TM"].rolling(7, min_periods=1).mean()

    meteo["HDD_3j"] = (
        meteo["HDD_moyen"].rolling(3, min_periods=1).mean()
    )
    meteo["HDD_7j"] = (
        meteo["HDD_moyen"].rolling(7, min_periods=1).mean()
    )

    # --------------------------------------------------------
    # Précipitations cumulées
    # --------------------------------------------------------

    meteo["RR_3j"] = (
        meteo["RR"].rolling(3, min_periods=1).sum()
    )
    meteo["RR_7j"] = (
        meteo["RR"].rolling(7, min_periods=1).sum()
    )

    # --------------------------------------------------------
    # Indicateurs binaires
    # --------------------------------------------------------

    meteo["a_plu"] = (meteo["RR"] > SEUIL_PLUIE_MM).astype(int)
    meteo["forte_pluie"] = (
        meteo["RR"] > SEUIL_FORTE_PLUIE_MM
    ).astype(int)

    meteo["gel"] = (meteo["TN"] <= SEUIL_GEL_C).astype(int)
    meteo["fort_gel"] = (
        meteo["TN"] <= SEUIL_FORT_GEL_C
    ).astype(int)

    meteo["vent_fort"] = (
        meteo["FXY"] > SEUIL_VENT_FORT_MS
    ).astype(int)

    # --------------------------------------------------------
    # Renommage lisible
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Réorganisation
    # --------------------------------------------------------

    colonnes_finales = [
        "AAAAMMJJ",
        "date",

        # Températures
        "temperature_moyenne",
        "temperature_min",
        "temperature_max",
        "temperature_min_sol",
        "amplitude_thermique",

        # Degrés-jours
        "HDD_moyen",
        "HDD_min",
        "HDD_max",
        "CDD_moyen",
        "CDD_max",

        # Tendances
        "delta_TM_J1",
        "delta_TM_J2",
        "delta_HDD_J1",
        "delta_HDD_J2",

        # Moyennes glissantes
        "TM_3j",
        "TM_7j",
        "HDD_3j",
        "HDD_7j",

        # Précipitations
        "precipitations_totales_mm",
        "RR_3j",
        "RR_7j",
        "duree_precipitations_min",
        "a_plu",
        "forte_pluie",

        # Froid
        "duree_gel_min",
        "gel",
        "fort_gel",

        # Vent
        "vent_moyen_ms",
        "vent_max_ms",
        "vent_fort",
    ]

    colonnes_presentes = [
        c for c in colonnes_finales
        if c in meteo.columns
    ]

    meteo = meteo[colonnes_presentes].copy()

    return meteo


# ============================================================
# 4. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("AGRÉGATION MÉTÉO-FRANCE ET CRÉATION D'INDICATEURS")
    print("=" * 70)

    print(f"\nDossier d'entrée : {INPUT_DIR}")
    print(f"Fichier de sortie : {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Lecture des fichiers départementaux
    # --------------------------------------------------------

    fichiers = sorted(
        INPUT_DIR.glob("Q_*_reduit.csv")
    )

    print(f"\nFichiers trouvés : {len(fichiers)}")

    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans {INPUT_DIR}"
        )

    # --------------------------------------------------------
    # Traitement par département
    # --------------------------------------------------------

    frames = []

    for chemin in fichiers:

        # Extraire le numéro de département
        nom = chemin.stem  # ex: Q_75_reduit

        parties = nom.split("_")

        if len(parties) < 2:
            continue

        dept = parties[1].zfill(2)

        if dept not in DEPARTEMENTS_RETENUS:
            print(f"  [SKIP] Département {dept} non retenu")
            continue

        print(f"\nTraitement département {dept}...")

        df = lire_fichier_departemental(chemin)

        print(f"  Lignes lues : {len(df):,}")

        # Agrégation intra-département
        df_dept = agreger_departement(df)

        df_dept["departement"] = dept

        print(f"  Jours agrégés : {len(df_dept):,}")

        frames.append(df_dept)

    if not frames:
        raise ValueError("Aucun département traité.")

    # --------------------------------------------------------
    # Concaténation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONCATÉNATION DES DÉPARTEMENTS")
    print("=" * 70)

    df_national = pd.concat(frames, ignore_index=True)

    print(f"Lignes totales : {len(df_national):,}")
    print(f"Départements : {df_national['departement'].nunique()}")
    print(
        f"Jours uniques : "
        f"{df_national['AAAAMMJJ'].nunique():,}"
    )

    # --------------------------------------------------------
    # Agrégation nationale pondérée
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AGRÉGATION NATIONALE PONDÉRÉE")
    print("=" * 70)

    df_final = agreger_national(
        df_national,
        PONDERATIONS_POPULATION
    )

    print(f"Lignes finales : {len(df_final):,}")

    # --------------------------------------------------------
    # Indicateurs dérivés
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CRÉATION DES INDICATEURS DÉRIVÉS")
    print("=" * 70)

    meteo = creer_indicateurs(df_final)

    print(f"Colonnes finales : {len(meteo.columns)}")

    for col in meteo.columns:
        print(f"  - {col}")

    # --------------------------------------------------------
    # Vérifications finales
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VÉRIFICATIONS FINALES")
    print("=" * 70)

    print(
        f"Première date : "
        f"{meteo['date'].min().date()}"
    )

    print(
        f"Dernière date : "
        f"{meteo['date'].max().date()}"
    )

    print(
        f"Nombre de jours : {len(meteo):,}"
    )

    # --------------------------------------------------------
    # Valeurs manquantes
    # --------------------------------------------------------

    valeurs_manquantes = meteo.isna().sum()

    if valeurs_manquantes.sum() > 0:
        print("\nValeurs manquantes par colonne :")
        for col, n in valeurs_manquantes.items():
            if n > 0:
                taux = n / len(meteo) * 100
                print(f"  {col} : {n} ({taux:.1f} %)")

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    meteo.to_csv(
        OUTPUT_FILE,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nFichier sauvegardé : {OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()
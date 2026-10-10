# -*- coding: utf-8 -*-

"""
09_meteo_regionale.py

Agrège la météo départementale en météo régionale.

STRATÉGIE
---------
Pour chaque région :
    1. On essaie chaque département préféré dans l'ordre.
    2. Si aucun n'est disponible, on prend N'IMPORTE QUEL
       département disponible (fallback absolu).
Cela garantit qu'AUCUNE région n'est laissée de côté.

Lancement :
    python 09_meteo_regionale.py
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
REDUIT_DIR = BASE_DIR / "data" / "processed" / "meteofrance_reduit"
FICHIER_SORTIE = BASE_DIR / "data" / "processed" / "meteo_regionale_journaliere.csv"

DATE_MIN = 20100101

DEPT_PREFERENCE_REGION = {
    "11": ["75", "92", "93", "94", "78", "91", "95", "77"],
    "24": ["45", "37", "41", "28", "36", "18", "91", "77", "89"],
    "27": ["21", "25", "39", "71", "58", "70", "89", "90", "69", "38", "54", "67"],
    "28": ["76", "14", "27", "61", "50", "75", "78", "95", "92"],
    "32": ["59", "62", "80", "60", "02", "75", "77", "78"],
    "44": ["67", "54", "57", "88", "68", "10", "51", "52", "55", "08"],
    "52": ["44", "49", "72", "85", "53", "35", "91", "77"],
    "53": ["35", "29", "56", "22", "44", "49", "72"],
    "75": ["33", "64", "86", "87", "16", "17", "19", "24", "40", "47", "79"],
    "76": ["31", "34", "64", "09", "11", "12", "30", "32", "46", "48", "65", "66", "81", "82"],
    "84": ["69", "38", "63", "01", "03", "07", "15", "26", "42", "43", "73", "74"],
    "93": ["13", "06", "83", "84", "04", "05"],
}

NOM_REGION = {
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

CODES_QUALITE_VALIDES = {0, 1, 9}
T_REF_CHAUFFAGE = 18.0

VARIABLES_MESURES = [
    ("RR", "QRR"), ("TN", "QTN"), ("TX", "QTX"), ("TM", "QTM"),
    ("TAMPLI", "QTAMPLI"), ("TNSOL", "QTNSOL"), ("DG", "QDG"),
    ("FFM", "QFFM"), ("FXY", "QFXY"), ("DRR", "QDRR"),
]


# ============================================================
# 2. UTILITAIRES
# ============================================================

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
# 3. LECTURE FICHIER DÉPARTEMENTAL
# ============================================================

def nettoyer_qualite(df, var, qual):
    if var not in df.columns or qual not in df.columns:
        return df
    masque = ~df[qual].isin(CODES_QUALITE_VALIDES)
    df.loc[masque, var] = np.nan
    return df


def lire_fichier_departemental(chemin):

    df = pd.read_csv(chemin, sep=";", encoding="utf-8", low_memory=False)

    for var, qual in VARIABLES_MESURES:
        df = nettoyer_qualite(df, var, qual)

    cols_qual = [q for _, q in VARIABLES_MESURES if q in df.columns]
    df = df.drop(columns=cols_qual, errors="ignore")

    df = df[df["AAAAMMJJ"] >= DATE_MIN].copy()

    return df


# ============================================================
# 4. AGRÉGATION DES STATIONS
# ============================================================

def agreger_stations_departement(df):

    agg = {
        "RR": "mean",
        "TN": ["mean", "min"],
        "TX": ["mean", "max"],
        "TM": "mean",
        "TAMPLI": "mean",
        "TNSOL": "mean",
        "DG": "mean",
        "FFM": "mean",
        "FXY": ["mean", "max"],
        "DRR": "mean",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}

    g = df.groupby("AAAAMMJJ").agg(agg)
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    g = g.reset_index()

    ren = {
        "RR_mean": "RR",
        "TN_mean": "TN", "TN_min": "TN_min",
        "TX_mean": "TX", "TX_max": "TX_max",
        "TM_mean": "TM",
        "TAMPLI_mean": "TAMPLI",
        "TNSOL_mean": "TNSOL",
        "DG_mean": "DG",
        "FFM_mean": "FFM",
        "FXY_mean": "FXY", "FXY_max": "FXY_max",
        "DRR_mean": "DRR",
    }
    g = g.rename(columns=ren)

    return g


# ============================================================
# 5. CHOIX DU DÉPARTEMENT DE RÉFÉRENCE
# ============================================================

def choisir_departement(code_region, depts_dispo):
    """
    Retourne (dept, raison) où raison est :
        - "pref_0", "pref_1", ... : département préféré n°i
        - "fallback_absolu" : aucun préféré n'était dispo
        - None, None : aucun département du tout
    """

    preferences = DEPT_PREFERENCE_REGION.get(code_region, [])

    for i, dept in enumerate(preferences):
        if dept in depts_dispo:
            return dept, f"pref_{i}"

    # Fallback absolu : n'importe quel dept dispo (trié)
    if depts_dispo:
        dept_fallback = sorted(depts_dispo)[0]
        return dept_fallback, "fallback_absolu"

    return None, None


# ============================================================
# 6. INDICATEURS DÉRIVÉS
# ============================================================

def creer_indicateurs(df):

    df = df.copy()
    df = df.sort_values(["code_region", "date"]).reset_index(drop=True)

    def par_region(g):

        g = g.sort_values("date").copy()

        g["HDD_moyen"] = np.maximum(0, T_REF_CHAUFFAGE - g["TM"])
        g["HDD_min"] = np.maximum(0, T_REF_CHAUFFAGE - g["TN"])
        g["HDD_max"] = np.maximum(0, T_REF_CHAUFFAGE - g["TX"])

        g["TM_3j"] = g["TM"].rolling(3, min_periods=1).mean()
        g["TM_7j"] = g["TM"].rolling(7, min_periods=1).mean()
        g["HDD_3j"] = g["HDD_moyen"].rolling(3, min_periods=1).mean()
        g["HDD_7j"] = g["HDD_moyen"].rolling(7, min_periods=1).mean()

        return g

    df = df.groupby("code_region", group_keys=False).apply(par_region)

    df = df.rename(columns={
        "TM": "temperature_moyenne",
        "TN": "temperature_min",
        "TX": "temperature_max",
        "TAMPLI": "amplitude_thermique",
        "DG": "duree_gel_min",
    })

    df["amplitude_thermique"] = (
        df["temperature_max"] - df["temperature_min"]
    )

    df["nom_region"] = df["code_region"].map(NOM_REGION)

    cols = [
        "date", "code_region", "nom_region",
        "temperature_moyenne", "temperature_min", "temperature_max",
        "amplitude_thermique",
        "HDD_moyen", "HDD_min", "HDD_max",
        "TM_3j", "TM_7j", "HDD_3j", "HDD_7j",
        "duree_gel_min",
    ]

    df = df[[c for c in cols if c in df.columns]].copy()

    return df


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PRÉPARATION DE LA MÉTÉO RÉGIONALE")
    print("=" * 70)

    if not REDUIT_DIR.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {REDUIT_DIR}\n"
            "Lance d'abord 04_meteo.py."
        )

    # --------------------------------------------------------
    # Inventaire
    # --------------------------------------------------------

    fichiers = sorted(REDUIT_DIR.glob("Q_*_reduit.csv"))

    depts_dispo = set()
    for f in fichiers:
        parties = f.stem.split("_")
        if len(parties) >= 2:
            depts_dispo.add(normalize_code(parties[1]))

    print(f"\nDépartements météo disponibles : {len(depts_dispo)}")
    print(f"  {sorted(depts_dispo)}")

    if not depts_dispo:
        raise ValueError("Aucun département météo trouvé.")

    # --------------------------------------------------------
    # Choix d'un dept par région
    # --------------------------------------------------------

    print("\nChoix du département de référence :")

    choix = {}

    for code_region in NOM_REGION.keys():

        dept, raison = choisir_departement(code_region, depts_dispo)

        if dept is None:
            print(f"  [{code_region}] {NOM_REGION[code_region]} : AUCUN")
            continue

        if raison == "fallback_absolu":
            tag = "⚠ FALLBACK ABSOLU"
        elif raison.startswith("pref_"):
            idx = int(raison.split("_")[1])
            tag = "(représentatif)" if idx == 0 else f"(fallback n°{idx + 1})"
        else:
            tag = ""

        print(f"  [{code_region}] {NOM_REGION[code_region]:<32s} -> {dept} {tag}")

        choix[code_region] = dept

    if not choix:
        raise ValueError("Aucune région associée.")

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    print("\nChargement des météos régionales...")

    frames = []

    for code_region, dept in choix.items():

        chemin = REDUIT_DIR / f"Q_{dept}_reduit.csv"

        if not chemin.exists():
            print(f"  [{code_region}] {chemin.name} introuvable, skip")
            continue

        df = lire_fichier_departemental(chemin)
        df = agreger_stations_departement(df)

        df["code_region"] = code_region
        df["nom_region"] = NOM_REGION[code_region]
        df["dept_reference"] = dept

        frames.append(df)

        print(
            f"  [{code_region}] {NOM_REGION[code_region]:<32s} "
            f"({dept}) : {len(df):,} jours"
        )

    if not frames:
        raise ValueError("Aucune météo régionale produite.")

    df_reg = pd.concat(frames, ignore_index=True)

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df_reg["date"] = pd.to_datetime(
        df_reg["AAAAMMJJ"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df_reg = df_reg.dropna(subset=["date"]).drop(columns=["AAAAMMJJ"])

    # --------------------------------------------------------
    # Indicateurs
    # --------------------------------------------------------

    df_final = creer_indicateurs(df_reg)

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    df_final.to_csv(FICHIER_SORTIE, sep=";", index=False, encoding="utf-8")

    print()
    print("=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    print(f"Fichier : {FICHIER_SORTIE}")
    print(f"Lignes  : {len(df_final):,}")
    print(
        f"Période : {df_final['date'].min().date()} -> "
        f"{df_final['date'].max().date()}"
    )
    print(f"Régions : {df_final['code_region'].nunique()}")

    codes_attendus = set(NOM_REGION.keys())
    codes_presents = set(df_final["code_region"].unique())
    manquantes = codes_attendus - codes_presents

    if manquantes:
        print(f"\n⚠ Régions MANQUANTES :")
        for c in sorted(manquantes):
            print(f"    {c} : {NOM_REGION[c]}")
    else:
        print(f"\n✓ Les 13 régions ont une météo.")

    print()


if __name__ == "__main__":
    main()
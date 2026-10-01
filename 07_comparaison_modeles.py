# ============================================================
# 07_comparaison_modeles.py
# Comparaison finale des modèles de prévision
#
# Modèles comparés :
# 1. Random Forest sans météo
# 2. Benchmark J-7
# 3. Benchmark J-1 an
# 4. Benchmark moyenne J-7 / J-1 an
# 5. Benchmark dernier jour complet disponible à 14h
# 6. Prévision RTE / éCO2mix J-1
# 7. Random Forest + SYNOP
#
# IMPORTANT :
# - Aucun benchmark n'est utilisé pour entraîner le Random Forest.
# - La prévision RTE J-1 est uniquement un benchmark externe.
# - Les données météo utilisées par le modèle SYNOP sont limitées
#   aux informations disponibles avant 14h.
# ============================================================

from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import requests


# ============================================================
# 1. CHEMINS
# ============================================================

DATASET_BASE = Path("data/processed/dataset_prevision.csv")
DATASET_SYNOP = Path("data/processed/dataset_prevision_synop.csv")

MODELE_BASE = Path("models/modele_random_forest.joblib")
MODELE_SYNOP = Path("models/modele_random_forest_synop.joblib")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. PARAMÈTRES
# ============================================================

RTE_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/eco2mix-national-cons-def/records"
)

# Même découpage temporel que dans 04_modele.py
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15


# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

def normaliser_date(x):
    """
    Convertit une colonne de date en datetime64[ns] sans timezone
    et ramène la date à minuit.
    """
    x = pd.to_datetime(x, errors="coerce", utc=True)

    return (
        x.dt.tz_convert("Europe/Paris")
        .dt.tz_localize(None)
        .dt.normalize()
    )


def normaliser_date_simple(x):
    """
    Normalisation robuste lorsqu'une colonne peut être déjà
    timezone-naive ou timezone-aware.
    """
    x = pd.to_datetime(x, errors="coerce")

    # Si la série possède une timezone
    if hasattr(x.dt, "tz") and x.dt.tz is not None:
        x = x.dt.tz_convert("Europe/Paris").dt.tz_localize(None)

    return x.dt.normalize()


def calculer_split_dates(df):
    """
    Reproduit exactement le découpage chronologique du modèle 04.
    """

    dates = (
        pd.to_datetime(df["origine_prevision"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n = len(dates)

    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_dates = dates.iloc[:n_train]
    valid_dates = dates.iloc[n_train:n_train + n_valid]
    test_dates = dates.iloc[n_train + n_valid:]

    return train_dates, valid_dates, test_dates


def afficher_periode(nom, dates):
    if len(dates) == 0:
        print(f"{nom} : aucune date")
        return

    print(
        f"{nom} : "
        f"{dates.min().date()} -> {dates.max().date()} "
        f"({len(dates)} jours)"
    )


# ============================================================
# 4. MÉTRIQUES
# ============================================================

def calculer_metriques_horaires(y_true, y_pred):
    """
    MAE et RMSE sur les valeurs horaires.
    """

    masque = (
        pd.notna(y_true)
        & pd.notna(y_pred)
    )

    y_true = np.asarray(y_true)[masque]
    y_pred = np.asarray(y_pred)[masque]

    if len(y_true) == 0:
        return {
            "nombre_heures": 0,
            "MAE_MW": np.nan,
            "RMSE_MW": np.nan,
        }

    erreur = y_pred - y_true

    mae = np.mean(np.abs(erreur))
    rmse = np.sqrt(np.mean(erreur ** 2))

    return {
        "nombre_heures": len(y_true),
        "MAE_MW": mae,
        "RMSE_MW": rmse,
    }


def calculer_metriques_journalieres(df, colonne_prediction):
    """
    Compare le total journalier prévu au total journalier réel.
    """

    df = df.copy()

    df["erreur"] = (
        df[colonne_prediction]
        - df["consommation_cible"]
    )

    df["abs_erreur"] = df["erreur"].abs()

    journ = (
        df.groupby("date_cible")
        .agg(
            total_reel_MWh=("consommation_cible", "sum"),
            total_prevu_MWh=(colonne_prediction, "sum"),
            erreur_totale_MWh=("erreur", "sum"),
            erreur_totale_absolue_MWh=("abs_erreur", "sum"),
        )
        .reset_index()
    )

    journ["erreur_carre"] = (
        journ["erreur_totale_MWh"] ** 2
    )

    mae_total = (
        journ["erreur_totale_MWh"]
        .abs()
        .mean()
    )

    rmse_total = np.sqrt(
        journ["erreur_carre"].mean()
    )

    biais_total = (
        journ["erreur_totale_MWh"]
        .mean()
    )

    return {
        "nombre_jours": len(journ),
        "MAE_total_MWh": mae_total,
        "RMSE_total_MWh": rmse_total,
        "biais_total_MWh": biais_total,
        "table_journaliere": journ,
    }


def calculer_metriques_pics(df, colonne_prediction):
    """
    Compare le pic journalier prévu au pic réel.

    Pour le pic prévu :
    - valeur maximale prévue dans la journée
    - heure correspondante

    Pour le pic réel :
    - valeur maximale réellement observée
    - heure correspondante
    """

    resultats = []

    for date, groupe in df.groupby("date_cible"):

        groupe = groupe.sort_values("heure_cible")

        if len(groupe) == 0:
            continue

        # Pic réel
        idx_reel = groupe["consommation_cible"].idxmax()

        pic_reel = groupe.loc[
            idx_reel,
            "consommation_cible"
        ]

        heure_pic_reel = groupe.loc[
            idx_reel,
            "heure_cible"
        ]

        # Pic prévu
        idx_pred = groupe[colonne_prediction].idxmax()

        pic_prevu = groupe.loc[
            idx_pred,
            colonne_prediction
        ]

        heure_pic_prevu = groupe.loc[
            idx_pred,
            "heure_cible"
        ]

        resultats.append(
            {
                "date_cible": date,
                "pic_reel_MW": pic_reel,
                "pic_prevu_MW": pic_prevu,
                "erreur_pic_MW": pic_prevu - pic_reel,
                "abs_erreur_pic_MW": abs(
                    pic_prevu - pic_reel
                ),
                "heure_pic_reel": heure_pic_reel,
                "heure_pic_prevu": heure_pic_prevu,
                "erreur_heure_pic": abs(
                    heure_pic_prevu - heure_pic_reel
                ),
                "heure_pic_correct": (
                    heure_pic_prevu == heure_pic_reel
                ),
            }
        )

    if not resultats:
        return {
            "nombre_jours": 0,
            "MAE_pic_MW": np.nan,
            "biais_pic_MW": np.nan,
            "ecart_moyen_heure_pic": np.nan,
            "taux_heure_pic_correct": np.nan,
            "table_pic": pd.DataFrame(),
        }

    pics = pd.DataFrame(resultats)

    return {
        "nombre_jours": len(pics),
        "MAE_pic_MW": pics["abs_erreur_pic_MW"].mean(),
        "biais_pic_MW": pics["erreur_pic_MW"].mean(),
        "ecart_moyen_heure_pic": (
            pics["erreur_heure_pic"].mean()
        ),
        "taux_heure_pic_correct": (
            100
            * pics["heure_pic_correct"].mean()
        ),
        "table_pic": pics,
    }


# ============================================================
# 5. RÉCUPÉRATION DE LA PRÉVISION RTE J-1
# ============================================================

def recuperer_prevision_rte(date_debut, date_fin):
    """
    Récupère la variable prevision_j1 depuis l'API ODRÉ.

    IMPORTANT :
    L'API impose :
        offset + limit <= 10000

    On découpe donc la période en petites fenêtres temporelles
    afin de ne jamais atteindre cette limite.

    La prévision RTE est utilisée uniquement comme benchmark.
    """

    print()
    print("=" * 70)
    print("RÉCUPÉRATION DE LA PRÉVISION RTE / ÉCO2MIX J-1")
    print("=" * 70)

    date_debut = pd.Timestamp(date_debut).normalize()
    date_fin = pd.Timestamp(date_fin).normalize()

    print(
        f"Période demandée : "
        f"{date_debut.date()} -> {date_fin.date()}"
    )

    # Fenêtres de 3 mois : suffisamment petites pour éviter
    # une pagination trop importante.
    periodes = []

    debut = date_debut

    while debut <= date_fin:
        fin = min(
            debut + pd.DateOffset(months=3) - pd.Timedelta(days=1),
            date_fin,
        )

        periodes.append((debut, fin))

        debut = fin + pd.Timedelta(days=1)

    toutes_les_lignes = []

    for numero, (debut, fin) in enumerate(periodes, start=1):

        print(
            f"\nFenêtre {numero}/{len(periodes)} : "
            f"{debut.date()} -> {fin.date()}"
        )

        # Filtre OData.
        where = (
            "perimetre='France' "
            f"AND date_heure >= '{debut.strftime('%Y-%m-%dT00:00:00')}' "
            f"AND date_heure < "
            f"'{(fin + pd.Timedelta(days=1)).strftime('%Y-%m-%dT00:00:00')}'"
        )

        offset = 0
        limit = 100

        while True:

            params = {
                "where": where,
                "limit": limit,
                "offset": offset,
            }

            try:
                response = requests.get(
                    RTE_URL,
                    params=params,
                    timeout=60,
                )

            except requests.RequestException as e:
                print(
                    f"Erreur réseau RTE : {e}"
                )
                break

            if response.status_code != 200:
                print(
                    f"Erreur API RTE "
                    f"{response.status_code} : "
                    f"{response.text[:500]}"
                )
                break

            try:
                data = response.json()
            except Exception as e:
                print(
                    f"Impossible de lire la réponse JSON : {e}"
                )
                break

            lignes = data.get("results", [])

            if not lignes:
                break

            toutes_les_lignes.extend(lignes)

            print(
                f"  offset={offset} : "
                f"{len(lignes)} lignes"
            )

            # S'il y a moins de lignes que la limite,
            # on a atteint la fin de la fenêtre.
            if len(lignes) < limit:
                break

            offset += limit

            # Sécurité supplémentaire.
            # L'API interdit offset + limit > 10000.
            if offset + limit > 10000:
                print(
                    "  Limite de pagination atteinte "
                    "pour cette fenêtre."
                )
                break

            time.sleep(0.05)

    print()
    print(
        f"Nombre total de lignes RTE récupérées : "
        f"{len(toutes_les_lignes)}"
    )

    if not toutes_les_lignes:
        print(
            "Aucune donnée RTE récupérée."
        )
        return None

    df = pd.DataFrame(toutes_les_lignes)

    print(
        "Colonnes RTE disponibles :"
    )
    print(
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # Vérification des colonnes
    # --------------------------------------------------------

    if "date_heure" not in df.columns:
        print(
            "ERREUR : colonne date_heure absente "
            "des données RTE."
        )
        return None

    if "prevision_j1" not in df.columns:
        print(
            "ERREUR : colonne prevision_j1 absente "
            "des données RTE."
        )
        return None

    # --------------------------------------------------------
    # Date / heure
    # --------------------------------------------------------

    df["date_heure"] = pd.to_datetime(
        df["date_heure"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["date_heure"]
    ).copy()

    # Conversion UTC -> heure française
    df["date_heure"] = (
        df["date_heure"]
        .dt.tz_convert("Europe/Paris")
    )

    df["date_cible"] = (
        df["date_heure"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df["heure_cible"] = (
        df["date_heure"]
        .dt.hour
    )

    # Prévision numérique
    df["prevision_j1"] = pd.to_numeric(
        df["prevision_j1"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "date_cible",
            "heure_cible",
            "prevision_j1",
        ]
    ).copy()

    # --------------------------------------------------------
    # Périmètre France
    # --------------------------------------------------------

    if "perimetre" in df.columns:
        df = df[
            df["perimetre"].astype(str).str.strip()
            == "France"
        ].copy()

    # --------------------------------------------------------
    # Doublons
    # --------------------------------------------------------

    df = (
        df.sort_values(
            ["date_cible", "heure_cible"]
        )
        .drop_duplicates(
            subset=["date_cible", "heure_cible"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # On ne conserve que ce dont nous avons besoin.
    df = df[
        [
            "date_cible",
            "heure_cible",
            "prevision_j1",
        ]
    ].copy()

    # Normalisation finale des types.
    df["date_cible"] = pd.to_datetime(
        df["date_cible"],
        errors="coerce",
    ).dt.normalize()

    df["heure_cible"] = pd.to_numeric(
        df["heure_cible"],
        errors="coerce",
    ).astype("Int64")

    df["prevision_j1"] = pd.to_numeric(
        df["prevision_j1"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "date_cible",
            "heure_cible",
            "prevision_j1",
        ]
    ).copy()

    print(
        f"Nombre de prévisions RTE J-1 utilisables : "
        f"{len(df)}"
    )

    if len(df) > 0:
        print(
            f"Période RTE disponible : "
            f"{df['date_cible'].min().date()} "
            f"-> "
            f"{df['date_cible'].max().date()}"
        )

    return df


# ============================================================
# 6. CHARGEMENT DU DATASET PRINCIPAL
# ============================================================

print("=" * 70)
print("CHARGEMENT DES DONNÉES")
print("=" * 70)

if not DATASET_BASE.exists():
    print(
        f"ERREUR : fichier absent : {DATASET_BASE}"
    )
    sys.exit(1)

df = pd.read_csv(
    DATASET_BASE,
    sep=";",
)

print(
    f"Nombre total de lignes : {len(df)}"
)

# Dates
df["origine_prevision"] = pd.to_datetime(
    df["origine_prevision"],
    errors="coerce",
)

df["date_cible"] = pd.to_datetime(
    df["date_cible"],
    errors="coerce",
).dt.normalize()

df["heure_cible"] = pd.to_numeric(
    df["heure_cible"],
    errors="coerce",
)

# ============================================================
# 7. SPLIT CHRONOLOGIQUE
# ============================================================

train_dates, valid_dates, test_dates = (
    calculer_split_dates(df)
)

print()
print("=" * 70)
print("DÉCOUPAGE CHRONOLOGIQUE")
print("=" * 70)

afficher_periode("TRAIN", train_dates)
afficher_periode("VALIDATION", valid_dates)
afficher_periode("TEST", test_dates)

df_test = df[
    df["origine_prevision"].isin(test_dates)
].copy()

print(
    f"Nombre de lignes test : {len(df_test)}"
)

# Normalisation finale des clés.
df_test["date_cible"] = pd.to_datetime(
    df_test["date_cible"],
    errors="coerce",
).dt.normalize()

df_test["heure_cible"] = pd.to_numeric(
    df_test["heure_cible"],
    errors="coerce",
)


# ============================================================
# 8. BENCHMARKS
# ============================================================

print()
print("=" * 70)
print("CONSTRUCTION DES BENCHMARKS")
print("=" * 70)

# ------------------------------------------------------------
# Benchmark J-7
# ------------------------------------------------------------

df_test["prediction_j7"] = (
    df_test["consommation_j7"]
)

# ------------------------------------------------------------
# Benchmark J-1 an
# ------------------------------------------------------------

df_test["prediction_j1an"] = (
    df_test["consommation_j1an"]
)

# ------------------------------------------------------------
# Benchmark moyenne
# ------------------------------------------------------------

df_test["prediction_moyenne"] = (
    df_test["consommation_j7"]
    + df_test["consommation_j1an"]
) / 2.0

# ------------------------------------------------------------
# Benchmark dernier jour complet disponible à 14h
#
# Pour prévoir J+1 à 14h le jour J :
# le dernier jour entièrement observé est J-1.
#
# Pour une heure h du jour J+1 :
# on prend donc J-1 à la même heure.
# Cela correspond à date_cible - 2 jours.
# ------------------------------------------------------------

df_test["date_dernier_jour_complet"] = (
    df_test["date_cible"]
    - pd.Timedelta(days=2)
)

# On utilise le dataset complet pour récupérer
# la consommation du dernier jour complet.
historique = df[
    [
        "date_cible",
        "heure_cible",
        "consommation_cible",
    ]
].copy()

historique["date_cible"] = pd.to_datetime(
    historique["date_cible"],
    errors="coerce",
).dt.normalize()

historique["heure_cible"] = pd.to_numeric(
    historique["heure_cible"],
    errors="coerce",
)

historique = (
    historique
    .dropna(
        subset=[
            "date_cible",
            "heure_cible",
            "consommation_cible",
        ]
    )
    .drop_duplicates(
        subset=["date_cible", "heure_cible"]
    )
)

df_test = df_test.merge(
    historique.rename(
        columns={
            "date_cible":
                "date_dernier_jour_complet",
            "consommation_cible":
                "prediction_dernier_jour",
        }
    ),
    on=[
        "date_dernier_jour_complet",
        "heure_cible",
    ],
    how="left",
)

# ============================================================
# 9. RANDOM FOREST SANS MÉTÉO
# ============================================================

print()
print("=" * 70)
print("RANDOM FOREST SANS MÉTÉO")
print("=" * 70)

if not MODELE_BASE.exists():
    print(
        f"ERREUR : modèle absent : {MODELE_BASE}"
    )
    sys.exit(1)

modele_base = joblib.load(
    MODELE_BASE
)

features_base = [
    "consommation_j7",
    "consommation_j1an",
    "heure",
    "jour_semaine",
    "mois",
    "weekend",
]

print(
    "Variables utilisées :"
)

for variable in features_base:
    print(f"  - {variable}")

# Vérification des colonnes
colonnes_manquantes = [
    c for c in features_base
    if c not in df_test.columns
]

if colonnes_manquantes:
    print(
        "ERREUR : variables manquantes :"
    )
    print(colonnes_manquantes)
    sys.exit(1)

df_test["prediction_random_forest"] = (
    modele_base.predict(
        df_test[features_base]
    )
)


# ============================================================
# 10. RANDOM FOREST + SYNOP
# ============================================================

print()
print("=" * 70)
print("RANDOM FOREST + SYNOP")
print("=" * 70)

if DATASET_SYNOP.exists() and MODELE_SYNOP.exists():

    df_synop = pd.read_csv(
        DATASET_SYNOP,
        sep=";",
    )

    print(
        f"Lignes dataset SYNOP : "
        f"{len(df_synop)}"
    )

    df_synop["origine_prevision"] = pd.to_datetime(
        df_synop["origine_prevision"],
        errors="coerce",
    )

    df_synop["date_cible"] = pd.to_datetime(
        df_synop["date_cible"],
        errors="coerce",
    ).dt.normalize()

    df_synop["heure_cible"] = pd.to_numeric(
        df_synop["heure_cible"],
        errors="coerce",
    )

    # On ne garde que la période test.
    df_synop_test = df_synop[
        df_synop["origine_prevision"].isin(
            test_dates
        )
    ].copy()

    print(
        "Lignes SYNOP dans la période test "
        "commune : "
        f"{len(df_synop_test)}"
    )

    features_synop = [
        "consommation_j7",
        "consommation_j1an",
        "heure",
        "jour_semaine",
        "mois",
        "weekend",
        "temperature_moyenne",
        "temperature_min",
        "temperature_max",
    ]

    print(
        "Variables réellement utilisées "
        "lors de l'entraînement :"
    )

    for variable in features_synop:
        print(f"  - {variable}")

    modele_synop = joblib.load(
        MODELE_SYNOP
    )

    # Vérification
    colonnes_manquantes_synop = [
        c for c in features_synop
        if c not in df_synop_test.columns
    ]

    if colonnes_manquantes_synop:
        print(
            "ATTENTION : variables SYNOP manquantes :"
        )
        print(colonnes_manquantes_synop)

        df_synop_test = pd.DataFrame()

    else:

        df_synop_test = df_synop_test.dropna(
            subset=features_synop
        ).copy()

        print(
            "Lignes SYNOP utilisables pour "
            "la prédiction : "
            f"{len(df_synop_test)}"
        )

        if len(df_synop_test) > 0:

            df_synop_test[
                "prediction_random_forest_synop"
            ] = modele_synop.predict(
                df_synop_test[
                    features_synop
                ]
            )

            print(
                "Prédictions SYNOP alignées : "
                f"{len(df_synop_test)}"
            )

else:

    print(
        "ATTENTION : dataset ou modèle SYNOP absent."
    )

    print(
        f"Dataset attendu : {DATASET_SYNOP}"
    )

    print(
        f"Modèle attendu : {MODELE_SYNOP}"
    )

    df_synop_test = pd.DataFrame()


# ============================================================
# 11. PRÉVISION RTE J-1
# ============================================================

df_rte = recuperer_prevision_rte(
    df_test["date_cible"].min(),
    df_test["date_cible"].max(),
)

if df_rte is not None:

    # --------------------------------------------------------
    # CORRECTION IMPORTANTE :
    # les deux clés de fusion doivent avoir exactement
    # le même type datetime64[ns].
    # --------------------------------------------------------

    df_test["date_cible"] = pd.to_datetime(
        df_test["date_cible"],
        errors="coerce",
    ).dt.normalize()

    df_rte["date_cible"] = pd.to_datetime(
        df_rte["date_cible"],
        errors="coerce",
    ).dt.normalize()

    df_test["heure_cible"] = pd.to_numeric(
        df_test["heure_cible"],
        errors="coerce",
    ).astype("Int64")

    df_rte["heure_cible"] = pd.to_numeric(
        df_rte["heure_cible"],
        errors="coerce",
    ).astype("Int64")

    print()
    print(
        "Types avant fusion RTE :"
    )
    print(
        "df_test date_cible :",
        df_test["date_cible"].dtype,
    )
    print(
        "df_rte date_cible :",
        df_rte["date_cible"].dtype,
    )

    df_test = df_test.merge(
        df_rte,
        on=[
            "date_cible",
            "heure_cible",
        ],
        how="left",
    )

else:

    df_test["prevision_j1"] = np.nan


# ============================================================
# 12. CALCUL DES MÉTRIQUES
# ============================================================

print()
print("=" * 70)
print("CALCUL DES MÉTRIQUES")
print("=" * 70)


# ------------------------------------------------------------
# Liste des modèles
# ------------------------------------------------------------

predictions_base = {
    "prediction_random_forest":
        "prediction_random_forest",

    "benchmark_j7":
        "prediction_j7",

    "benchmark_j1an":
        "prediction_j1an",

    "benchmark_moyenne":
        "prediction_moyenne",

    "benchmark_dernier_jour_complet":
        "prediction_dernier_jour_complet",

    "prediction_rte_j1":
        "prevision_j1",
}


# ============================================================
# 13. MÉTRIQUES HORAIRES
# ============================================================

resultats_horaires = []

for nom_modele, colonne in predictions_base.items():

    if colonne not in df_test.columns:
        continue

    metriques = calculer_metriques_horaires(
        df_test["consommation_cible"],
        df_test[colonne],
    )

    resultats_horaires.append(
        {
            "modele": nom_modele,
            **metriques,
        }
    )


# ------------------------------------------------------------
# SYNOP
# ------------------------------------------------------------

if (
    not df_synop_test.empty
    and "prediction_random_forest_synop"
    in df_synop_test.columns
):

    metriques = calculer_metriques_horaires(
        df_synop_test["consommation_cible"],
        df_synop_test[
            "prediction_random_forest_synop"
        ],
    )

    resultats_horaires.append(
        {
            "modele":
                "random_forest_synop",
            **metriques,
        }
    )


df_resultats_horaires = pd.DataFrame(
    resultats_horaires
)


# ============================================================
# 14. MÉTRIQUES JOURNALIÈRES
# ============================================================

resultats_journaliers = []

tables_journalieres = {}

for nom_modele, colonne in predictions_base.items():

    if colonne not in df_test.columns:
        continue

    sous_df = df_test[
        [
            "date_cible",
            "heure_cible",
            "consommation_cible",
            colonne,
        ]
    ].dropna(
        subset=[
            "consommation_cible",
            colonne,
        ]
    ).copy()

    if len(sous_df) == 0:
        continue

    metriques = calculer_metriques_journalieres(
        sous_df,
        colonne,
    )

    tables_journalieres[
        nom_modele
    ] = metriques["table_journaliere"]

    resultats_journaliers.append(
        {
            "modele": nom_modele,
            "nombre_jours":
                metriques["nombre_jours"],
            "MAE_total_MWh":
                metriques["MAE_total_MWh"],
            "RMSE_total_MWh":
                metriques["RMSE_total_MWh"],
            "biais_total_MWh":
                metriques["biais_total_MWh"],
        }
    )


# ------------------------------------------------------------
# SYNOP journalier
# ------------------------------------------------------------

if (
    not df_synop_test.empty
    and "prediction_random_forest_synop"
    in df_synop_test.columns
):

    sous_df = df_synop_test[
        [
            "date_cible",
            "heure_cible",
            "consommation_cible",
            "prediction_random_forest_synop",
        ]
    ].dropna(
        subset=[
            "consommation_cible",
            "prediction_random_forest_synop",
        ]
    ).copy()

    if len(sous_df) > 0:

        metriques = calculer_metriques_journalieres(
            sous_df,
            "prediction_random_forest_synop",
        )

        tables_journalieres[
            "random_forest_synop"
        ] = metriques["table_journaliere"]

        resultats_journaliers.append(
            {
                "modele":
                    "random_forest_synop",
                "nombre_jours":
                    metriques["nombre_jours"],
                "MAE_total_MWh":
                    metriques["MAE_total_MWh"],
                "RMSE_total_MWh":
                    metriques["RMSE_total_MWh"],
                "biais_total_MWh":
                    metriques["biais_total_MWh"],
            }
        )


df_resultats_journaliers = pd.DataFrame(
    resultats_journaliers
)


# ============================================================
# 15. MÉTRIQUES DES PICS
# ============================================================

resultats_pics = []

tables_pics = {}

for nom_modele, colonne in predictions_base.items():

    if colonne not in df_test.columns:
        continue

    sous_df = df_test[
        [
            "date_cible",
            "heure_cible",
            "consommation_cible",
            colonne,
        ]
    ].dropna(
        subset=[
            "consommation_cible",
            colonne,
        ]
    ).copy()

    if len(sous_df) == 0:
        continue

    metriques = calculer_metriques_pics(
        sous_df,
        colonne,
    )

    tables_pics[
        nom_modele
    ] = metriques["table_pic"]

    resultats_pics.append(
        {
            "modele": nom_modele,
            "nombre_jours":
                metriques["nombre_jours"],
            "MAE_pic_MW":
                metriques["MAE_pic_MW"],
            "biais_pic_MW":
                metriques["biais_pic_MW"],
            "ecart_moyen_heure_pic":
                metriques[
                    "ecart_moyen_heure_pic"
                ],
            "taux_heure_pic_correct":
                metriques[
                    "taux_heure_pic_correct"
                ],
        }
    )


# ------------------------------------------------------------
# SYNOP pics
# ------------------------------------------------------------

if (
    not df_synop_test.empty
    and "prediction_random_forest_synop"
    in df_synop_test.columns
):

    sous_df = df_synop_test[
        [
            "date_cible",
            "heure_cible",
            "consommation_cible",
            "prediction_random_forest_synop",
        ]
    ].dropna(
        subset=[
            "consommation_cible",
            "prediction_random_forest_synop",
        ]
    ).copy()

    if len(sous_df) > 0:

        metriques = calculer_metriques_pics(
            sous_df,
            "prediction_random_forest_synop",
        )

        tables_pics[
            "random_forest_synop"
        ] = metriques["table_pic"]

        resultats_pics.append(
            {
                "modele":
                    "random_forest_synop",
                "nombre_jours":
                    metriques["nombre_jours"],
                "MAE_pic_MW":
                    metriques["MAE_pic_MW"],
                "biais_pic_MW":
                    metriques["biais_pic_MW"],
                "ecart_moyen_heure_pic":
                    metriques[
                        "ecart_moyen_heure_pic"
                    ],
                "taux_heure_pic_correct":
                    metriques[
                        "taux_heure_pic_correct"
                    ],
            }
        )


df_resultats_pics = pd.DataFrame(
    resultats_pics
)


# ============================================================
# 16. AFFICHAGE DES RÉSULTATS
# ============================================================

print()
print("=" * 70)
print("MÉTRIQUES HORAIRES")
print("=" * 70)

if len(df_resultats_horaires) > 0:

    print(
        df_resultats_horaires.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

else:

    print(
        "Aucun résultat horaire."
    )


print()
print("=" * 70)
print("MÉTRIQUES JOURNALIÈRES")
print("=" * 70)

if len(df_resultats_journaliers) > 0:

    print(
        df_resultats_journaliers.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

else:

    print(
        "Aucun résultat journalier."
    )


print()
print("=" * 70)
print("MÉTRIQUES DES PICS")
print("=" * 70)

if len(df_resultats_pics) > 0:

    print(
        df_resultats_pics.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

else:

    print(
        "Aucun résultat pic."
    )


# ============================================================
# 17. PIRES JOURNÉES DU RANDOM FOREST DE BASE
# ============================================================

print()
print("=" * 70)
print("PIRES JOURNÉES - RANDOM FOREST SANS MÉTÉO")
print("=" * 70)

df_rf = df_test[
    [
        "date_cible",
        "heure_cible",
        "consommation_cible",
        "prediction_random_forest",
    ]
].dropna().copy()

df_rf["erreur"] = (
    df_rf["prediction_random_forest"]
    - df_rf["consommation_cible"]
)

pires_jours = (
    df_rf
    .groupby("date_cible")
    .agg(
        MAE_jour_MW=(
            "erreur",
            lambda x: np.mean(np.abs(x))
        ),
        RMSE_jour_MW=(
            "erreur",
            lambda x: np.sqrt(
                np.mean(x ** 2)
            )
        ),
        erreur_totale_MWh=(
            "erreur",
            "sum"
        ),
    )
    .reset_index()
)

pires_jours[
    "erreur_totale_absolue_MWh"
] = pires_jours[
    "erreur_totale_MWh"
].abs()

pires_jours = (
    pires_jours
    .sort_values(
        "erreur_totale_absolue_MWh",
        ascending=False,
    )
    .head(10)
)

print(
    pires_jours.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 18. SAUVEGARDE DES RÉSULTATS
# ============================================================

print()
print("=" * 70)
print("SAUVEGARDE")
print("=" * 70)

fichier_horaires = (
    RESULTS_DIR
    / "comparaison_modeles_horaires.csv"
)

fichier_journaliers = (
    RESULTS_DIR
    / "comparaison_modeles_journaliers.csv"
)

fichier_pics = (
    RESULTS_DIR
    / "comparaison_modeles_pics.csv"
)

fichier_pires = (
    RESULTS_DIR
    / "pires_jours_random_forest.csv"
)

fichier_rapport = (
    RESULTS_DIR
    / "rapport_comparaison_modeles.txt"
)

df_resultats_horaires.to_csv(
    fichier_horaires,
    sep=";",
    index=False,
)

df_resultats_journaliers.to_csv(
    fichier_journaliers,
    sep=";",
    index=False,
)

df_resultats_pics.to_csv(
    fichier_pics,
    sep=";",
    index=False,
)

pires_jours.to_csv(
    fichier_pires,
    sep=";",
    index=False,
)


# ============================================================
# 19. RAPPORT TEXTE
# ============================================================

with open(
    fichier_rapport,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "COMPARAISON DES MODELES DE PREVISION\n"
    )
    f.write(
        "=====================================\n\n"
    )

    f.write(
        "Periode du test : "
        f"{df_test['date_cible'].min().date()} "
        "-> "
        f"{df_test['date_cible'].max().date()}\n\n"
    )

    f.write(
        "Variables du Random Forest sans meteo :\n"
    )

    for variable in features_base:
        f.write(
            f"  - {variable}\n"
        )

    f.write("\n")

    f.write(
        "Variables du Random Forest + SYNOP :\n"
    )

    for variable in features_synop:
        f.write(
            f"  - {variable}\n"
        )

    f.write("\n")

    f.write(
        "IMPORTANT : la prevision RTE J-1 "
        "est uniquement un benchmark externe.\n"
    )

    f.write(
        "Elle n'est pas utilisee comme variable "
        "d'entree des Random Forest.\n\n"
    )

    f.write(
        "RESULTATS HORAIRES\n"
        "------------------\n"
    )

    f.write(
        df_resultats_horaires.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "RESULTATS JOURNALIERS\n"
        "---------------------\n"
    )

    f.write(
        df_resultats_journaliers.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "RESULTATS PICS\n"
        "---------------\n"
    )

    f.write(
        df_resultats_pics.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "PIRES JOURNEES DU RANDOM FOREST\n"
        "--------------------------------\n"
    )

    f.write(
        pires_jours.to_string(
            index=False
        )
    )


print()
print(
    "Fichiers créés :"
)

print(
    f"  - {fichier_horaires}"
)

print(
    f"  - {fichier_journaliers}"
)

print(
    f"  - {fichier_pics}"
)

print(
    f"  - {fichier_pires}"
)

print(
    f"  - {fichier_rapport}"
)

print()
print("=" * 70)
print("COMPARAISON TERMINÉE")
print("=" * 70)
"""
02_preparation.py

Préparation des données RTE éCO2mix pour la prévision horaire.

IMPORTANT :
- AUCUNE moyenne entre 2 demi-heures.
- Nous utilisons uniquement les observations exactement à HH:00.
- Exemple :
      07:00 -> consommation de 07:00
      08:00 -> consommation de 08:00
- L'observation 07:30 n'est PAS utilisée pour construire 07:00 ou 08:00.
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

INPUT_FILE = Path("data/raw/eco2mix_national_brut.csv")
OUTPUT_FILE = Path("data/processed/eco2mix_national_horaire.csv")
REPORT_FILE = Path("data/processed/rapport_preparation.txt")

TIMEZONE = "Europe/Paris"

REQUIRED_COLUMNS = [
    "perimetre",
    "date_heure",
    "consommation",
    "nature",
]


# ============================================================
# 2. CRÉATION DES DOSSIERS
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. CHARGEMENT
# ============================================================

print("=" * 70)
print("CHARGEMENT DES DONNÉES")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_FILE}\n"
        "Lance d'abord 01_telechargement.py."
    )

df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    low_memory=False
)

print(f"Nombre de lignes chargées : {len(df):,}")


# ============================================================
# 4. VÉRIFICATION DES COLONNES
# ============================================================

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Colonnes obligatoires manquantes : "
        + ", ".join(missing_columns)
    )


# ============================================================
# 5. PÉRIMÈTRE
# ============================================================

print("\n" + "=" * 70)
print("VÉRIFICATION DU PÉRIMÈTRE")
print("=" * 70)

perimetres = df["perimetre"].dropna().astype(str).str.strip().unique()

print("Périmètres trouvés :", perimetres)

if not (df["perimetre"].astype(str).str.strip() == "France").all():
    raise ValueError(
        "Le fichier contient un périmètre différent de 'France'. "
        "Le script s'arrête pour éviter de mélanger plusieurs périmètres."
    )

print("OK : toutes les observations sont sur le périmètre France.")


# ============================================================
# 6. CONVERSION DE LA DATE
# ============================================================

print("\n" + "=" * 70)
print("CONVERSION DES DATES")
print("=" * 70)

# Les données RTE sont horodatées avec un fuseau.
# On commence par interpréter explicitement les dates comme UTC,
# puis on les convertit en heure locale française.

df["date_heure_utc"] = pd.to_datetime(
    df["date_heure"],
    errors="coerce",
    utc=True
)

nb_dates_invalides = df["date_heure_utc"].isna().sum()

if nb_dates_invalides > 0:
    print(
        f"Attention : {nb_dates_invalides:,} dates invalides supprimées."
    )

df = df.dropna(subset=["date_heure_utc"]).copy()

# Conversion vers l'heure locale française.
df["date_heure"] = (
    df["date_heure_utc"]
    .dt.tz_convert(TIMEZONE)
)


# ============================================================
# 7. CONSOMMATION NUMÉRIQUE
# ============================================================

df["consommation"] = pd.to_numeric(
    df["consommation"],
    errors="coerce"
)

nb_conso_invalides = df["consommation"].isna().sum()

print(
    f"Consommations non numériques / manquantes : "
    f"{nb_conso_invalides:,}"
)


# ============================================================
# 8. TRI CHRONOLOGIQUE
# ============================================================

df = df.sort_values("date_heure_utc").reset_index(drop=True)


# ============================================================
# 9. DOUBLONS
# ============================================================

# On retire uniquement les doublons correspondant au même instant
# physique en UTC.
#
# IMPORTANT :
# lors du passage à l'heure d'hiver, deux heures locales peuvent
# avoir le même affichage mais correspondent à deux instants différents.
# On ne doit donc PAS supprimer les doublons sur date_heure locale.

nb_avant = len(df)

df = df.drop_duplicates(
    subset=["date_heure_utc"],
    keep="first"
).copy()

nb_doublons = nb_avant - len(df)

print(f"Doublons physiques supprimés : {nb_doublons:,}")


# ============================================================
# 10. INFORMATIONS CALENDAIRES
# ============================================================

df["date"] = df["date_heure"].dt.date
df["heure"] = df["date_heure"].dt.hour
df["minute"] = df["date_heure"].dt.minute

df["jour_semaine"] = df["date_heure"].dt.dayofweek
df["mois"] = df["date_heure"].dt.month
df["annee"] = df["date_heure"].dt.year

# 0 = lundi
# 6 = dimanche

df["weekend"] = (
    df["jour_semaine"] >= 5
).astype(int)


# ============================================================
# 11. IDENTIFICATION DES HORAIRES EXACTS
# ============================================================

print("\n" + "=" * 70)
print("SÉLECTION DES HEURES ENTIÈRES")
print("=" * 70)

print(
    "Règle : seule une observation avec minute == 0 est conservée."
)

nb_total = len(df)

nb_heures_entieres = (
    df["minute"] == 0
).sum()

nb_demi_heures = (
    df["minute"] == 30
).sum()

print(f"Observations totales : {nb_total:,}")
print(f"Observations à HH:00 : {nb_heures_entieres:,}")
print(f"Observations à HH:30 : {nb_demi_heures:,}")


# ============================================================
# 12. CONSERVATION UNIQUEMENT DES HH:00
# ============================================================

df_horaire = df[
    df["minute"] == 0
].copy()

df_horaire = df_horaire.sort_values(
    "date_heure_utc"
).reset_index(drop=True)


# ============================================================
# 13. VÉRIFICATION : AUCUNE MOYENNE
# ============================================================

# La consommation horaire est exactement la valeur RTE
# correspondant au timestamp HH:00.
#
# Il n'y a volontairement aucune opération du type :
#
#   mean()
#   resample("1h").mean()
#   interpolation
#
# La colonne consommation_horaire est simplement une copie
# de la consommation observée à HH:00.

df_horaire["consommation_horaire"] = (
    df_horaire["consommation"]
)


# ============================================================
# 14. VÉRIFICATION DES HEURES
# ============================================================

heures_valides = set(range(24))

heures_trouvees = set(
    df_horaire["heure"].dropna().astype(int).unique()
)

heures_inconnues = heures_trouvees - heures_valides

if heures_inconnues:
    raise ValueError(
        f"Heures invalides détectées : {heures_inconnues}"
    )


# ============================================================
# 15. VÉRIFICATION DES VALEURS NÉGATIVES
# ============================================================

nb_negatives = (
    df_horaire["consommation_horaire"] < 0
).sum()

print(
    f"Consommations négatives : {nb_negatives:,}"
)

if nb_negatives > 0:
    print(
        "ATTENTION : les valeurs négatives sont conservées "
        "pour inspection. Elles ne sont pas remplacées."
    )


# ============================================================
# 16. VÉRIFICATION DES VALEURS MANQUANTES
# ============================================================

nb_missing = (
    df_horaire["consommation_horaire"]
    .isna()
    .sum()
)

print(
    f"Consommations manquantes à HH:00 : {nb_missing:,}"
)

# Aucune interpolation.
# Aucune moyenne.
# Les valeurs manquantes restent manquantes.


# ============================================================
# 17. CONTRÔLE DES INTERVALLES TEMPORELS
# ============================================================

print("\n" + "=" * 70)
print("CONTRÔLE DE LA STRUCTURE TEMPORELLE")
print("=" * 70)

# On regarde les écarts entre observations HH:00.
#
# En période normale :
#     1 heure
#
# Au changement d'heure :
#     certains écarts peuvent être différents.
#
# On ne corrige rien automatiquement.

diffs = (
    df_horaire["date_heure_utc"]
    .diff()
    .dropna()
)

print("Écarts temporels observés :")
print(diffs.value_counts().sort_index().head(20))


# ============================================================
# 18. CONTRÔLE DES JOURNÉES
# ============================================================

print("\n" + "=" * 70)
print("CONTRÔLE DU NOMBRE D'HEURES PAR JOUR")
print("=" * 70)

heures_par_jour = (
    df_horaire
    .groupby("date")["heure"]
    .nunique()
)

print(
    "Distribution du nombre d'heures disponibles par jour :"
)

print(
    heures_par_jour
    .value_counts()
    .sort_index()
)


# ============================================================
# 19. IDENTIFICATION DES JOURNÉES COMPLÈTES
# ============================================================

# Pour notre problème de prévision :
#
#     J à 14h
#        ↓
#     prévoir J+1
#        ↓
#     24 valeurs horaires
#
# Une journée cible doit donc disposer des 24 heures HH:00.
#
# On NE crée pas artificiellement les heures manquantes.

dates_completes = heures_par_jour[
    heures_par_jour == 24
].index

dates_incompletes = heures_par_jour[
    heures_par_jour != 24
].index

print(
    f"\nJournées avec exactement 24 heures : "
    f"{len(dates_completes):,}"
)

print(
    f"Journées avec un nombre différent de 24 heures : "
    f"{len(dates_incompletes):,}"
)


# ============================================================
# 20. VARIABLE DE JOURNÉE COMPLÈTE
# ============================================================

df_horaire["jour_complet"] = (
    df_horaire["date"]
    .isin(dates_completes)
)


# ============================================================
# 21. ORDRE FINAL DES COLONNES
# ============================================================

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

df_horaire = df_horaire[
    columns_finales
].copy()


# ============================================================
# 22. VÉRIFICATIONS FINALES
# ============================================================

print("\n" + "=" * 70)
print("VÉRIFICATIONS FINALES")
print("=" * 70)

# Toutes les observations doivent être HH:00.

assert (
    df_horaire["minute"] == 0
).all(), (
    "ERREUR : une observation autre que HH:00 "
    "est présente."
)

# La consommation horaire doit être strictement identique
# à la consommation source.

assert (
    df_horaire["consommation_horaire"]
    .equals(df_horaire["consommation"])
), (
    "ERREUR : consommation_horaire n'est pas identique "
    "à la consommation source."
)

# Pas de doublon physique.

assert not df_horaire["date_heure_utc"].duplicated().any(), (
    "ERREUR : doublon temporel UTC détecté."
)

# Ordre chronologique.

assert df_horaire["date_heure_utc"].is_monotonic_increasing, (
    "ERREUR : données non chronologiques."
)

print("✓ Toutes les observations sont à HH:00.")
print("✓ Aucune moyenne n'a été calculée.")
print("✓ Aucune interpolation n'a été effectuée.")
print("✓ Aucune consommation artificielle n'a été créée.")
print("✓ La consommation horaire = consommation RTE à HH:00.")


# ============================================================
# 23. SAUVEGARDE
# ============================================================

df_horaire.to_csv(
    OUTPUT_FILE,
    sep=";",
    index=False
)

print(
    f"\nFichier préparé enregistré dans : {OUTPUT_FILE}"
)


# ============================================================
# 24. RAPPORT
# ============================================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("RAPPORT DE PRÉPARATION DES DONNÉES\n")
    f.write("=" * 70 + "\n\n")

    f.write("Principe fondamental\n")
    f.write("--------------------\n")
    f.write(
        "La consommation horaire utilisée par le projet correspond "
        "EXACTEMENT à l'observation RTE située à HH:00.\n"
    )
    f.write(
        "Aucune moyenne entre HH:00 et HH:30 n'est calculée.\n"
    )
    f.write(
        "Aucune interpolation n'est réalisée.\n"
    )
    f.write(
        "Aucune valeur artificielle n'est créée.\n\n"
    )

    f.write("Exemple\n")
    f.write("-------\n")
    f.write(
        "07:00 -> consommation observée à 07:00\n"
    )
    f.write(
        "07:30 -> non utilisée pour construire 07:00\n"
    )
    f.write(
        "08:00 -> consommation observée à 08:00\n\n"
    )

    f.write("Statistiques\n")
    f.write("------------\n")
    f.write(
        f"Observations initiales : {nb_total:,}\n"
    )
    f.write(
        f"Observations HH:00 conservées : "
        f"{len(df_horaire):,}\n"
    )
    f.write(
        f"Observations HH:30 écartées : "
        f"{nb_demi_heures:,}\n"
    )
    f.write(
        f"Doublons physiques supprimés : "
        f"{nb_doublons:,}\n"
    )
    f.write(
        f"Consommations manquantes HH:00 : "
        f"{nb_missing:,}\n"
    )
    f.write(
        f"Consommations négatives : "
        f"{nb_negatives:,}\n"
    )
    f.write(
        f"Journées complètes de 24 heures : "
        f"{len(dates_completes):,}\n"
    )
    f.write(
        f"Journées non complètes : "
        f"{len(dates_incompletes):,}\n"
    )

print(
    f"Rapport enregistré dans : {REPORT_FILE}"
)

print("\n" + "=" * 70)
print("PRÉPARATION TERMINÉE")
print("=" * 70)
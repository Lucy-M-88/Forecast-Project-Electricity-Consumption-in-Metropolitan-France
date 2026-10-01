from pathlib import Path
import pandas as pd
import re


# ============================================================
# CONFIGURATION
# ============================================================

RESULTATS_DIR = Path("results")

FICHIER_COMPARAISON = RESULTATS_DIR / "comparaison_modeles.csv"
FICHIER_SYNOP = RESULTATS_DIR / "resultat_modele_synop.txt"

FICHIER_SORTIE = RESULTATS_DIR / "comparaison_modeles_complete.csv"


# ============================================================
# LECTURE DES RESULTATS DES 6 MODELES
# ============================================================

print("=" * 70)
print("COMPARAISON COMMUNE DES MODÈLES")
print("=" * 70)

if not FICHIER_COMPARAISON.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER_COMPARAISON}"
    )

print(f"\nLecture : {FICHIER_COMPARAISON}")

# IMPORTANT : le fichier de 05_evaluation.py utilise ';'
comparaison = pd.read_csv(
    FICHIER_COMPARAISON,
    sep=";"
)

print(f"Nombre de modèles déjà présents : {len(comparaison)}")

print("\nModèles existants :")
for modele in comparaison["modele"]:
    print(f"  - {modele}")


# ============================================================
# LECTURE DU RESULTAT RF + SYNOP
# ============================================================

if not FICHIER_SYNOP.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER_SYNOP}"
    )

print(f"\nLecture : {FICHIER_SYNOP}")

texte = FICHIER_SYNOP.read_text(encoding="utf-8")


# ============================================================
# EXTRACTION DES METRIQUES TEST
# ============================================================

def extraire_float(pattern, texte):
    match = re.search(pattern, texte, re.IGNORECASE)

    if match is None:
        raise ValueError(
            f"Impossible de trouver la métrique correspondant à : {pattern}"
        )

    return float(match.group(1).replace(",", "."))


mae_synop = extraire_float(
    r"Test MAE\s*:\s*([0-9]+(?:[.,][0-9]+)?)",
    texte
)

rmse_synop = extraire_float(
    r"Test RMSE\s*:\s*([0-9]+(?:[.,][0-9]+)?)",
    texte
)


# ============================================================
# CREATION DE LA LIGNE RF + SYNOP
# ============================================================

ligne_synop = pd.DataFrame([{
    "modele": "prediction_random_forest_synop",
    "nombre_heures": pd.NA,
    "MAE_MW": mae_synop,
    "RMSE_MW": rmse_synop
}])


# ============================================================
# AJOUT AU TABLEAU EXISTANT
# ============================================================

# Si le script est relancé, éviter d'ajouter deux fois le modèle SYNOP
comparaison = comparaison[
    comparaison["modele"] != "prediction_random_forest_synop"
].copy()

comparaison_complete = pd.concat(
    [comparaison, ligne_synop],
    ignore_index=True
)


# ============================================================
# AFFICHAGE
# ============================================================

print("\n" + "=" * 70)
print("COMPARAISON FINALE")
print("=" * 70)

print(
    comparaison_complete.to_string(index=False)
)


# ============================================================
# SAUVEGARDE
# ============================================================

comparaison_complete.to_csv(
    FICHIER_SORTIE,
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("Fichier sauvegardé :")
print(FICHIER_SORTIE)
print("=" * 70)


# ============================================================
# RAPPEL METHODOLOGIQUE
# ============================================================

print("""
ATTENTION :

Le RF + SYNOP n'a pas été évalué sur la même période de test
que les modèles du fichier 05_evaluation.py.

RF classique :
    2024-06-20 -> 2026-06-29

RF + SYNOP :
    2025-09-13 -> 2025-12-31

Les résultats sont donc regroupés dans un même tableau,
mais les MAE/RMSE ne sont pas directement comparables comme
si les deux modèles avaient été évalués sur exactement le
même échantillon de test.
""")
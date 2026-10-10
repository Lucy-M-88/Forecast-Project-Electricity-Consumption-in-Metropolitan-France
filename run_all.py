# -*- coding: utf-8 -*-

"""
run_all.py

Script de lancement automatique du pipeline complet.

Exécute dans l'ordre :
    1. 01_telechargement.py       (téléchargement éCO2mix)
    2. 02_preparation.py          (préparation nationale)
    3. 03_creation_dataset.py     (dataset supervisé national)
    4. 04_meteo.py                (météo nationale)
    5. 05_modelisation.py         (modèles nationaux)
    6. 08_consommation_regionale.py (conso régionale éCO2mix)
    7. 09_meteo_regionale.py      (météo régionale)
    8. 12_completer_aura.py       (complétion AURA)
    9. 11_modelisation_regionale.py (modèles régionaux)
   10. 07_dashboard.py            (Dashboard Streamlit)

Utilisation :
    python run_all.py

Prérequis :
    - Python 3.9+
    - Avoir lancé 04_meteo.py au moins une fois pour que
      meteofrance_reduit/ existe (ou avoir les fichiers bruts).

Le script s'arrête si une étape échoue.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "source"

# Dossier où se trouvent les scripts Python
SCRIPTS_DIR = SOURCE_DIR if SOURCE_DIR.exists() else BASE_DIR

# Pipeline : (nom_affichage, fichier_python, args_optionnels)
PIPELINE = [
    ("01 - Téléchargement éCO2mix national",   "01_telechargement.py",          []),
    ("02 - Préparation nationale",             "02_preparation.py",             []),
    ("03 - Création dataset supervisé",        "03_creation_dataset.py",        []),
    ("04 - Météo nationale",                   "04_meteo.py",                   []),
    ("05 - Modélisation nationale",            "05_modelisation.py",            []),
    ("08 - Conso régionale éCO2mix",           "08_consommation_regionale.py",  []),
    ("09 - Météo régionale",                   "09_meteo_regionale.py",         []),
    ("12 - Complétion AURA",                   "12_completer_aura.py",          []),
    ("11 - Modélisation régionale",            "11_modelisation_regionale.py",  []),
]

# Le dashboard est lancé à part (via streamlit run)
DASHBOARD_SCRIPT = "07_dashboard.py"


# ============================================================
# 2. UTILITAIRES
# ============================================================

def print_header(texte: str, char: str = "=", largeur: int = 78):

    print()
    print(char * largeur)
    print(texte)
    print(char * largeur)


def print_banniere():

    print()
    print("=" * 78)
    print(" " * 20 + "PRÉVISION DE CONSOMMATION ÉLECTRIQUE")
    print(" " * 25 + "Lancement complet du pipeline")
    print("=" * 78)
    print()
    print(f"Dossier source  : {SCRIPTS_DIR}")
    print(f"Dossier racine  : {BASE_DIR}")
    print()
    print(f"Étapes prévues  : {len(PIPELINE)}")
    for i, (nom, _, _) in enumerate(PIPELINE, 1):
        print(f"  {i:2d}. {nom}")
    print(f"  {len(PIPELINE) + 1:2d}. Lancement du Dashboard Streamlit")
    print()


def verifier_python():

    version = sys.version_info

    print(f"Python détecté : {sys.version.split()[0]}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(" Python 3.9+ requis.")
        sys.exit(1)

    print("✓ Version Python compatible.")


def verifier_dependances():

    print("\nVérification des dépendances...")

    dependances = [
        "pandas",
        "numpy",
        "requests",
        "sklearn",
        "xgboost",
        "joblib",
        "streamlit",
        "plotly",
        "matplotlib",
    ]

    manquantes = []

    for pkg in dependances:

        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (manquant)")
            manquantes.append(pkg)

    if manquantes:

        print()
        print("=" * 78)
        print(" Dépendances manquantes")
        print("=" * 78)
        print()
        print("Installe-les avec :")
        print()
        print("    pip install " + " ".join(manquantes))
        print()
        print("Ou avec le requirements.txt du projet :")
        print("    pip install -r requirements.txt")
        print()
        sys.exit(1)

    print("\n✓ Toutes les dépendances sont installées.")


def verifier_scripts():

    print("\nVérification des scripts...")

    manquants = []

    tous_scripts = [s[1] for s in PIPELINE] + [DASHBOARD_SCRIPT]

    for script in tous_scripts:

        chemin = SCRIPTS_DIR / script

        if chemin.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} (introuvable)")
            manquants.append(script)

    if manquants:

        print()
        print("=" * 78)
        print(" Scripts manquants")
        print("=" * 78)
        print()
        for s in manquants:
            print(f"  - {s}")

        # Chercher dans BASE_DIR si pas trouvés dans SCRIPTS_DIR
        if SCRIPTS_DIR != BASE_DIR:
            print()
            print(f"Vérifie qu'ils sont bien dans : {SCRIPTS_DIR}")

        sys.exit(1)

    print("\n✓ Tous les scripts sont présents.")


def executer_script(nom_etape, script, args, index, total):

    print()
    print("=" * 78)
    print(f"ÉTAPE {index}/{total} : {nom_etape}")
    print("=" * 78)

    chemin_script = SCRIPTS_DIR / script

    if not chemin_script.exists():
        print(f"❌ Script introuvable : {chemin_script}")
        return False

    cmd = [sys.executable, str(chemin_script)] + args

    print(f"Commande : {' '.join(str(c) for c in cmd)}")
    print(f"Dossier  : {SCRIPTS_DIR}")
    print()

    t0 = time.time()

    # On lance depuis SCRIPTS_DIR pour que les chemins relatifs
    # dans les scripts (data/, models/, results/) soient cohérents.
    try:
        resultat = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            check=False,
        )
    except KeyboardInterrupt:
        print("\n\n Interruption manuelle (Ctrl+C).")
        return False
    except Exception as e:
        print(f"\n Erreur lors de l'exécution : {e}")
        return False

    duree = time.time() - t0

    if resultat.returncode != 0:
        print()
        print(f" Échec (code retour {resultat.returncode}) "
              f"après {duree:.1f}s")
        return False

    print()
    print(f"✓ Terminé en {duree:.1f}s")
    return True


def lancer_dashboard():

    print()
    print("=" * 78)
    print("LANCEMENT DU DASHBOARD STREAMLIT")
    print("=" * 78)
    print()
    print("Le dashboard va s'ouvrir dans ton navigateur.")
    print("Pour l'arrêter : Ctrl+C dans ce terminal.")
    print()
    print(f"URL : http://localhost:8501")
    print()

    chemin_dashboard = SCRIPTS_DIR / DASHBOARD_SCRIPT

    if not chemin_dashboard.exists():
        print(f" Dashboard introuvable : {chemin_dashboard}")
        return False

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(chemin_dashboard),
    ]

    try:
        subprocess.run(cmd, cwd=str(SCRIPTS_DIR), check=False)
    except KeyboardInterrupt:
        print("\n\n Dashboard arrêté.")
    except Exception as e:
        print(f"\n Erreur Streamlit : {e}")
        return False

    return True


# ============================================================
# 3. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print_banniere()

    # --------------------------------------------------------
    # Vérifications préliminaires
    # --------------------------------------------------------

    verifier_python()
    verifier_dependances()
    verifier_scripts()

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("PRÊT À LANCER LE PIPELINE")
    print("=" * 78)
    print()
    print("⚠ Le pipeline complet peut prendre 20-60 minutes selon :")
    print("   - la vitesse de ta connexion (téléchargements)")
    print("   - la puissance de ta machine (entraînements)")
    print()
    print("   Les modèles régionaux (11) sont les plus longs.")
    print()

    try:
        reponse = input("Continuer ? [o/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAnnulé.")
        return

    if reponse not in ("o", "oui", "y", "yes"):
        print("Annulé.")
        return

    # --------------------------------------------------------
    # Exécution du pipeline
    # --------------------------------------------------------

    t_debut = time.time()
    total = len(PIPELINE)

    for i, (nom, script, args) in enumerate(PIPELINE, 1):

        ok = executer_script(nom, script, args, i, total)

        if not ok:
            print()
            print("=" * 78)
            print(f" ARRÊT DU PIPELINE À L'ÉTAPE {i}/{total}")
            print("=" * 78)
            print()
            print(f"L'étape '{nom}' a échoué.")
            print()
            print("Pour corriger :")
            print(f"  1. Regarde les erreurs au-dessus")
            print(f"  2. Corrige le script {script}")
            print(f"  3. Relance ce fichier run_all.py")
            print()
            return

    duree_totale = time.time() - t_debut

    print()
    print("=" * 78)
    print("✓ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 78)
    print()
    print(f"Durée totale : {duree_totale / 60:.1f} minutes")
    print()

    # --------------------------------------------------------
    # Lancement du dashboard
    # --------------------------------------------------------

    lancer_dashboard()


if __name__ == "__main__":
    main()
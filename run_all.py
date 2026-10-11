# -*- coding: utf-8 -*-

"""
run_all.py

Pipeline complet de prévision de la consommation électrique
en France métropolitaine.

Lance dans l'ordre :
    01_telechargement.py            Telechargement eCO2mix national
    02_preparation.py               Preparation horaire
    03_creation_dataset.py          Dataset supervise
    04_meteo.py                     Meteo nationale agregee
    05_modelisation.py              2 scenarios (op + meteo parfaite)
    06_prevision_operationnelle.py  Prevision J+1 (API Open-Meteo, non bloquant)
    08_consommation_regionale.py    Conso regionale eCO2mix
    09_meteo_regionale.py           Meteo regionale agregee
    12_completer_aura.py            Completion AURA (non bloquant)
    11_modelisation_regionale.py    12 modeles regionaux
    07_dashboard.py                 Dashboard Streamlit (a la fin)

Note :
    Le script 10_prevision_regionale.py (fallback rapide utilisant le
    modele national) n'est PAS inclus : 11_modelisation_regionale.py
    produit un meilleur resultat et ecrit le meme fichier de sortie.

Compatibilite :
    - Windows 11 (PowerShell / cmd / Git Bash)
    - macOS (zsh / bash)
    - Linux (bash)

Usage :
    python run_all.py
    python run_all.py --no-dashboard
    python run_all.py --start 5          Reprend a l'etape 5
    python run_all.py --only 5           Execute uniquement l'etape 5
    python run_all.py --list             Liste les etapes
    python run_all.py --yes              Sans confirmation
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPTS_DIR = Path(__file__).resolve().parent

DASHBOARD_SCRIPT = "07_dashboard.py"

# Etapes dont l'echec ne stoppe PAS le pipeline
ETAPES_NON_BLOQUANTES = {
    "06_prevision_operationnelle.py",
    "12_completer_aura.py",
}

# Pipeline : (nom affiche, nom du script, arguments)
PIPELINE = [
    ("Telechargement eCO2mix national",      "01_telechargement.py",           []),
    ("Preparation des donnees horaires",     "02_preparation.py",              []),
    ("Creation du dataset de prevision",     "03_creation_dataset.py",         []),
    ("Preparation de la meteo nationale",    "04_meteo.py",                    []),
    ("Modelisation nationale (2 scenarios)", "05_modelisation.py",             []),
    ("Prevision operationnelle J+1 (API)",   "06_prevision_operationnelle.py", []),
    ("Telechargement conso regionale",       "08_consommation_regionale.py",   []),
    ("Agregation meteo regionale",           "09_meteo_regionale.py",          []),
    ("Completion AURA (reconstruit)",        "12_completer_aura.py",           []),
    ("Modelisation regionale (12 regions)",  "11_modelisation_regionale.py",   []),
]

DEPENDANCES_PYTHON = [
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


# ============================================================
# BANNIERE
# ============================================================

def print_banniere():
    print()
    print("=" * 78)
    print(" PIPELINE COMPLET - PREVISION DE CONSOMMATION ELECTRIQUE")
    print(" France metropolitaine - Master 2 Temporal Data Analysis")
    print("=" * 78)
    print()
    print(f"Repertoire : {SCRIPTS_DIR}")
    print(f"Python     : {sys.executable}")
    print()


# ============================================================
# PREPARATION DE L'ENVIRONNEMENT
# ============================================================

def preparer_liens_symboliques():
    """
    Cree les sous-dossiers necessaires puis, si les modeles
    existent deja (run precedent), cree les alias de modeles.
    """
    for sous in ["data", "data/raw", "data/processed",
                 "models", "results"]:
        (SCRIPTS_DIR / sous).mkdir(parents=True, exist_ok=True)

    creer_alias_modeles(silencieux=True)


def creer_alias_modeles(silencieux: bool = False):
    """
    Cree des alias (symlink ou copie) pour assurer la compatibilite
    entre les noms de modeles produits par 05_modelisation.py
    (suffixes _operationnel / _meteo_parfaite) et ceux attendus
    par 06_prevision_operationnelle.py et 10_prevision_regionale.py.

    Sur Windows, os.symlink() echoue souvent (necessite admin ou
    mode developpeur). On bascule alors sur une copie.
    """
    models_dir = SCRIPTS_DIR / "models"

    if not models_dir.exists():
        return

    # (nom_cible_ancien, nom_source_nouveau)
    alias = [
        ("modele_ensemble_rf_xgb.joblib",
         "modele_ensemble_rf_xgb_operationnel.joblib"),
        ("modele_xgboost_meteo.joblib",
         "modele_xgboost_operationnel.joblib"),
    ]

    for cible_nom, source_nom in alias:

        cible = models_dir / cible_nom
        source = models_dir / source_nom

        if cible.exists():
            continue

        if not source.exists():
            continue

        # Tentative de symlink (macOS/Linux, ou Windows en mode dev)
        try:
            cible.symlink_to(source.name)
            if not silencieux:
                print(f"  [symlink] {cible_nom} -> {source_nom}")
            continue
        except (OSError, NotImplementedError, AttributeError):
            pass

        # Fallback : copie (Windows sans privileges)
        try:
            shutil.copy2(source, cible)
            if not silencieux:
                print(f"  [copy]    {cible_nom} -> {source_nom}")
        except Exception as e:
            if not silencieux:
                print(f"  [WARN]    alias {cible_nom} : {e}")


# ============================================================
# VERIFICATIONS
# ============================================================

def verifier_python():
    version = sys.version_info
    print(f"Python detecte : {sys.version.split()[0]}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("[ERREUR] Python 3.9+ requis.")
        sys.exit(1)

    print("[OK] Version compatible.")


def verifier_dependances():

    print("\nVerification des dependances...")

    manquantes = []

    for pkg in DEPENDANCES_PYTHON:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [MANQUANT] {pkg}")
            manquantes.append(pkg)

    if manquantes:
        print()
        print("=" * 78)
        print("[ERREUR] Dependances manquantes")
        print("=" * 78)
        print()
        print("Installe-les avec :")
        print()
        print("    pip install " + " ".join(manquantes))
        print()
        sys.exit(1)

    print("\n[OK] Toutes les dependances sont installees.")


def verifier_scripts():

    print("\nVerification des scripts...")

    manquants = []
    tous_scripts = [s[1] for s in PIPELINE] + [DASHBOARD_SCRIPT]

    for script in tous_scripts:
        chemin = SCRIPTS_DIR / script
        if chemin.exists():
            print(f"  [OK] {script}")
        else:
            print(f"  [MANQUANT] {script}")
            manquants.append(script)

    if manquants:
        print()
        print("=" * 78)
        print("[ERREUR] Scripts manquants")
        print("=" * 78)
        for s in manquants:
            print(f"  - {s}")
        sys.exit(1)

    print("\n[OK] Tous les scripts sont presents.")


# ============================================================
# EXECUTION D'UN SCRIPT
# ============================================================

def executer_script(nom_etape, script, args, index, total):

    print()
    print("=" * 78)
    print(f"ETAPE {index}/{total} : {nom_etape}")
    print("=" * 78)

    chemin_script = SCRIPTS_DIR / script

    if not chemin_script.exists():
        print(f"[ERREUR] Script introuvable : {chemin_script}")
        return False

    cmd = [sys.executable, str(chemin_script)] + [str(a) for a in args]

    print(f"Commande : {' '.join(cmd)}")
    print()

    # Force UTF-8 pour le sous-processus (indispensable sur Windows)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    t0 = time.time()

    try:
        resultat = subprocess.run(
            cmd,
            cwd=str(SCRIPTS_DIR),
            check=False,
            env=env,
        )
    except KeyboardInterrupt:
        print("\n\n[ATTENTION] Interruption manuelle (Ctrl+C).")
        return False
    except Exception as e:
        print(f"\n[ERREUR] Exception lors de l'execution : {e}")
        return False

    duree = time.time() - t0

    if resultat.returncode != 0:
        print()
        print(f"[ECHEC] Code retour {resultat.returncode} apres {duree:.1f}s")
        return False

    print()
    print(f"[OK] Termine en {duree:.1f}s")

    # Hook : apres 05_modelisation.py, creer les alias de modeles
    if script == "05_modelisation.py":
        print()
        print("Creation des alias de modeles pour compatibilite...")
        creer_alias_modeles()

    return True


# ============================================================
# DASHBOARD
# ============================================================

def lancer_dashboard():

    print()
    print("=" * 78)
    print("LANCEMENT DU DASHBOARD STREAMLIT")
    print("=" * 78)
    print()
    print("Le dashboard va s'ouvrir dans ton navigateur.")
    print("Pour l'arreter : Ctrl+C dans ce terminal.")
    print()
    print("URL : http://localhost:8501")
    print()

    chemin_dashboard = SCRIPTS_DIR / DASHBOARD_SCRIPT

    if not chemin_dashboard.exists():
        print(f"[ERREUR] Dashboard introuvable : {chemin_dashboard}")
        return False

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(chemin_dashboard),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        subprocess.run(cmd, cwd=str(SCRIPTS_DIR), check=False, env=env)
    except KeyboardInterrupt:
        print("\n\n[ATTENTION] Dashboard arrete.")
    except Exception as e:
        print(f"\n[ERREUR] Streamlit : {e}")
        return False

    return True


# ============================================================
# CLI
# ============================================================

def parser_arguments():

    parser = argparse.ArgumentParser(
        description="Pipeline complet de prevision de consommation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Ne pas lancer le dashboard a la fin.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Reprendre a l'etape N (1-indexee).",
    )

    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help="N'executer que l'etape N (1-indexee).",
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Ne pas demander de confirmation.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister les etapes du pipeline et quitter.",
    )

    return parser.parse_args()


def afficher_liste():
    print()
    print("Etapes du pipeline :")
    print()
    for i, (nom, script, _) in enumerate(PIPELINE, 1):
        bloquant = "non bloquant" if script in ETAPES_NON_BLOQUANTES else "bloquant"
        print(f"  {i:2d}. {nom:<40s} ({script}) [{bloquant}]")
    print()
    print(f"  Dashboard final : {DASHBOARD_SCRIPT}")
    print()


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    args = parser_arguments()

    if args.list:
        afficher_liste()
        return

    print_banniere()

    preparer_liens_symboliques()

    verifier_python()
    verifier_dependances()
    verifier_scripts()

    # ------------------------------------------------
    # Selection des etapes
    # ------------------------------------------------

    total = len(PIPELINE)

    if args.only is not None:
        if not (1 <= args.only <= total):
            print(f"[ERREUR] --only doit etre entre 1 et {total}.")
            sys.exit(1)
        etapes = [(args.only, PIPELINE[args.only - 1])]
    else:
        debut = max(1, args.start)
        if debut > total:
            print(f"[ERREUR] --start doit etre entre 1 et {total}.")
            sys.exit(1)
        etapes = [(i, PIPELINE[i - 1]) for i in range(debut, total + 1)]

    # ------------------------------------------------
    # Confirmation
    # ------------------------------------------------

    print()
    print("=" * 78)
    print("PRET A LANCER LE PIPELINE")
    print("=" * 78)
    print()
    print(f"Nombre d'etapes a executer : {len(etapes)} / {total}")
    print()
    print("[ATTENTION] Le pipeline complet peut prendre 20-60 minutes :")
    print("   - Telechargements eCO2mix (longs la premiere fois)")
    print("   - Entrainements XGBoost + RF (par region)")
    print()

    if not args.yes:
        try:
            reponse = input("Continuer ? [o/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAnnule.")
            return

        if reponse not in ("o", "oui", "y", "yes"):
            print("Annule.")
            return

    # ------------------------------------------------
    # Execution
    # ------------------------------------------------

    t_debut = time.time()

    for idx_absolu, (nom, script, args_script) in etapes:

        ok = executer_script(
            nom, script, args_script,
            index=idx_absolu, total=total,
        )

        if ok:
            continue

        # Echec : verifier si l'etape est non bloquante
        if script in ETAPES_NON_BLOQUANTES:
            print()
            print(f"[ATTENTION] L'etape '{nom}' a echoue mais est "
                  f"non bloquante.")
            print("On continue avec l'etape suivante.")
            continue

        print()
        print("=" * 78)
        print(f"[ECHEC] ARRET DU PIPELINE A L'ETAPE {idx_absolu}/{total}")
        print("=" * 78)
        print()
        print(f"L'etape '{nom}' a echoue.")
        print()
        print("Pour corriger :")
        print(f"  1. Corrige le script {script}")
        print(f"  2. Relance : python run_all.py --start {idx_absolu}")
        print()
        return

    duree_totale = time.time() - t_debut

    print()
    print("=" * 78)
    print("[OK] PIPELINE TERMINE AVEC SUCCES")
    print("=" * 78)
    print()
    print(f"Duree totale : {duree_totale / 60:.1f} minutes")
    print()

    if not args.no_dashboard:
        lancer_dashboard()


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

"""
run_all.py

Script de lancement automatique du pipeline complet.

Structure attendue :
    projet/
    |-- run_all.py           <- ce fichier
    |-- source/              <- contient les scripts .py
    |-- data/
    |-- models/
    +-- results/

Le script consolide automatiquement source/data, source/models
et source/results vers les dossiers correspondants a la racine,
via des liens symboliques.

Utilisation :
    python run_all.py
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "source"
SCRIPTS_DIR = SOURCE_DIR if SOURCE_DIR.exists() else BASE_DIR

DOSSIERS_A_LIER = ["data", "models", "results"]

# Pipeline : (nom_affichage, fichier_python, args_optionnels)
PIPELINE = [
    ("01 - Telechargement eCO2mix national",  "01_telechargement.py",         []),
    ("02 - Preparation nationale",            "02_preparation.py",            []),
    ("03 - Creation dataset supervise",       "03_creation_dataset.py",       []),
    ("04 - Meteo nationale",                  "04_meteo.py",                  []),
    ("05 - Modelisation nationale",           "05_modelisation.py",           []),
    ("08 - Conso regionale eCO2mix",          "08_consommation_regionale.py", []),
    ("09 - Meteo regionale",                  "09_meteo_regionale.py",        []),
    ("12 - Completion AURA",                  "12_completer_aura.py",         []),
    ("11 - Modelisation regionale",           "11_modelisation_regionale.py", []),
]

DASHBOARD_SCRIPT = "07_dashboard.py"


# ============================================================
# 2. UTILITAIRES D'AFFICHAGE
# ============================================================

def print_header(texte: str, char: str = "=", largeur: int = 78):
    print()
    print(char * largeur)
    print(texte)
    print(char * largeur)


def print_banniere():

    print()
    print("=" * 78)
    print(" " * 20 + "PREVISION DE CONSOMMATION ELECTRIQUE")
    print(" " * 25 + "Lancement complet du pipeline")
    print("=" * 78)
    print()
    print(f"Dossier racine : {BASE_DIR}")
    print(f"Dossier source : {SCRIPTS_DIR}")
    print()
    print(f"Etapes prevues ({len(PIPELINE)}) :")
    for i, (nom, _, _) in enumerate(PIPELINE, 1):
        print(f"  {i:2d}. {nom}")
    print(f"  {len(PIPELINE) + 1:2d}. Lancement du Dashboard Streamlit")
    print()


# ============================================================
# 3. GESTION DES DOSSIERS (fusion + symlink)
# ============================================================

def creer_symlink(source: Path, cible: Path) -> bool:
    """
    Cree un symlink (Unix) ou une jonction (Windows) de
    source vers cible. Suppose que source n'existe pas.
    """

    systeme = platform.system()

    try:
        if systeme == "Windows":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J",
                 str(source), str(cible)],
                check=True,
                capture_output=True,
            )
        else:
            source.symlink_to(cible, target_is_directory=True)

        return True

    except Exception as e:
        print(f"    [ERREUR] Impossible de lier {source.name} -> {cible} : {e}")
        return False


def fusionner_dossiers(source: Path, cible: Path):
    """
    Fusionne source dans cible :
        - si un element existe des deux cotes, celui de cible
          est conserve (pour ne pas ecraser les donnees brutes
          de meteofrance, par exemple)
        - sinon, on deplace l'element de source vers cible
    """

    for item in source.iterdir():

        dest = cible / item.name

        if dest.exists():
            if item.is_dir() and not item.is_symlink():
                # Fusion recursive
                dest.mkdir(parents=True, exist_ok=True)
                fusionner_dossiers(item, dest)
                # La source fille est videe, on la supprime
                try:
                    item.rmdir()
                except OSError:
                    pass
            else:
                # Conflit fichier : on garde celui de cible
                # (le fichier source est ecrase)
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception:
                    pass
        else:
            # Pas de conflit, on deplace
            try:
                shutil.move(str(item), str(dest))
            except Exception as e:
                print(f"    [ATTENTION] Deplacement {item.name} : {e}")


def consolider_dossier(dossier: str) -> bool:
    """
    Pour un dossier donne (data, models, results) :
        - Si source/<dossier> est deja un symlink, on ne fait rien
        - Si source/<dossier> n'existe pas, on cree le symlink
        - Si source/<dossier> est un vrai dossier, on fusionne
          son contenu dans <racine>/<dossier> puis on le remplace
          par un symlink
    """

    source = SCRIPTS_DIR / dossier
    cible = BASE_DIR / dossier

    # Cas 1 : deja un symlink
    if source.is_symlink():
        try:
            if source.resolve() == cible.resolve():
                print(f"  [OK] {dossier}/ -> {cible} (deja lie)")
                return True
        except Exception:
            pass
        print(f"  [ATTENTION] {dossier}/ est un symlink casse, on le remplace")
        source.unlink()

    # Cas 2 : source n'existe pas
    if not source.exists():
        cible.mkdir(parents=True, exist_ok=True)
        ok = creer_symlink(source, cible)
        if ok:
            print(f"  [OK] {dossier}/ -> {cible} (cree)")
        return ok

    # Cas 3 : source est un vrai dossier
    print(f"  [FUSION] {dossier}/ : source est un vrai dossier")

    cible.mkdir(parents=True, exist_ok=True)

    # Sauvegarde du contenu de source dans cible
    fusionner_dossiers(source, cible)

    # Supprimer le dossier source vide
    try:
        source.rmdir()
    except OSError as e:
        print(f"    [ATTENTION] Impossible de supprimer {source} : {e}")
        return False

    # Creer le symlink
    ok = creer_symlink(source, cible)
    if ok:
        print(f"  [OK] {dossier}/ -> {cible} (fusionne + lie)")

    return ok


def preparer_liens_symboliques():

    print_header("PREPARATION DES CHEMINS")

    if SCRIPTS_DIR == BASE_DIR:
        print("\n[INFO] Les scripts sont a la racine, aucun lien necessaire.")
        return True

    print(f"\nConsolidation des dossiers dans {SCRIPTS_DIR} :")
    print()

    tous_ok = True

    for dossier in DOSSIERS_A_LIER:
        ok = consolider_dossier(dossier)
        if not ok:
            tous_ok = False

    if not tous_ok:
        print()
        print("[ATTENTION] Certains dossiers n'ont pas pu etre lies.")
        print("  Tu peux essayer manuellement :")
        print(f"    1. Deplacer {BASE_DIR}/data vers {SCRIPTS_DIR}/data")
        print(f"    2. Idem pour models/ et results/")

    return tous_ok


# ============================================================
# 4. VERIFICATIONS
# ============================================================

def verifier_python():

    version = sys.version_info
    print(f"\nPython detecte : {sys.version.split()[0]}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("[ERREUR] Python 3.9+ requis.")
        sys.exit(1)

    print("[OK] Version compatible.")


def verifier_dependances():

    print("\nVerification des dependances...")

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
# 5. EXECUTION
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

    cmd = [sys.executable, str(chemin_script)] + args

    print(f"Commande : {' '.join(str(c) for c in cmd)}")
    print()

    t0 = time.time()

    try:
        resultat = subprocess.run(
            cmd,
            cwd=str(SCRIPTS_DIR),
            check=False,
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
    return True


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

    try:
        subprocess.run(cmd, cwd=str(SCRIPTS_DIR), check=False)
    except KeyboardInterrupt:
        print("\n\n[ATTENTION] Dashboard arrete.")
    except Exception as e:
        print(f"\n[ERREUR] Streamlit : {e}")
        return False

    return True


# ============================================================
# 6. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print_banniere()

    preparer_liens_symboliques()

    verifier_python()
    verifier_dependances()
    verifier_scripts()

    print()
    print("=" * 78)
    print("PRET A LANCER LE PIPELINE")
    print("=" * 78)
    print()
    print("[ATTENTION] Le pipeline complet peut prendre 20-60 minutes :")
    print("   - Telechargements eCO2mix (longs la premiere fois)")
    print("   - Entrainements XGBoost + RF (par region)")
    print()

    try:
        reponse = input("Continuer ? [o/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAnnule.")
        return

    if reponse not in ("o", "oui", "y", "yes"):
        print("Annule.")
        return

    t_debut = time.time()
    total = len(PIPELINE)

    for i, (nom, script, args) in enumerate(PIPELINE, 1):

        ok = executer_script(nom, script, args, i, total)

        if not ok:
            print()
            print("=" * 78)
            print(f"[ECHEC] ARRET DU PIPELINE A L'ETAPE {i}/{total}")
            print("=" * 78)
            print()
            print(f"L'etape '{nom}' a echoue.")
            print()
            print("Pour corriger :")
            print(f"  1. Corrige le script {script}")
            print(f"  2. Relance python run_all.py")
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

    lancer_dashboard()


if __name__ == "__main__":
    main()
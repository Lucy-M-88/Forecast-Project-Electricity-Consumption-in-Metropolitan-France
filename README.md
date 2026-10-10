# Prévision de la consommation électrique en France métropolitaine

Projet de **prévision horaire de la consommation électrique** en France métropolitaine.

L'objectif est de prévoir à **14h le jour J** les **24 consommations horaires du jour J+1**, à partir des informations disponibles à cette heure.

---

## Sommaire

- [Objectif du projet](#objectif-du-projet)
- [Résultats principaux](#résultats-principaux)
- [Méthodologie](#méthodologie)
- [Structure du projet](#structure-du-projet)
- [Sources de données](#sources-de-données)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Détail des scripts](#détail-des-scripts)
- [Dashboard interactif](#dashboard-interactif)
- [Choix méthodologiques](#choix-méthodologiques)
- [Limites et perspectives](#limites-et-perspectives)

---

## Objectif du projet

À **14h le jour J**, on souhaite prévoir la consommation électrique horaire de **l'ensemble de la journée J+1** (24 valeurs).

**Contraintes** :
- **Aucune donnée observée après 14h le jour J** ne peut être utilisée comme variable explicative.
- La météo réelle du jour J+1 n'est pas connue à 14h J (sauf via des prévisions).
- Le protocole doit être reproductible et honnête.

**Objectifs secondaires** :
- Comparer plusieurs familles de modèles.
- Évaluer les performances de manière rigoureuse (MAE, RMSE, erreur sur les pics).
- Analyser de manière critique les résultats.

---

## Résultats principaux

### Modèle final

**Ensemble Random Forest + XGBoost** (moyenne pondérée).

| Métrique | Valeur |
|---|---|
| **MAE test** | **1449 MW** |
| **RMSE test** | **1963 MW** |
| **Taux pic correct** | **64,7 %** |
| **Ratio vs RTE J-1** | **1,13x** |

### Comparaison avec les benchmarks

| Modèle | MAE (MW) | RMSE (MW) |
|---|---|---|
| **Ensemble RF + XGBoost** | **1449** | **1963** |
| XGBoost seul | 1469 | 1988 |
| Random Forest seul | 1528 | 2104 |
| Benchmark J-7 | 3333 | 4899 |
| Benchmark J-1 an | 4073 | 5604 |
| Benchmark moyenne | 3120 | 4343 |
| RTE J-1 (référence) | 1280 | 1601 |

**Gain vs benchmark J-7** : -56,5 %
**Gain vs RTE** : ratio 1,13x (RTE reste 13 % meilleur)

### Analyse par saison

| Saison | MAE (MW) |
|---|---|
| Hiver | 2171 |
| Printemps | 1398 |
| Été | 1120 |
| Automne | 1159 |

### Analyse par type de jour

| Type | MAE (MW) |
|---|---|
| Ouvré | 1475 |
| Weekend | 1372 |
| Férié | 1559 |

---

## Méthodologie

### Split chronologique retenu

Après avoir testé **6 stratégies de découpage différentes**, le split suivant a été retenu :

| Jeu | Période | Durée |
|---|---|---|
| **Train** | 2012-12-31 vers 2023-12-31 | ~11 ans |
| **Validation** | 2024-01-01 vers 2024-12-31 | 1 an |
| **Test** | 2025-01-01 vers 2026-12-31 | ~1,75 an |

**Justification** :
- Train long incluant les régimes **pré-COVID, COVID et post-COVID**.
- Validation complète (toutes les saisons).
- Test long (2 hivers, 2 étés), donnant une évaluation robuste.

### Features utilisées (19 au total)

| Catégorie | Variables |
|---|---|
| **Consommation historique** | `consommation_j7`, `consommation_j1an` |
| **Calendrier** | `heure`, `jour_semaine`, `mois`, `weekend`, `est_ferie` |
| **Températures** | `temperature_moyenne`, `temperature_min`, `temperature_max`, `amplitude_thermique` |
| **Degrés-jours** | `HDD_moyen`, `HDD_min`, `HDD_max` |
| **Moyennes glissantes** | `TM_3j`, `TM_7j`, `HDD_3j`, `HDD_7j` |
| **Froid** | `duree_gel_min` |

### Modèles comparés

1. **Benchmarks simples** :
   - Consommation J-7.
   - Consommation J-1 an.
   - Moyenne J-7 / J-1 an.

2. **Modèles d'apprentissage** :
   - XGBoost (hyperparamètres optimisés par grid search).
   - Random Forest.
   - **Ensemble** (moyenne pondérée avec alpha = 0,2).

3. **Référence externe** :
   - Prévision J-1 de RTE.

---

## Structure du projet

    projet/
    |-- run_all.py                          # Script de lancement automatique
    |-- README.md
    |
    |-- source/                             # Scripts Python
    |   |-- 01_telechargement.py
    |   |-- 02_preparation.py
    |   |-- 03_creation_dataset.py
    |   |-- 04_meteo.py
    |   |-- 05_modelisation.py
    |   |-- 06_prevision_operationnelle.py
    |   |-- 07_dashboard.py
    |   |-- 08_consommation_regionale.py
    |   |-- 09_meteo_regionale.py
    |   |-- 11_modelisation_regionale.py
    |   |-- 12_completer_aura.py
    |   +-- 13_diagnostic_regions.py
    |
    |-- data/
    |   |-- raw/
    |   |   |-- eco2mix_national_brut.csv
    |   |   |-- eco2mix_national_temps_reel.csv
    |   |   +-- meteofrance/                # Archives Meteo-France (~2,6 Go)
    |   +-- processed/
    |       |-- eco2mix_national_horaire_fusionne.csv
    |       |-- eco2mix_regional_horaire.csv
    |       |-- meteo_france_nationale_journaliere.csv
    |       |-- meteo_regionale_journaliere.csv
    |       |-- dataset_prevision.csv
    |       +-- meteofrance_reduit/
    |
    |-- models/
    |   |-- modele_xgboost_meteo.joblib
    |   |-- modele_ensemble_rf_xgb.joblib
    |   +-- regional/                       # Un modele par region
    |       |-- 11_ensemble.joblib
    |       |-- 24_ensemble.joblib
    |       +-- ...
    |
    |-- results/
    |   |-- predictions.csv
    |   |-- previsions_regionales.csv
    |   |-- evaluation_*.csv
    |   |-- pires_journees.csv
    |   |-- rapport.txt
    |   |-- regional_rapport.txt
    |   +-- prevision_operationnelle_*.{csv,png}
    |
    +-- .gitignore

---

## Sources de données

### Consommation électrique

**eCO2mix** (RTE), via l'API ODRE :
- **Consolidé et définitif** : eco2mix-national-cons-def (2012 vers quelques semaines).
- **Temps réel** : eco2mix-national-tr (dernières semaines).
- **Régional consolidé** : eco2mix-regional-cons-def (2013 vers quelques semaines).
- **Régional temps réel** : eco2mix-regional-tr.

### Météo

**Données climatologiques de base** (Meteo-France) :
- Source : meteo.data.gouv.fr/datasets/6569b51ae64326786e4e8e1a
- 22 départements retenus, pondérés par population.
- Granularité quotidienne.
- Période : 2010 vers 2026.

### Calendrier

Variables construites par le groupe :
- Jours fériés (fixes + mobiles).
- Vacances scolaires (zones A/B/C).
- Weekends.

### API externes

- **Open-Meteo Forecast** : prévisions météo J+1 (scénario opérationnel).
- **eCO2mix temps réel** : prévision J-1 de RTE (benchmark).

---

## Installation

### Prérequis

- **Python 3.10+**
- **pip**

### Environnement virtuel

    python -m venv .venv

    # Activer (Windows PowerShell)
    .\.venv\Scripts\Activate.ps1

    # Activer (Linux/macOS)
    source .venv/bin/activate

    # Dépendances
    pip install pandas numpy scikit-learn xgboost joblib requests matplotlib streamlit plotly

### Fichiers non versionnés

Les éléments suivants ne sont pas dans le dépôt (via le fichier `.gitignore`) :

- `data/raw/meteofrance/` : données météo brutes de 1950 à 2026 (~2,6 Go), téléchargeables via le site meteo.data.gouv.fr.
- `models/*.joblib` : modèles entraînés (~1,1 Go pour les modèles nationaux, ~12 Go pour les modèles régionaux), régénérables via `05_modelisation.py` et `11_modelisation_regionale.py`.

**Pour régénérer** : exécuter le script `run_all.py`.

---

## Utilisation

### Lancement automatique (recommandé)

    python run_all.py

Ce script :
1. Crée les liens symboliques nécessaires entre `source/` et les dossiers `data/`, `models/`, `results/`.
2. Vérifie les dépendances Python.
3. Exécute le pipeline complet dans le bon ordre.
4. Lance le dashboard Streamlit à la fin.

### Lancement manuel

    python source/01_telechargement.py
    python source/02_preparation.py
    python source/03_creation_dataset.py
    python source/04_meteo.py
    python source/05_modelisation.py
    python source/08_consommation_regionale.py
    python source/09_meteo_regionale.py
    python source/12_completer_aura.py
    python source/11_modelisation_regionale.py
    streamlit run source/07_dashboard.py

---

## Détail des scripts

### 1. `01_telechargement.py`

Télécharge les données eCO2mix depuis l'API ODRE :

- Dataset **données consolidées et définitives** (2012 vers quelques semaines).
- Dataset **données temps réel** (dernières semaines).

**Sorties** :
- `data/raw/eco2mix_national_brut.csv`
- `data/raw/eco2mix_national_temps_reel.csv`

### 2. `02_preparation.py`

Prépare les données au pas horaire :

- Sélection des observations à **HH:00 uniquement** (aucune moyenne, aucune interpolation).
- Reconstruction des colonnes calendaires.
- Fusion **consolidé/définitif + temps réel** avec priorité au consolidé.

**Sortie** : `data/processed/eco2mix_national_horaire_fusionne.csv`

### 3. `03_creation_dataset.py`

Crée le dataset supervisé :

- Pour chaque heure `h` de J+1 : `X = (conso J-7, conso J-1 an, calendrier)`, `y = conso J+1`.
- Aucune fuite d'information (anti-leakage vérifié).

**Sortie** : `data/processed/dataset_prevision.csv`

### 4. `04_meteo.py`

Prépare la météo nationale :

- Allège les fichiers Meteo-France (2,6 Go vers 188 Mo).
- Agrège par département puis nationalement (pondération par population).
- Crée les indicateurs dérivés (HDD, CDD, moyennes glissantes).

**Sortie** : `data/processed/meteo_france_nationale_journaliere.csv`

### 5. `05_modelisation.py`

Entraîne et évalue les modèles nationaux :

- **XGBoost** (hyperparamètres optimisés).
- **Random Forest**.
- **Ensemble pondéré** (recherche de alpha optimal sur la validation).

Produit les métriques, les prédictions, les importances et un rapport consolidé.

**Sorties** :
- `models/modele_xgboost_meteo.joblib`
- `models/modele_ensemble_rf_xgb.joblib`
- `results/predictions.csv`
- `results/evaluation_*.csv`
- `results/pires_journees.csv`
- `results/importances.csv`
- `results/rapport.txt`

### 6. `06_prevision_operationnelle.py`

Prévision opérationnelle J+1 (scénario réel) :

- Récupère la météo prévue via **Open-Meteo Forecast**.
- Construit les features pour J+1 avec uniquement les informations disponibles à 14h J.
- Fait la prédiction avec le modèle d'ensemble.
- Compare avec la prévision J-1 de RTE.

**Sortie** : `results/prevision_operationnelle_YYYY-MM-DD.{csv,png}`

### 7. `07_dashboard.py`

Dashboard interactif Streamlit :

- Onglet **Courbes nationales** : prévision du modèle, prévision RTE, consommation réelle, plus la **somme des modèles régionaux**.
- Onglet **Carte régionale** : carte choroplèthe des 12 régions métropolitaines en GW, avec slider temporel.
- Sélection d'une région : courbe prévision vs réel, MAE et biais régionaux.
- Classement des régions par **précision du modèle** (MAE en %).
- **Limité à la période de test** (post-entraînement) pour éviter toute confusion sur les performances in-sample.

### 8. `08_consommation_regionale.py`

Télécharge et prépare la consommation régionale eCO2mix :

- Sources : `eco2mix-regional-cons-def` (définitif) et `eco2mix-regional-tr` (temps réel).
- Filtre automatique sur les 12 régions métropolitaines (Corse exclue pour cause de données incomplètes).
- Conversion au pas horaire HH:00 uniquement.

**Sortie** : `data/processed/eco2mix_regional_horaire.csv`

### 9. `09_meteo_regionale.py`

Prépare la météo régionale :

- Pour chaque région, choisit un **département de référence** (liste ordonnée de préférence avec fallback automatique).
- Agrège les stations du département, puis calcule les indicateurs dérivés (HDD, moyennes glissantes).
- Utilise la même structure de données que la météo nationale.

**Sortie** : `data/processed/meteo_regionale_journaliere.csv`

### 10. `11_modelisation_regionale.py`

Entraîne un modèle (ensemble RF + XGBoost) **par région** :

- Même split temporel que le modèle national (train jusqu'à fin 2023, validation 2024, test après).
- Sauvegarde un modèle `.joblib` par région.
- Produit un rapport des performances par région (MAE, RMSE, biais).

**Sorties** :
- `models/regional/{code_region}_ensemble.joblib`
- `results/previsions_regionales.csv`
- `results/regional_rapport.txt`

### 11. `12_completer_aura.py`

Complète les données manquantes d'Auvergne-Rhône-Alpes :

- **Problème** : l'API eCO2mix régionale ne fournit pas AURA après février 2026 (~7 mois de trou).
- **Méthode** : la consommation nationale = somme des 12 régions. Donc `AURA(t) = National(t) - somme(11 autres régions)(t)`.
- **Validation** : sur les timestamps où AURA existe, on compare observé vs reconstruit pour valider l'écart médian.
- **Sécurité** : backup automatique du fichier original.

**Sortie** : `data/processed/eco2mix_regional_horaire.csv` (mis à jour)

### 12. `13_diagnostic_regions.py`

Script de diagnostic (optionnel) qui vérifie la présence de chaque région à chaque étape du pipeline :

1. `eco2mix_regional_horaire.csv` (conso régionale)
2. `meteo_regionale_journaliere.csv` (météo régionale)
3. `models/regional/*.joblib` (modèles entraînés)
4. `results/previsions_regionales.csv` (prévisions finales)

Pour chaque étape et chaque région, affiche la présence, le nombre de lignes et la plage temporelle.

---

## Dashboard interactif

**Lancement** :

    streamlit run source/07_dashboard.py

ou via `python run_all.py` (lancement automatique en fin de pipeline).

**Fonctionnalités** :

### Onglet Courbes nationales
- Choix de la plage de dates (limitée à la période de test).
- Graphique interactif (Plotly) avec 4 courbes : modèle national, somme des modèles régionaux, RTE J-1, consommation réelle.
- Métriques en direct : MAE et RMSE pour chaque source.
- Diagnostic des sources (fichier présent/absent, plage couverte).

### Onglet Carte régionale
- Carte choroplèthe des 12 régions métropolitaines.
- Échelle de couleur beige vers bordeaux (bordeaux foncé = forte consommation).
- Slider temporel pour choisir l'heure précise.
- Choix entre prévision du modèle et consommation réelle.
- **Classement des régions par précision du modèle** (MAE en %) : meilleures et moins bonnes.
- Détail d'une région : courbe temporelle, MAE, biais.
- Unité affichée : **GW** (plus lisible que MW au niveau régional).

**Principe méthodologique** : le dashboard **empêche** de visualiser des dates antérieures au test, évitant ainsi toute confusion sur les performances in-sample.

---

## Choix méthodologiques

### Anti-fuite d'information

- **Aucune donnée observée après 14h J** n'est utilisée comme variable explicative.
- Les features `consommation_j7` et `consommation_j1an` sont des **observations historiques**.
- La météo du jour J+1 est **une prévision** (scénario opérationnel) ou **observée a posteriori** (scénario à météo parfaite, non déployable).

### Split chronologique

**Aucun split aléatoire**. Séparation stricte :

    Passé -> Présent -> Futur
    Train -> Validation -> Test

### Comparaison honnête

Tous les modèles sont comparés sur le même test et avec la même information disponible.

### Étude d'ablation

Nous avons testé plusieurs groupes de features :

- **Météo** : apport majeur (+43 % de gain).
- **Fériés** : apport significatif (-3 % MAE).
- **Vacances scolaires** : apport marginal (< 0,5 %).
- **IPI** : apport négatif (+2,5 % MAE, rejeté).

### Choix du split

**6 splits** testés :

1. 70/15/15 proportionnel.
2. Split 2 (2012-2023 / 2024 / 2025-2026) : **retenu**.
3. Split 3 (2018-2025.06) : biais réduit mais pics dégradés.
4. Split 4 (2012-2024.10) : moins bon que Split 2.
5. Split E1 (validation longue) : identique à E2.
6. Split E2 (gap) : identique à E1.

### Modélisation régionale

Le modèle national a été étendu aux régions avec deux approches testées :

1. **Approche initiale** : appliquer le modèle national à des features régionales (approximation).
2. **Approche finale retenue** : entraîner un modèle par région sur les données régionales uniquement, avec la même structure de split que le national.

L'approche finale donne de meilleurs résultats et permet des comparaisons inter-régions plus justes.

---

## Limites et perspectives

### Limites identifiées

1. **Biais résiduel** : +408 MW (surestimation).
2. **Hiver difficile** : MAE 2171 MW (vs 1120 en été).
3. **Pics sous-estimés** : biais pic -85 MW.
4. **Ratio vs RTE** : 1,13x (RTE reste meilleur).
5. **Météo du jour J** : nous utilisons la météo observée à J, pas J+1.
6. **Cohérence spatiale** : le modèle national agrège (perte d'information régionale), et les modèles régionaux utilisent un seul département météo comme représentant.
7. **AURA** : données incomplètes dans l'API eCO2mix, reconstituées par différence nationale sur ~7 mois.
8. **Corse** : exclue du périmètre à cause de données eCO2mix insuffisantes.

### Perspectives

**Features supplémentaires** :
- Événements sportifs (Coupe du monde, JO).
- Événements exceptionnels (grèves).
- Prévisions météo d'ensemble.

**Modèles plus avancés** :
- LightGBM.
- Modèles par heure ou par saison.
- Réseaux de neurones (LSTM, TFT).

**Post-traitement** :
- Correction du biais (risque de détériorer grandement les performances sur les pics).
- Calibration des pics.
- Ensemble pondéré par le contexte.

**Pondération spatiale** :
- Utiliser plusieurs départements par région (au lieu d'un seul).
- Modèle hiérarchique national + régional.

---

## Références

- **eCO2mix** - RTE. https://odre.opendatasoft.com/
- **Données climatologiques de base** - Meteo-France. https://meteo.data.gouv.fr/
- **Open-Meteo Forecast API**. https://open-meteo.com/
- **Indice de production industrielle** (abandonné) - INSEE. https://www.insee.fr/

---

## Auteurs

Projet réalisé dans le cadre du cours **Temporal Data Analysis** (Master 2).

- **Auteurs** : Lucy MARTIN et Clément CORREIA--PELTIER

---

## Licence

Projet académique. Utilisation libre à des fins pédagogiques.

# Prévision de la consommation électrique en France métropolitaine

Projet de **prévision horaire de la consommation électrique** en France métropolitaine.

L'objectif est de prévoir à **14h le jour J** les **24 consommations horaires du jour J+1**, à partir des informations disponibles à cette heure.

---

## 📋 Sommaire

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
| **Ratio vs RTE J-1** | **1,13×** |

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
**Gain vs RTE** : ratio 1,13× (RTE reste 13 % meilleur)

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

##  Méthodologie

### Split chronologique retenu

Après avoir testé **6 stratégies de découpage différentes**, le split suivant a été retenu :

| Jeu | Période | Durée |
|---|---|---|
| **Train** | 2012-12-31 → 2023-12-31 | ~11 ans |
| **Validation** | 2024-01-01 → 2024-12-31 | 1 an |
| **Test** | 2025-01-01 → 2026-12-31 | ~1,75 an |

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
   - **Ensemble** (moyenne pondérée avec α = 0,2).

3. **Référence externe** :
   - Prévision J-1 de RTE.

---

## Structure du projet
Code/
├── 01_telechargement.py # Téléchargement éCO2mix (consolidé + temps réel)
├── 02_preparation.py # Préparation au pas horaire + fusion
├── 03_creation_dataset.py # Création du dataset supervisé (features + cible)
├── 04_meteo.py # Préparation de la météo nationale
├── 05_modelisation.py # Entraînement + évaluation
├── 06_prevision_operationnelle.py # Prévision J+1 (scénario opérationnel)
├── 07_dashboard.py # Dashboard interactif Streamlit
│
├── data/
│ ├── raw/ # Données brutes (non versionnées)
│ │ ├── eco2mix_national_brut.csv
│ │ ├── eco2mix_national_temps_reel.csv
│ │ └── meteofrance/ # Archives Météo-France (~2.6 Go)
│ └── processed/ # Données préparées
│ ├── eco2mix_national_horaire_fusionne.csv
│ ├── meteo_france_nationale_journaliere.csv
│ ├── dataset_prevision.csv
│ └── meteofrance_reduit/ # Météo-France allégée (par département)
│
├── models/ # Modèles entraînés (non versionnés)
│ ├── modele_xgboost_meteo.joblib
│ └── modele_ensemble_rf_xgb.joblib
│
├── results/ # Résultats produits
│ ├── predictions.csv
│ ├── evaluation_.csv
│ ├── pires_journees.csv
│ ├── rapport.txt
│ └── prevision_operationnelle_.{csv,png}
│
├── archive/ # Scripts historiques (traçabilité)
│
├── .gitignore
└── README.md


---

## Sources de données

### Consommation électrique

**éCO2mix** (RTE), via l'API ODRÉ :
- **Consolidé et définitif** : [eco2mix-national-cons-def](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def/) (2012 → quelques semaines).
- **Temps réel** : [eco2mix-national-tr](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr/) (dernières semaines).

### Météo

**Données climatologiques de base** (Météo-France) :
- Source : [https://meteo.data.gouv.fr/datasets/6569b51ae64326786e4e8e1a](https://meteo.data.gouv.fr/datasets/6569b51ae64326786e4e8e1a)
- 22 départements retenus, pondérés par population.
- Granularité quotidienne.
- Période : 2010 → 2026.

### Calendrier

Variables construites par le groupe :
- Jours fériés (fixes + mobiles).
- Vacances scolaires (zones A/B/C).
- Weekends.

### API externes

- **Open-Meteo Forecast** : prévisions météo J+1 (scénario opérationnel).
- **éCO2mix temps réel** : prévision J-1 de RTE (benchmark).

---

## Installation

### Prérequis

- **Python 3.10+**
- **pip**

### Environnement virtuel

```bash
# Créer l'environnement
python -m venv .venv

# Activer (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activer (Linux/macOS)
source .venv/bin/activate

# Dépendances
pip install pandas numpy scikit-learn xgboost joblib requests matplotlib streamlit plotly


### Fichiers non versionnés

Les éléments suivants ne sont pas dans le dépôt (via le fichier `.gitignore`) :

- `data/raw/` : données météo brutes de 1950 à 2026 (~2,6 Go), téléchargeables via le site meteo.data.gouv.fr.
- `models/*.joblib` : modèles entraînés (~1,1 Go), régénérables via `05_modelisation.py`.

**Pour régénérer** : exécuter les scripts dans l'ordre ou le fichier run_all.py.

---

## Détail des scripts

### 1. `01_telechargement.py`

Télécharge les données éCO2mix depuis l'API ODRÉ :

- Dataset **données consolidées et définitives** (2012 → quelques semaines).
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
- Aucune fuite d'information.

**Sortie** : `data/processed/dataset_prevision.csv`

### 4. `04_meteo.py`

Prépare la météo nationale :

- Allège les fichiers Météo-France (2,6 Go → 188 Mo, sauf si dossier data/processed/meteofrance_reduit déjà rempli).
- Agrège par département puis nationalement (pondération par population).
- Crée les indicateurs dérivés (HDD, CDD, moyennes glissantes).

**Sortie** : `data/processed/meteo_france_nationale_journaliere.csv`

### 5. `05_modelisation.py`

Entraîne et évalue :

- **XGBoost** (hyperparamètres optimisés).
- **Random Forest**.
- **Ensemble pondéré**.

Produit les métriques, les prédictions, les importances et un rapport consolidé.

### 6. `06_prevision_operationnelle.py`

Prévision opérationnelle J+1 :

- Récupère la météo prévue via **Open-Meteo Forecast**.
- Construit les features pour J+1.
- Fait la prédiction avec le modèle d'ensemble.
- Compare avec la prévision J-1 de RTE.
- Produit un graphique et un fichier csv avec la comparaison

**Sortie** : `results/prevision_operationnelle_YYYY-MM-DD.{csv,png}`

### 7. `07_dashboard.py`

Dashboard interactif Streamlit :

- Affiche la prévision du modèle, la prévision RTE et la consommation réelle.
- **Ne couvre que la période de test** (post-entraînement), pour éviter toute confusion.
- Permet de choisir la plage de dates et de calculer les métriques en direct.

---

## Dashboard interactif

**Lancement** :

```bash
streamlit run 07_dashboard.py

**Fonctionnalités** :

- Choix de la plage de dates (limitée à la période de test).
- Graphique interactif (Plotly) avec 3 courbes.
- Métriques en direct (MAE, RMSE).
- Diagnostic des sources.

**Principe méthodologique** : le dashboard **empêche** de visualiser des dates antérieures au test, évitant ainsi toute confusion sur les performances in-sample.

---

## Choix méthodologiques

### Anti-fuite d'information

- **Aucune donnée observée après 14h J** n'est utilisée comme variable explicative.
- Les features `consommation_j7` et `consommation_j1an` sont des **observations historiques**.
- La météo du jour J+1 est **une prévision** (scénario opérationnel) ou **observée a posteriori** (scénario à météo parfaite, non déployable).

### Split chronologique

**Aucun split aléatoire**. Séparation stricte :
Passé → Présent → Futur
Train → Validation → Test


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

---

## Limites et perspectives

### Limites identifiées

1. **Biais résiduel** : +408 MW (surestimation).
2. **Hiver difficile** : MAE 2171 MW (vs 1120 en été).
3. **Pics sous-estimés** : biais pic -85 MW.
4. **Ratio vs RTE** : 1,13× (RTE reste meilleur).
5. **Météo du jour J** : nous utilisons la météo observée à J, pas J+1.
6. **Cohérence spatiale** : agrégation nationale (perte d'information régionale).

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

- Utiliser des données régionales plutôt que nationales.

---

## Références

- **éCO2mix** — RTE. [https://odre.opendatasoft.com/](https://odre.opendatasoft.com/)
- **Données climatologiques de base** — Météo-France. [https://meteo.data.gouv.fr/](https://meteo.data.gouv.fr/)
- **Open-Meteo Forecast API**. [https://open-meteo.com/](https://open-meteo.com/)
- **Indice de production industrielle** (abandonné) — INSEE. [https://www.insee.fr/](https://www.insee.fr/)

---

## Auteurs

Projet réalisé dans le cadre du cours **Temporal Data Analysis** (Master 2).

- **Auteurs** : Lucy MARTIN et Clément CORREIA--PELTIER

---

## Licence

Projet académique. Utilisation libre à des fins pédagogiques.
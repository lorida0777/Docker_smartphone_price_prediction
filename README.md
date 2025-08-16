# 📱Prédiction de Prix de Smartphones

Ce projet permet de prédire le **prix d’un smartphone** à partir de ses caractéristiques techniques (RAM, stockage, caméra, processeur, etc.) en se basant sur un modèle de machine learning entraîné sur un large dataset.

---

## 🚀 Fonctionnalités Clés

- 📊 Prédiction du prix du smartphone en dollars ($).
- 🔧 Ajustement automatique pour éviter les prédictions incohérentes.
- 🎯 Comparaison avec le **prix moyen de téléphones similaires** (même RAM et stockage).
- 📈 Visualisations dynamiques (barres, radar).
- 🧠 Prise en compte d'indicateurs dérivés (prix par Go, ratio batterie/écran...).
- ✅ Interface claire avec messages dynamiques (succès, alerte, info).

---

## 🖼️ Captures d'écran de l'application

| Formulaire de saisie     | Résultat estimé          |
| ------------------------ | ------------------------ |
| ![](screenshoot/SC1.png) | ![](screenshoot/SC2.png) |

| Visualisations           | Informations complémentaires |
| ------------------------ | ---------------------------- |
| ![](screenshoot/SC3.png) | ![](screenshoot/SC4.png)     |

## 🧪 Caractéristiques utilisées pour la prédiction

- Marque (`Brand`)
- Processeur (`Processor`)
- Batterie (`Battery capacity (mAh)`)
- Taille écran (`Screen size (inches)`)
- RAM (`RAM (GB)`)
- Stockage (`Internal storage (GB)`)
- Caméra arrière (`Rear camera`)
- Caméra avant (`Front camera`)

Et plusieurs **caractéristiques dérivées** (créées automatiquement lors du prétraitement).

---

## 🧠 Modèle de Machine Learning

- Le modèle sélectionné correspond **au meilleur des trois testés** pour ce cas de prédiction (Linear Regression, Random Forest Regressor, Gradient Boosting Regressor).

- Entraînement réalisé sur le dataset [![© Kaggle Dataset](https://img.shields.io/badge/©%20Kaggle%20Dataset-grey?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/pratikgarai/mobile-phone-specifications-and-prices) : `ndtv_data_final.csv`. 

- Prétraitement incluant : encodage, scaling, et feature engineering.

---
## 📂 Arborescence du Projet

```
├── models/                   # Modèles après entraînement (pkl)
├── screenshoot/              # Captures d’écran de l’application
├── app_streamlit.py          # Application Streamlit
├── Dockerfile                # Déploiement de l'application avec Docker
├── ndtv_data_final.csv       # Dataset utilisé pour le réentraînement
├── README.md                 # Documentation du projet
├── requirements.txt          # Bibliothèques Python nécessaires
├── retrain_model.py          # Pour réentraîner et mettre à jour le modèle
```

---

## 🔁 Réentraînement du Modèle

Le script `retrain_model.py` permet de :

- Recharger les données
- Refabriquer toutes les variables dérivées
- Réentraîner le modèle et les encodeurs
- Sauvegarder dans le dossier `models/`

---

## 🌐 Lancement de l'application

### 💻 Option 1 — via Docker

1. Construire l’image Docker :

   ```bash
   docker build -t phone-price-app .
   ```

2. Lancer l’application :

   ```bash
   docker run -p 8501:8501 phone-price-app
   ```

3. Ouvrir dans un navigateur :
   [http://localhost:8501](http://localhost:8501)



### 💻 Option 2 — version hébergée (facultatif)

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartphone-prediction.streamlit.app)

---


## 🔍 Interprétation des indicateurs

- **Prix par Go** : Coût moyen du stockage, permet d’évaluer la compétitivité d’un téléphone selon sa capacité.
- **Prix par MP** : Coût estimé de la qualité photo (caméras), pour comparer la performance visuelle à prix égal.
- Ces indicateurs facilitent la **comparaison marché / modèle prédit**.

---

## 🙋‍♂️ Auteurs

Projet réalisé (Août 2025) dans un cadre pédagogique de l'**INSI** par :
- [ANDRIATSIFERANA No Kanto Lorida](mailto:kantonotsiferana@gmail.com)
- [ANDRIANTSALAMA Rijamampianina](mailto:rijamampianina@gmail.com)

---

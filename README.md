<<<<<<< HEAD
# Docker_smartphone_price_prediction
#First commit
=======
# 📱 Application de Prédiction de Prix de Smartphones

Cette application de Machine Learning prédit le **prix d’un smartphone (en roupies indiennes ₹, avec conversion USD)** à partir de ses caractéristiques techniques.  
Elle propose une interface web simple via **Streamlit**, conteneurisée avec **Docker**, permettant une utilisation sans installation locale de Python.

---

## 🔧 Fonctionnalités principales

- Formulaire interactif de saisie.
- Prédiction du prix du smartphone (₹ et USD).
- Affichage du **prix moyen de téléphones similaires**.
- Visualisations interactives (barres & radar) avec Plotly.
- Indicateurs personnalisés : **prix par Go**, **prix par MP**.
- Interface utilisateur moderne, pédagogique et réactive.
- Déploiement Dockerisé pour portabilité maximale.

---

## 📂 Structure du projet

```
.
├── app_streamlit.py             # Application Streamlit
├── Dockerfile                   # Image Docker
├── requirements.txt             # Dépendances Python
├── models/                      # Modèles & encodages
│   ├── phone_price_model.pkl
│   ├── scaler.pkl
│   ├── brand_encoder.pkl
│   ├── processor_encoder.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
├── screenshoot/                 # Captures d’écran UI
├── ndtv_data_final.csv          # Dataset d'entraînement
└── retrain_model.py             # Script de re-entraînement
```

---

## 🧠 Algorithmes & Modèles

- Modèle supervisé de régression : RandomForestRegressor ou autre, selon les entraînements.
- Sélection dynamique du **meilleur modèle** selon RMSE, MAE...
- Possibilité de re-train via `retrain_model.py`.

---

## 🧪 Exécution locale (hors Docker)

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

---

## 🐳 Exécution avec Docker

1. Construction de l’image :
```bash
docker build -t phone-price-app .
```

2. Lancement du conteneur :
```bash
docker run -p 8501:8501 phone-price-app
```

Accès via navigateur : [http://localhost:8501](http://localhost:8501)

---

## 🌐 Démo en ligne (facultatif)

[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-deployment-url.streamlit.app)

---

## 📊 Interface Utilisateur

L'application présente **3 onglets** :
1. **📱 Prix estimé**  
   → Affichage du **prix prédit**, du **prix moyen similaire**, et un **message dynamique** selon le positionnement du prix.
2. **📈 Visualisations**  
   → Diagramme à barres et graphique radar interactifs avec **Plotly**.
3. **ℹ️ Informations Complémentaires**  
   → Détails des caractéristiques saisies + indicateurs :
   - Prix par Go (₹/GB, $/GB)
   - Prix par MP (₹/MP, $/MP)

---

## 🖼️ Captures d'écran de l'application

| Formulaire de saisie     | Résultat estimé          |
| ------------------------ | ------------------------ |
| ![](screenshoot/SC1.png) | ![](screenshoot/SC2.png) |

| Visualisations           | Informations complémentaires |
| ------------------------ | ---------------------------- |
| ![](screenshoot/SC3.png) | ![](screenshoot/SC4.png)     |

---

## 🔍 Interprétation des indicateurs

- **Prix par Go (₹/GB)** : Coût moyen du stockage, permet d’évaluer la compétitivité d’un téléphone selon sa capacité.
- **Prix par MP (₹/MP)** : Coût estimé de la qualité photo (caméras), pour comparer la performance visuelle à prix égal.
- Ces indicateurs facilitent la **comparaison marché / modèle prédit**.

---

## 📌 Améliorations prévues

- Export PDF ou Excel des résultats.
- Authentification utilisateur (version cloud).

---

## 👨‍💻 Auteur

Projet académique de Data Science avec Streamlit, Scikit-learn, Plotly & Docker.  
- Kanto...
- Rijamampianina...

---
>>>>>>> 9ece4da (Initial commit)

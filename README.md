<<<<<<< HEAD
<<<<<<< HEAD
# Docker_smartphone_price_prediction
#First commit
=======
# 📱 Application de Prédiction de Prix de Smartphones
=======
>>>>>>> 49e8c39 (version 2 optimisation)

# 📦 Projet de Prédiction du Prix des Smartphones

Ce projet inclut :
- ✅ Entraînement d’un modèle de régression sur le **prix log-transformé**
- ✅ Comparaison de 3 modèles (Random Forest, Gradient Boosting, Extra Trees)
- ✅ Tuning automatique du meilleur modèle
- ✅ Application Streamlit en USD uniquement, avec visualisation dynamique

## 🧪 Fichiers inclus

- `retrain_model_clean_full_models.py` : script complet d'entraînement + sélection automatique
- `app_streamlit_usd_visuals.py` : application Streamlit (prévision + visualisations)
- `ndtv_data_final.csv` : jeu de données original

## 💵 Conversion
Tous les prix sont en **USD** avec la conversion : `1 USD = 86.14 ₹`.

## 📈 Visualisation
Un graphique `Prix réel vs Prix prédit` est généré automatiquement dans l’onglet 2.

## ▶️ Lancement
```bash
streamlit run app_streamlit_usd_visuals.py
```

Entraînez d'abord le modèle via :
```bash
python retrain_model_clean_full_models.py
```

<<<<<<< HEAD
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
=======
>>>>>>> 49e8c39 (version 2 optimisation)

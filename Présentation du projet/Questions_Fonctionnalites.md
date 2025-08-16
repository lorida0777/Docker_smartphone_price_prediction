# 📱 Projet — Prédiction de Prix de Smartphones (Q&A + Fonctionnalités)

## I. Questions fréquentes — Réponses précises

**1) Quel est l’objectif exact de l’application ?**  
Prédire en **$** le prix d’un smartphone à partir de ses caractéristiques (marque, processeur, RAM, stockage, caméras, écran), puis **comparer** ce prix au **prix moyen d’appareils similaires** (±20% sur RAM et stockage).

**2) D’où proviennent les données ?**  
D’un dataset Kaggle listant spécifications et prix. Les données sont nettoyées, normalisées et enrichies de caractéristiques dérivées.

**3) Quel modèle est utilisé ?**  
Trois familles sont comparées (**Random Forest**, **Gradient Boosting**, **Extra Trees**) avec **cross‑validation** ; on retient le meilleur, puis on fait un **GridSearchCV** pour régler les hyperparamètres.

**4) Pourquoi la cible est en log (LogPrice) ?**  
Le log stabilise la variance et réduit l’influence des valeurs extrêmes, ce qui améliore souvent la qualité de la régression. La prédiction est reconvertie ensuite en prix réel.

**5) L’« ajustement du prix » est‑il une tricherie ?**  
Non dans une logique produit : c’est un **garde‑fou pragmatique** qui mélange la prédiction du modèle et la moyenne des similaires quand l’écart est grand, afin d’éviter des valeurs aberrantes. Pour l’évaluation pure du modèle, on affiche idéalement **prix brut** et **prix ajusté**.

**6) Comment est calculée la moyenne des téléphones similaires ?**  
On filtre le dataset sur les téléphones ayant **RAM** et **stockage** dans **±20%** des valeurs saisies, puis on prend la **moyenne du prix** de ce sous‑ensemble.

**7) Quand l’ajustement s’applique‑t‑il et avec quels poids ?**  
Si au moins **5 similaires** existent :  
- Écart relatif ≤ 15% → **poids modèle = 0.8**  
- 15% < écart ≤ 30% → **poids modèle = 0.5**  
- Écart > 30% → **poids modèle = 0.3**  
Le prix ajusté = `w * prix_modèle + (1 - w) * prix_moyen`, plafonné à **+10%** au‑dessus du prix moyen.

**8) Que signifie le badge ±% affiché ?**  
Il montre l’**écart relatif** entre le **prix ajusté** et la **moyenne des similaires**. Couleur verte si inférieur, rouge si supérieur, gris si nul.

**9) Comment lire le graphique en barres ?**  
Comparaison visuelle directe **Prix prédit** vs **Prix moyen similaire** en $. Utile pour juger rapidement si l’estimation est sous/sur‑évaluée.

**10) Comment interpréter le graphique radar ?**  
Chaque axe représente une caractéristique **normalisée** dans [0, 1] :  
- Batterie (mAh/5000), Écran (pouces/7), RAM (GB/8), Stockage (GB/256), Caméra arrière (MP/64), Caméra avant (MP/32).  
La **courbe bleue** = téléphone saisi, la **courbe orange** = moyenne du dataset.  
- Bleu > Orange sur un axe → **au‑dessus de la moyenne** sur cette caractéristique.  
- Bleu < Orange → **en‑dessous de la moyenne**.  
Cela révèle rapidement les **forces/faiblesses** et l’équilibre du profil (ex. grosse batterie mais RAM faible).

**11) Quelles sont les limites ?**  
La qualité dépend du **dataset** et de la **pertinence des features**. L’ajustement masque parfois des faiblesses du modèle : il faut continuer à améliorer données et features.

**12) Comment déployer ?**  
Via **Docker** : `docker build -t phone-price-app .` puis `docker run -p 8501:8501 phone-price-app`.


## II. Fonctionnalités de l’application

- Saisie interactive des caractéristiques (sidebar).  
- **Prédiction** du prix (modèle ML) + conversion en $.  
- **Comparaison** avec **prix moyen** d’appareils similaires (±20% RAM/stockage).  
- **Ajustement automatique** du prix si écart élevé, avec règles de pondération.  
- **Graphiques** : barres (prix vs moyenne) et **radar** (profil de caractéristiques).  
- **Indicateurs dérivés** : prix/Go, prix/MP, ratios batterie/écran, etc.  
- **Alertes** et messages contextuels (au‑dessus, dans la moyenne, en dessous).  
- **Transparence recommandée** : afficher **prix brut** et **ajusté** côté à côte.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Chargement des données
print("Chargement du dataset...")
df = pd.read_csv('ndtv_data_final.csv')

# 2. Création des features avancées (doit correspondre à app_streamlit.py)
df['Camera_Total'] = df['Rear camera'] + df['Front camera']
df['RAM_GB'] = df['RAM (MB)'] / 1000
df['Price_per_GB'] = df['Price'] / df['Internal storage (GB)']
df['Price_per_MP'] = df['Price'] / (df['Rear camera'] + df['Front camera'])
df['Screen_to_Battery_Ratio'] = df['Screen size (inches)'] / (df['Battery capacity (mAh)'] / 1000)
df['Price_per_RAM'] = df['Price'] / df['RAM_GB']
df['Battery_to_Screen_Ratio'] = df['Battery capacity (mAh)'] / df['Screen size (inches)']

# 3. Sélection des features (doit correspondre à feature_names.pkl)
feature_names = [
    'Brand', 'Battery capacity (mAh)', 'Screen size (inches)', 'Processor',
    'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera',
    'Camera_Total', 'RAM_GB', 'Price_per_GB', 'Price_per_MP',
    'Screen_to_Battery_Ratio', 'Price_per_RAM', 'Battery_to_Screen_Ratio'
]
X = df[feature_names]
y = df['Price']

# 4. Encodage des variables catégorielles
brand_encoder = LabelEncoder()
processor_encoder = LabelEncoder()
X['Brand'] = brand_encoder.fit_transform(X['Brand'])
X['Processor'] = processor_encoder.fit_transform(X['Processor'])

# 5. Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Entraînement du modèle
print("Entraînement du modèle RandomForest...")
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Évaluation
print("Évaluation sur le jeu de test...")
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f} | R2: {r2:.3f}")

# 9. Cross-validation (optionnel)
scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
print("MAE moyen (cross-val):", -scores.mean())

# 10. Sauvegarde des objets pour l'app
joblib.dump(model, 'phone_price_model.pkl')
joblib.dump(brand_encoder, 'brand_encoder.pkl')
joblib.dump(processor_encoder, 'processor_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
print("Modèle et encodeurs sauvegardés.") 
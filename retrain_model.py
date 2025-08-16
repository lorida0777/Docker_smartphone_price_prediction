
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

print("🔧 Entraînement du modèle (multi-modèles + tuning)")
print("=" * 60)

# Chargement
df = pd.read_csv('ndtv_data_final.csv').dropna()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Suppression des extrêmes
q_low, q_high = df['Price'].quantile([0.01, 0.99])
df = df[(df['Price'] >= q_low) & (df['Price'] <= q_high)]

# Feature engineering
df['Camera_Total'] = df['Rear camera'] + df['Front camera']
df['RAM_GB'] = df['RAM (MB)'] / 1000

df['Price_per_GB'] = np.where(df['Internal storage (GB)'] > 0,
                              df['Price'] / df['Internal storage (GB)'],
                              df['Price'].mean())

df['Price_per_MP'] = np.where(df['Camera_Total'] > 0,
                              df['Price'] / df['Camera_Total'],
                              df['Price'].mean())

df['Screen_to_Battery_Ratio'] = np.where(df['Battery capacity (mAh)'] > 0,
                                         df['Screen size (inches)'] / (df['Battery capacity (mAh)'] / 1000),
                                         1.0)

df['Price_per_RAM'] = np.where(df['RAM (MB)'] > 0,
                               df['Price'] / df['RAM (MB)'],
                               df['Price'].mean())

df['Battery_to_Screen_Ratio'] = np.where(df['Screen size (inches)'] > 0,
                                         df['Battery capacity (mAh)'] / df['Screen size (inches)'],
                                         3000)

df['Is_Premium'] = df['Brand'].isin(['Apple', 'Samsung', 'OnePlus']).astype(int)

df['LogPrice'] = np.log1p(df['Price'])

features = ['Brand', 'Battery capacity (mAh)', 'Screen size (inches)', 'Processor',
            'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera',
            'Price_per_GB', 'Price_per_MP', 'Screen_to_Battery_Ratio', 'Camera_Total',
            'RAM_GB', 'Price_per_RAM', 'Battery_to_Screen_Ratio', 'Is_Premium']

X = df[features]
y = df['LogPrice']

# Nettoyage valeurs extrêmes
numeric_columns = X.select_dtypes(include=[np.number]).columns
X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

# Encodage
brand_encoder = LabelEncoder()
processor_encoder = LabelEncoder()
X.loc[:, 'Brand'] = brand_encoder.fit_transform(X['Brand'])
X.loc[:, 'Processor'] = processor_encoder.fit_transform(X['Processor'])

# Normalisation
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# Modèles à comparer
print("\n🤖 Comparaison de 3 modèles")
models = {
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(random_state=42, n_jobs=-1)
}

best_model = None
best_name = ""
best_score = -np.inf

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    mean_score = scores.mean()
    print(f"{name} : R² = {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_name = name

print(f"🏆 Meilleur modèle : {best_name}")

# Tuning du meilleur modèle
print(f"🔧 Tuning des hyperparamètres pour {best_name}...")

if best_name == "Random Forest":
    param_grid = {
        'n_estimators': [500, 700],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
elif best_name == "Gradient Boosting":
    param_grid = {
        'n_estimators': [500, 700],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
elif best_name == "Extra Trees":
    param_grid = {
        'n_estimators': [500, 700],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5]
    }
else:
    param_grid = {}

grid = GridSearchCV(best_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print(f"✅ Meilleurs paramètres : {grid.best_params_}")

# Évaluation
print("\n📈 Évaluation finale")
y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

r2 = r2_score(y_test_real, y_pred)
mae = mean_absolute_error(y_test_real, y_pred)
mape = mean_absolute_percentage_error(y_test_real, y_pred)

print(f"R² : {r2:.4f}")
print(f"MAE : {mae:.2f}")
print(f"MAPE : {mape * 100:.2f}%")

# Sauvegarde
print("\n💾 Sauvegarde du modèle et des objets...")
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/phone_price_model.pkl")
joblib.dump(brand_encoder, "models/brand_encoder.pkl")
joblib.dump(processor_encoder, "models/processor_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(features, "models/feature_names.pkl")

print("✅ Terminé avec succès !")

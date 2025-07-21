import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🔧 Retraining du modèle de prédiction de prix de téléphones")
print("=" * 60)

# Loading the dataset
print("📊 Chargement des données...")
df = pd.read_csv('ndtv_data_final.csv')
print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Data cleaning and preprocessing
print("\n🧹 Nettoyage des données...")
df = df.dropna()
print(f"Après nettoyage: {df.shape[0]} lignes")

# Remove outliers and infinite values
print("🔍 Suppression des valeurs aberrantes...")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
print(f"Après suppression des valeurs infinies: {df.shape[0]} lignes")

# Feature engineering with error handling
print("\n⚙️ Création de nouvelles fonctionnalités...")
try:
    # Safe division to avoid division by zero
    df['Price_per_GB'] = np.where(df['Internal storage (GB)'] > 0, 
                                 df['Price'] / df['Internal storage (GB)'], 
                                 df['Price'].mean())
    
    df['Price_per_MP'] = np.where((df['Rear camera'] + df['Front camera']) > 0,
                                 df['Price'] / (df['Rear camera'] + df['Front camera']),
                                 df['Price'].mean())
    
    df['Screen_to_Battery_Ratio'] = np.where(df['Battery capacity (mAh)'] > 0,
                                            df['Screen size (inches)'] / (df['Battery capacity (mAh)'] / 1000),
                                            1.0)
    
    df['Camera_Total'] = df['Rear camera'] + df['Front camera']
    df['RAM_GB'] = df['RAM (MB)'] / 1000
    
    # Additional features
    df['Price_per_RAM'] = np.where(df['RAM (MB)'] > 0,
                                  df['Price'] / df['RAM (MB)'],
                                  df['Price'].mean())
    
    df['Battery_to_Screen_Ratio'] = np.where(df['Screen size (inches)'] > 0,
                                            df['Battery capacity (mAh)'] / df['Screen size (inches)'],
                                            3000)
    
except Exception as e:
    print(f"Erreur lors de la création des fonctionnalités: {e}")
    # Fallback to basic features only
    df['Camera_Total'] = df['Rear camera'] + df['Front camera']
    df['RAM_GB'] = df['RAM (MB)'] / 1000
    df['Price_per_GB'] = df['Price'].mean()
    df['Price_per_MP'] = df['Price'].mean()
    df['Screen_to_Battery_Ratio'] = 1.0
    df['Price_per_RAM'] = df['Price'].mean()
    df['Battery_to_Screen_Ratio'] = 3000

# Select features including engineered ones
features = ['Brand', 'Battery capacity (mAh)', 'Screen size (inches)', 'Processor', 
            'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera',
            'Price_per_GB', 'Price_per_MP', 'Screen_to_Battery_Ratio', 'Camera_Total', 'RAM_GB',
            'Price_per_RAM', 'Battery_to_Screen_Ratio']

X = df[features]
y = df['Price']

# Final cleaning - only for numeric columns
numeric_columns = X.select_dtypes(include=[np.number]).columns
X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

print(f"Fonctionnalités utilisées: {len(features)}")

# Encoding categorical variables
print("\n🔤 Encodage des variables catégorielles...")
brand_encoder = LabelEncoder()
processor_encoder = LabelEncoder()
X['Brand'] = brand_encoder.fit_transform(X['Brand'])
X['Processor'] = processor_encoder.fit_transform(X['Processor'])

# Scaling features
print("📏 Normalisation des fonctionnalités...")
scaler = RobustScaler()  # More robust to outliers
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)
print(f"Ensemble d'entraînement: {X_train.shape[0]} échantillons")
print(f"Ensemble de test: {X_test.shape[0]} échantillons")

# Model comparison
print("\n🤖 Comparaison des modèles...")
models = {
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(random_state=42, n_jobs=-1)
}

best_score = 0
best_model_name = ""
best_model = None

for name, model in models.items():
    try:
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        avg_score = cv_scores.mean()
        print(f"{name}: CV R² = {avg_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        if avg_score > best_score:
            best_score = avg_score
            best_model_name = name
            best_model = model
    except Exception as e:
        print(f"{name}: Erreur lors de l'évaluation - {e}")

print(f"\n🏆 Meilleur modèle: {best_model_name} (CV R² = {best_score:.4f})")

# Hyperparameter tuning for the best model
print(f"\n🔧 Optimisation des hyperparamètres pour {best_model_name}...")

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [500, 700, 1000],
        'max_depth': [20, 25, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [500, 700, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
elif best_model_name == 'Extra Trees':
    param_grid = {
        'n_estimators': [500, 700, 1000],
        'max_depth': [20, 25, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
else:
    param_grid = {}

if param_grid:
    try:
        grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Meilleurs paramètres: {grid_search.best_params_}")
        print(f"Score CV optimisé: {grid_search.best_score_:.4f}")
    except Exception as e:
        print(f"Erreur lors de l'optimisation: {e}")

# Training the final model
print(f"\n🎯 Entraînement du modèle final...")
best_model.fit(X_train, y_train)

# Evaluation
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📈 Résultats finaux:")
print(f"Score R² d'entraînement: {train_score:.4f}")
print(f"Score R² de test: {test_score:.4f}")
print(f"Erreur quadratique moyenne (MSE): {mse:.2f}")
print(f"Erreur absolue moyenne (MAE): {mae:.2f}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.Series(best_model.feature_importances_, index=features)
    print(f"\n🎯 Importance des fonctionnalités:")
    print(feature_importance.sort_values(ascending=False).head(10))

# Saving the model and encoders
print(f"\n💾 Sauvegarde du modèle...")
joblib.dump(best_model, 'phone_price_model.pkl')
joblib.dump(brand_encoder, 'brand_encoder.pkl')
joblib.dump(processor_encoder, 'processor_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature names for the app
joblib.dump(features, 'feature_names.pkl')

# Also save to models directory
joblib.dump(best_model, 'models/price_predictor_model.pkl')

print(f"\n✅ Modèle sauvegardé avec succès!")
print(f"🎯 Précision cible (>90%): {'✅ ATTEINTE' if test_score > 0.90 else '❌ NON ATTEINTE'}")
print(f"📊 Précision actuelle: {test_score:.2%}")

if test_score < 0.90:
    print(f"\n💡 Suggestions pour améliorer:")
    print("- Collecter plus de données")
    print("- Ajouter de nouvelles fonctionnalités")
    print("- Essayer d'autres algorithmes (XGBoost, LightGBM)")
    print("- Ajuster les hyperparamètres")
else:
    print(f"\n🎉 Excellent! Le modèle atteint la précision cible!") 
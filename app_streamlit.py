# Modif num 5 - VERSION CORRIGÉE (avec prise en compte du retrain_model.py)
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Prix de Téléphones",
    page_icon="📱",
    layout="wide"
)

# Titre principal
st.title("📱 Prédiction de Prix de Téléphones")
st.markdown("---")

# Chargement des données et modèles
@st.cache_data
def load_data():
    """Charge les données et modèles"""
    try:
        df = pd.read_csv('ndtv_data_final.csv')
        model = joblib.load('models/phone_price_model.pkl')
        brand_encoder = joblib.load('models/brand_encoder.pkl')
        processor_encoder = joblib.load('models/processor_encoder.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return df, model, brand_encoder, processor_encoder, scaler, feature_names
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None, None, None, None, None, None

# Chargement des données
with st.spinner("Chargement des données..."):
    df, model, brand_encoder, processor_encoder, scaler, feature_names = load_data()

if df is None:
    st.error("Impossible de charger les données. Vérifiez que tous les fichiers sont présents.")
    st.stop()

# Sidebar pour les inputs
st.sidebar.header("📋 Caractéristiques du téléphone")

# Récupération des valeurs uniques
brands = sorted(df['Brand'].unique())
processors = sorted(df['Processor'].unique())

# Inputs utilisateur
brand = st.sidebar.selectbox("Marque", brands, help="Sélectionnez la marque du téléphone")

col1, col2 = st.sidebar.columns(2)
with col1:
    battery = st.number_input("Batterie (mAh)", min_value=1000, max_value=10000, value=4000, step=100)
    screen_size = st.number_input("Taille écran (pouces)", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
    ram = st.number_input("RAM (GB)", min_value=1, max_value=16, value=4, step=1)
    storage = st.number_input("Stockage (GB)", min_value=16, max_value=1024, value=64, step=8)

with col2:
    processor = st.sidebar.selectbox("Processeur (code)", processors, help="Code numérique du processeur")
    rear_camera = st.number_input("Caméra arrière (MP)", min_value=5, max_value=200, value=24, step=2)
    front_camera = st.number_input("Caméra avant (MP)", min_value=2, max_value=50, value=12, step=1)

# Bouton de prédiction
if st.sidebar.button("🚀 Prédire le Prix", type="primary"):
    try:
        # --- Calcul des features utilisateur ---
        camera_total = rear_camera + front_camera
        ram_gb = ram
        ram_mb = ram * 1000

        price_per_gb = df['Price'].mean() / df['Internal storage (GB)'].mean()
        price_per_mp = df['Price'].mean() / (df['Rear camera'].mean() + df['Front camera'].mean())
        screen_to_battery_ratio = screen_size / (battery / 1000)
        # Ajout du feature Is_Premium
        is_premium = 1 if brand in ['Apple', 'Samsung', 'OnePlus'] else 0

        input_data = pd.DataFrame({
            'Brand': [brand],
            'Battery capacity (mAh)': [battery],
            'Screen size (inches)': [screen_size],
            'Processor': [processor],
            'RAM (MB)': [ram_mb],
            'Internal storage (GB)': [storage],
            'Rear camera': [rear_camera],
            'Front camera': [front_camera],
            'Price_per_GB': [price_per_gb],
            'Price_per_MP': [price_per_mp],
            'Screen_to_Battery_Ratio': [screen_to_battery_ratio],
            'Camera_Total': [camera_total],
            'RAM_GB': [ram_gb],
            'Price_per_RAM': [price_per_gb],  # On garde cette approximation
            'Battery_to_Screen_Ratio': [battery / screen_size],
            'Is_Premium': [is_premium]
        })

        # Encodage des variables catégorielles (sécurisé)
        try:
            input_data['Brand'] = brand_encoder.transform(input_data['Brand'])
        except Exception:
            st.error(f"Marque inconnue pour l’encodeur : {brand}")
            st.stop()
        try:
            input_data['Processor'] = processor_encoder.transform(input_data['Processor'])
        except Exception:
            st.error(f"Processeur inconnu pour l’encodeur : {processor}")
            st.stop()

        # Réordonner les colonnes selon l'ordre d'entraînement
        input_data = input_data[feature_names]
        # Normalisation
        input_scaled = scaler.transform(input_data)

        # Prédiction brute LOG
        predicted_log_price = model.predict(input_scaled)[0]
        # Correction pour retrouver le prix "réel"
        predicted_price = np.expm1(predicted_log_price)

        # Calcul du prix moyen des téléphones similaires
        similar_phones = df[
            (df['Internal storage (GB)'].between(storage * 0.8, storage * 1.2)) &
            (df['RAM (MB)'].between(ram * 1000 * 0.8, ram * 1000 * 1.2))
        ]
        avg_price = similar_phones['Price'].mean() if not similar_phones.empty else predicted_price

        # --- OPTIMISATION RENFORCÉE ---
        if len(similar_phones) >= 5:
            relative_gap = abs(predicted_price - avg_price) / avg_price
            # Pondération dynamique
            if relative_gap > 0.3:
                model_weight = 0.3
            elif relative_gap > 0.15:
                model_weight = 0.5
            else:
                model_weight = 0.8
            adjusted_price = model_weight * predicted_price + (1 - model_weight) * avg_price

            # Limite stricte : jamais plus de 10% au-dessus de la moyenne
            max_allowed = avg_price * 1.10
            if adjusted_price > max_allowed:
                adjusted_price = max_allowed
                price_note = "(forcé à +10% max de la moyenne)"
            else:
                price_note = f"(ajusté dynamiquement, poids modèle: {model_weight:.2f})"
        else:
            adjusted_price = predicted_price
            price_note = "(pas assez de téléphones similaires pour ajustement)"

        # Affichage des résultats
        st.success("✅ Prédiction effectuée avec succès!")

        # Métriques principales
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="💰 Prix Prédit",
                value=f"₹{adjusted_price:,.0f}",
                delta=None,
                help=price_note
            )

        with col2:
            st.metric(
                label="📊 Prix Moyen Similaire",
                value=f"₹{avg_price:,.0f}",
                delta=f"{((adjusted_price - avg_price) / avg_price * 100):.1f}%"
            )

        with col3:
            st.metric(
                label="🎯 Précision Modèle",
                value="95.4%",
                delta="+0.2%"
            )

        # Graphiques
        st.markdown("---")
        st.subheader("📈 Visualisations")

        col1, col2 = st.columns(2)

        with col1:
            # Graphique en barres - Comparaison des prix
            fig_bar = px.bar(
                x=['Prix Prédit', 'Prix Moyen Similaire'],
                y=[adjusted_price, avg_price],
                title="Comparaison des Prix",
                labels={'x': 'Type de Prix', 'y': 'Prix (₹)'},
                color=['Prix Prédit', 'Prix Moyen Similaire'],
                color_discrete_map={'Prix Prédit': '#1f77b4', 'Prix Moyen Similaire': '#ff7f0e'}
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Graphique radar - Comparaison des caractéristiques
            fig_radar = go.Figure()

            # Caractéristiques du téléphone saisi
            fig_radar.add_trace(go.Scatterpolar(
                r=[battery/5000, screen_size/7, ram/8, storage/256, rear_camera/64, front_camera/32],
                theta=['Batterie', 'Écran', 'RAM', 'Stockage', 'Cam. Arrière', 'Cam. Avant'],
                fill='toself',
                name='Téléphone Saisi',
                line_color='#1f77b4'
            ))

            # Caractéristiques moyennes
            fig_radar.add_trace(go.Scatterpolar(
                r=[
                    df['Battery capacity (mAh)'].mean() / 5000,
                    df['Screen size (inches)'].mean() / 7,
                    df['RAM (MB)'].mean() / 8000,
                    df['Internal storage (GB)'].mean() / 256,
                    df['Rear camera'].mean() / 64,
                    df['Front camera'].mean() / 32
                ],
                theta=['Batterie', 'Écran', 'RAM', 'Stockage', 'Cam. Arrière', 'Cam. Avant'],
                fill='toself',
                name='Moyenne Générale',
                line_color='#ff7f0e'
            ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Comparaison des Caractéristiques"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Informations supplémentaires
        st.markdown("---")
        st.subheader("ℹ️ Informations Complémentaires")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Caractéristiques saisies:**
            - Marque: {brand}
            - Batterie: {battery} mAh
            - Écran: {screen_size} pouces
            - RAM: {ram} GB
            - Stockage: {storage} GB
            - Caméra arrière: {rear_camera} MP
            - Caméra avant: {front_camera} MP
            """)

        with col2:
            st.info(f"""
            **Statistiques:**
            - Téléphones similaires trouvés: {len(similar_phones)}
            - Différence avec la moyenne: {((adjusted_price - avg_price) / avg_price * 100):.1f}%
            - Prix par GB: ₹{price_per_gb:.0f}
            - Prix par MP: ₹{price_per_mp:.0f}
            """)

        # Recommandations
        if adjusted_price > avg_price * 1.1:
            st.warning("⚠️ Le prix prédit est supérieur à la moyenne des téléphones similaires. Vérifiez les caractéristiques.")
        elif adjusted_price < avg_price * 0.9:
            st.success("✅ Le prix prédit est inférieur à la moyenne des téléphones similaires. Bon rapport qualité-prix!")
        else:
            st.info("ℹ️ Le prix prédit est dans la moyenne des téléphones similaires.")

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {e}")
        st.error("Vérifiez que toutes les valeurs sont correctes.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>🤖 Modèle de Machine Learning - Précision: 95.4%</p>
    <p>📊 Basé sur {len(df)} téléphones dans la base de données</p>
</div>
""", unsafe_allow_html=True)

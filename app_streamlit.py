import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math

st.set_page_config(
    page_title="Prédiction de Prix de Téléphones (USD)",
    page_icon="📱",
    layout="wide"
)

USD_RATE = 86.14  # 1 USD = 86.14 ₹

@st.cache_data
def load_data():
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

with st.spinner("Chargement des données..."):
    df, model, brand_encoder, processor_encoder, scaler, feature_names = load_data()

if df is None:
    st.stop()

brands = sorted(df['Brand'].dropna().unique())
processors = sorted(df['Processor'].dropna().unique())

col_form, col_main = st.columns([1, 2], gap="large")

with col_form:
    st.header("📋 Caractéristiques du téléphone")
    with st.form("input_form", clear_on_submit=False):
        brand = st.selectbox("Marque", brands)
        processor = st.selectbox("Processeur", processors)
        battery = st.number_input("Batterie (mAh)", int(df['Battery capacity (mAh)'].min()), int(df['Battery capacity (mAh)'].max()), 4000, step=100)
        screen_size = st.number_input("Taille écran (pouces)", float(df['Screen size (inches)'].min()), float(df['Screen size (inches)'].max()), 6.0, step=0.1)
        ram = st.number_input("RAM (GB)", 1, 16, 4, step=1)
        storage = st.number_input("Stockage (GB)", 16, 1024, 64, step=16)
        rear_camera = st.number_input("Caméra arrière (MP)", 5, 200, 48)
        front_camera = st.number_input("Caméra avant (MP)", 2, 50, 12)
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #38b24a;
                color: white;
                font-size: 1.4em;
                font-weight: bold;
                width: 100%;
                border-radius: 8px;
                padding: 0.6em 0;
                border: none;
                margin-top: 18px;
                margin-bottom: 6px;
                box-shadow: 0 3px 10px rgba(56,178,74,0.11);
                transition: background 0.2s;
            }
            div.stButton > button:first-child:hover {
                background: #2d9e3f;
                cursor:pointer;
            }
            </style>
        """, unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Prédire le Prix")

with col_main:
    st.markdown(
        "<h1 style='text-align:center; font-size:2.6em; font-weight: bold; color:#1f77b4;'>"
        "Prédiction de prix de Téléphones (USD)"
        "</h1>", unsafe_allow_html=True
    )
    tab_css = '''
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.5em !important;
        font-weight: 900 !important;
        padding: 0.6em 2em !important;
        border-bottom: 5px solid #d0f5dd !important;
        transition: background 0.25s, color 0.25s;
    }
    .stTabs [aria-selected="true"] {
        color: #38b24a !important;
        border-bottom: 5px solid #38b24a !important;
        background-color: #f6fff7 !important;
        text-shadow: 0px 1px 6px #d0f5dd;
    }
    </style>
    '''
    st.markdown(tab_css, unsafe_allow_html=True)

    if submitted:
        try:
            # ---------- FEATURE ENGINEERING (exact retrain_model.py) ----------
            ram_mb = int(ram * 1000)
            camera_total = rear_camera + front_camera

            # Pour les features dérivées qui nécessitent le prix, utiliser la moyenne du dataset (comme dans retrain_model.py pour l'inférence)
            mean_price = df['Price'].mean()
            # price_per_gb
            price_per_gb = mean_price / storage if storage > 0 else mean_price
            # price_per_mp
            price_per_mp = mean_price / camera_total if camera_total > 0 else mean_price
            # screen_to_battery_ratio
            screen_to_battery_ratio = screen_size / (battery / 1000) if battery > 0 else 1.0
            # price_per_ram
            price_per_ram = mean_price / ram_mb if ram_mb > 0 else mean_price
            # battery_to_screen_ratio
            battery_to_screen_ratio = battery / screen_size if screen_size > 0 else 3000
            # is_premium
            is_premium = int(brand in ['Apple', 'Samsung', 'OnePlus'])
            ram_gb = ram_mb / 1000

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
                'Price_per_RAM': [price_per_ram],
                'Battery_to_Screen_Ratio': [battery_to_screen_ratio],
                'Is_Premium': [is_premium]
            })

            # Encodage
            input_data['Brand'] = brand_encoder.transform(input_data['Brand'])
            input_data['Processor'] = processor_encoder.transform(input_data['Processor'])
            input_data = input_data[feature_names]
            input_scaled = scaler.transform(input_data)

            # Prédiction log(price), puis inversion expm1
            predicted_log_price = model.predict(input_scaled)[0]
            predicted_price_rs = np.expm1(predicted_log_price)
            predicted_usd = predicted_price_rs / USD_RATE

            # Trouver téléphones similaires (pour info, garder la logique simple)
            similar_phones = df[
                (df['Brand'] == brand) &
                (df['Processor'] == processor) &
                (df['RAM (MB)'] == ram_mb) &
                (df['Internal storage (GB)'] == storage)
            ]
            avg_price = similar_phones['Price'].mean() if not similar_phones.empty else mean_price
            avg_usd = avg_price / USD_RATE

            # Gestion sécurité
            if not math.isfinite(predicted_usd) or predicted_usd < 0:
                predicted_usd = 0
            if not math.isfinite(avg_usd) or avg_usd <= 0:
                avg_usd = predicted_usd if predicted_usd > 0 else 1

            # Calcul écart
            if avg_usd > 0:
                variation = ((predicted_usd - avg_usd) / avg_usd) * 100
            else:
                variation = 0
            variation_text = f"{abs(variation):.1f}%"

            onglet1, onglet2, onglet3 = st.tabs([
                "📱 Prix estimé", "📈 Visualisations", "ℹ️ Informations Complémentaires"
            ])

            with onglet1:
                st.markdown(
                    f"<div style='text-align:center; font-size:2em;'><b>${predicted_usd:,.2f}</b></div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='text-align:center; margin-top:20px; font-size:1.2em;'>"
                    f"Prix moyen similaire : <b>${avg_usd:,.2f}</b></div>",
                    unsafe_allow_html=True
                )
                if variation > 0:
                    st.markdown(
                        f"<div style='text-align:center; margin-top:10px; padding:8px; background-color:#ffe6e6; color:#cc0000; font-weight:bold; border-radius:6px;'>"
                        f"⬆️ Prix supérieur de {variation_text} à la moyenne</div>",
                        unsafe_allow_html=True)
                elif variation < 0:
                    st.markdown(
                        f"<div style='text-align:center; margin-top:10px; padding:8px; background-color:#e6ffe6; color:#007500; font-weight:bold; border-radius:6px;'>"
                        f"⬇️ Prix inférieur de {variation_text} à la moyenne</div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div style='text-align:center; margin-top:10px; padding:8px; background-color:#f0f0f0; color:#555; font-weight:bold; border-radius:6px;'>"
                        "Prix équivalent à la moyenne</div>",
                        unsafe_allow_html=True)

                if predicted_usd > avg_usd * 1.1:
                    st.warning("⚠️ Prix supérieur à la moyenne — attention au rapport qualité/prix")
                elif predicted_usd < avg_usd * 0.9:
                    st.success("✅ Prix en dessous de la moyenne — bon rapport qualité/prix")
                else:
                    st.info("ℹ️ Prix dans la moyenne des téléphones similaires")

            with onglet2:
                col1, col2 = st.columns(2)
                fig_bar = px.bar(
                    x=['Prix Prédit', 'Prix Moyen Similaire'],
                    y=[predicted_usd, avg_usd],
                    title="Comparaison des Prix",
                    labels={'x': 'Type de Prix', 'y': 'Prix (USD)'},
                    color=['Prix Prédit', 'Prix Moyen Similaire'],
                    color_discrete_map={'Prix Prédit': '#1f77b4', 'Prix Moyen Similaire': '#ff7f0e'}
                )
                fig_bar.update_layout(template="plotly_white", showlegend=False)
                with col1:
                    st.plotly_chart(fig_bar, use_container_width=True)

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=[battery/5000, screen_size/7, ram/8, storage/256, rear_camera/64, front_camera/32],
                    theta=['Batterie', 'Écran', 'RAM', 'Stockage', 'Cam. Arrière', 'Cam. Avant'],
                    fill='toself', name='Téléphone Saisi', line_color='#1f77b4'))
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
                    fill='toself', name='Moyenne Générale', line_color='#ff7f0e'))
                fig_radar.update_layout(template="plotly_white",
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True,
                    title="Comparaison des Caractéristiques")
                with col2:
                    st.plotly_chart(fig_radar, use_container_width=True)

                # --- Ajout d'une distribution du prix en USD ---
                st.markdown("#### Distribution des prix (USD)")
                fig_hist = px.histogram(df, x=df['Price']/USD_RATE, nbins=30,
                                       labels={'x': 'Prix (USD)'}, title="Distribution des prix sur la base de données")
                fig_hist.add_vline(x=predicted_usd, line_dash="dash", line_color="red", annotation_text="Prix prédit", annotation_position="top left")
                st.plotly_chart(fig_hist, use_container_width=True)

            with onglet3:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **Caractéristiques saisies :**
                    - Marque : {brand}
                    - Processeur : {processor}
                    - Batterie : {battery} mAh
                    - Écran : {screen_size} pouces
                    - RAM : {ram} GB
                    - Stockage : {storage} GB
                    - Caméras : arrière {rear_camera} MP / avant {front_camera} MP
                    """)
                with col2:
                    st.info(f"""
                    **Statistiques :**
                    - Téléphones similaires trouvés : {len(similar_phones)}
                    - Différence avec la moyenne : {variation:.1f}%
                    - Prix par Go : ${(price_per_gb / USD_RATE):.2f}
                    - Prix par MP : ${(price_per_mp / USD_RATE):.2f}
                    """)

        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")
    else:
        st.markdown(
            "<div style='margin-top:40px; text-align:center; font-size:1.7em; color:#A0A0A0;'>"
            "Renseignez les caractéristiques à gauche puis cliquez sur <b>Prédire le Prix</b>."
            "</div>",
            unsafe_allow_html=True
        )

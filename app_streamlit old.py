import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Prédiction de Prix de Téléphones",
    page_icon="📱",
    layout="wide"
)

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

brands = sorted(df['Brand'].unique())
processors = sorted(df['Processor'].unique())
usd_rate = 83.0

# --- Layout principal
col_form, col_main = st.columns([1, 2], gap="large")

with col_form:
    st.header("📋 Caractéristiques du téléphone")
    # CSS pour cacher le bouton natif Streamlit
    st.markdown("""
        <style>
        .css-1x8cf1d .stButton > button {visibility: hidden;}
        .stButton>button {visibility: hidden;}
        #custom_predict_btn {
            width: 100%;
            background: #38b24a;
            color: white;
            font-weight: bold;
            font-size: 1.4em;
            border-radius: 8px;
            padding: 0.6em 0;
            border: none;
            margin-top: 18px;
            margin-bottom: 6px;
            box-shadow: 0 3px 10px rgba(56,178,74,0.11);
            transition: background 0.2s;
        }
        #custom_predict_btn:hover {
            background: #2d9e3f;
            cursor:pointer;
        }
        #custom_predict_btn .emoji {margin-right:0.5em;}
        </style>
        <script>
        function submitForm(){
            const e = window.parent.document.querySelector('button[kind="primary"]');
            if(e) e.click();
        }
        </script>
    """, unsafe_allow_html=True)
    with st.form("input_form", clear_on_submit=False):
        brand = st.selectbox("Marque", brands)
        processor = st.selectbox("Processeur", processors)
        battery = st.number_input("Batterie (mAh)", 1000, 10000, 4000, step=100)
        screen_size = st.number_input("Taille écran (pouces)", 4.0, 8.0, 6.1, step=0.1)
        ram = st.number_input("RAM (GB)", 1, 16, 6)
        storage = st.number_input("Stockage (GB)", 16, 1024, 128, step=16)
        rear_camera = st.number_input("Caméra arrière (MP)", 5, 200, 48)
        front_camera = st.number_input("Caméra avant (MP)", 2, 50, 12)
        # Notre bouton custom, avec JS qui simule le click du bouton natif
        st.markdown("""
            <button id="custom_predict_btn" onclick="submitForm();return false;">
                <span class="emoji">🚀</span>Prédire le Prix
            </button>
        """, unsafe_allow_html=True)
        submitted = st.form_submit_button("")

with col_main:
    st.markdown(
        "<h1 style='text-align:center; font-size:2.6em; font-weight: bold; color:#1f77b4;'>"
        "Prédiction de prix de Téléphones"
        "</h1>", unsafe_allow_html=True
    )

    # Custom CSS pour onglets plus visibles
    tab_css = '''
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.6em !important;
        font-weight: 800 !important;
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
            ram_mb = ram * 1000
            camera_total = rear_camera + front_camera
            price_per_gb = df['Price'].mean() / df['Internal storage (GB)'].mean()
            price_per_mp = df['Price'].mean() / (df['Rear camera'].mean() + df['Front camera'].mean())
            screen_to_battery_ratio = screen_size / (battery / 1000)

            input_data = pd.DataFrame({
                'Brand': [brand],
                'Battery capacity (mAh)': [battery],
                'Screen size (inches)': [screen_size],
                'Processor': [processor],
                'RAM (MB)': [ram_mb],
                'Internal storage (GB)': [storage],
                'Rear camera': [rear_camera],
                'Front camera': [front_camera],
                'Camera_Total': [camera_total],
                'RAM_GB': [ram],
                'Price_per_GB': [price_per_gb],
                'Price_per_MP': [price_per_mp],
                'Screen_to_Battery_Ratio': [screen_to_battery_ratio],
                'Price_per_RAM': [price_per_gb],
                'Battery_to_Screen_Ratio': [battery / screen_size]
            })

            input_data['Brand'] = brand_encoder.transform(input_data['Brand'])
            input_data['Processor'] = processor_encoder.transform(input_data['Processor'])
            input_data = input_data[feature_names]
            input_scaled = scaler.transform(input_data)

            predicted_price = model.predict(input_scaled)[0]
            predicted_usd = predicted_price / usd_rate

            similar_phones = df[
                (df['Internal storage (GB)'].between(storage * 0.8, storage * 1.2)) &
                (df['RAM (MB)'].between(ram * 1000 * 0.8, ram * 1000 * 1.2))
            ]
            avg_price = similar_phones['Price'].mean() if not similar_phones.empty else predicted_price
            avg_usd = avg_price / usd_rate

            onglet1, onglet2, onglet3 = st.tabs([
                "📱 Prix estimé", "📈 Visualisations", "ℹ️ Informations Complémentaires"
            ])

            with onglet1:
                st.markdown(
                    f"<div style='text-align:center; font-size:2em;'><b>₹{predicted_price:,.0f} (≈ ${predicted_usd:,.2f})</b></div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='text-align:center; margin-top:20px; font-size:1.2em;'>"
                    f"Prix moyen similaire : <b>₹{avg_price:,.0f} (≈ ${avg_usd:,.2f})</b></div>",
                    unsafe_allow_html=True
                )
                if predicted_price > avg_price * 1.1:
                    st.warning("⚠️ Prix supérieur à la moyenne — attention au rapport qualité/prix")
                elif predicted_price < avg_price * 0.9:
                    st.success("✅ Prix en dessous de la moyenne — bon rapport qualité/prix")
                else:
                    st.info("ℹ️ Prix dans la moyenne des téléphones similaires")

            with onglet2:
                col1, col2 = st.columns(2)
                fig_bar = px.bar(
                    x=['Prix Prédit', 'Prix Moyen Similaire'],
                    y=[predicted_price, avg_price],
                    title="Comparaison des Prix",
                    labels={'x': 'Type de Prix', 'y': 'Prix (₹)'},
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

            with onglet3:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **Caractéristiques saisies :**
                    - Marque : {brand}
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
                    - Différence avec la moyenne : {((predicted_price - avg_price) / avg_price * 100):.1f}%
                    - Prix par Go (₹/GB) : ₹{price_per_gb:.0f} (≈ ${price_per_gb / usd_rate:.2f})
                    - Prix par MP (₹/MP) : ₹{price_per_mp:.0f} (≈ ${price_per_mp / usd_rate:.2f})
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

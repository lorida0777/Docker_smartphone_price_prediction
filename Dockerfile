# 📦 Image légère avec Python 3.9
FROM python:3.9-slim

# 📁 Dossier de travail
WORKDIR /app

# 📄 Copier les dépendances
COPY requirements.txt .

# 🔧 Installer les paquets nécessaires
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# 📂 Copier le reste des fichiers
COPY . .

# 🌐 Ouvrir le port de Streamlit
EXPOSE 8501

# 🚀 Lancer l'application Streamlit
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.enableCORS=false"]

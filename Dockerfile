# Wybieramy lekką wersję Pythona
FROM python:3.11-slim

# Zainstaluj systemowe zależności (jeśli potrzebne)
RUN apt-get update && apt-get install -y \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ustaw katalog roboczy
WORKDIR /app

# Skopiuj pliki projektu
COPY . .

# Zainstaluj zależności
RUN pip install --no-cache-dir -r requirements.txt

# Otwarte porty
EXPOSE 8501

# Uruchom aplikację Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.headless=true"]

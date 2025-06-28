import sys

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import base64
import torch
import os
from pymongo import MongoClient
from datetime import datetime
from pytz import timezone, UTC


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "best_model.pth"
SCALER_PATH = BASE_DIR / "model" / "wine_scaler.pkl"
IMAGE_DIR = BASE_DIR / "photos"

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from model.wine_dataset import WineQualityMLP


def get_user_id():
    return st.session_state.get("user_id", None)


@st.cache_resource
def get_mongo_client():
    try:
        uri = st.secrets["MONGO_URI"]
    except Exception:
        uri = os.environ.get("MONGO_URI")
    if not uri:
        st.error("Nie znaleziono URI do MongoDB.")
        st.stop()
    return MongoClient(uri)


client = get_mongo_client()
db = client["wine_app"]
predictions_collection = db["predictions"]


def save_prediction(input_data: dict, prediction: float):
    record = {
        "user_id": get_user_id(),
        "timestamp": datetime.utcnow(),
        "input": input_data,
        "prediction": prediction,
        "model_version": "mlp-v1.0",
    }
    predictions_collection.insert_one(record)


@st.cache_resource
def load_model():
    model = WineQualityMLP(input_size=13)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def preprocess_input(user_data):
    user_data["acidity_ratio"] = user_data["fixed acidity"] / (
        user_data["volatile acidity"] + 1e-7
    )
    user_data["sulfur_ratio"] = user_data["free sulfur dioxide"] / (
        user_data["total sulfur dioxide"] + 1e-7
    )
    user_data["sweetness_index"] = user_data["residual sugar"] / (
        user_data["alcohol"] + 1e-7
    )
    user_data["total_acidity"] = (
        user_data["fixed acidity"]
        + user_data["volatile acidity"]
        + user_data["citric acid"]
    )
    user_data["body_score"] = (user_data["density"] * 1000) - user_data["alcohol"]
    user_data["minerality"] = user_data["sulphates"] + user_data["chlorides"]

    features_to_remove = ["fixed acidity", "residual sugar", "minerality", "density"]
    for feature in features_to_remove:
        if feature in user_data.index:
            user_data = user_data.drop(feature)

    final_features = [
        "volatile acidity",
        "citric acid",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "pH",
        "sulphates",
        "alcohol",
        "acidity_ratio",
        "sulfur_ratio",
        "sweetness_index",
        "total_acidity",
        "body_score",
    ]

    user_data_filtered = user_data[final_features]

    scaler = load_scaler()
    user_data_scaled = scaler.transform(user_data_filtered.to_frame().T)
    return pd.DataFrame(user_data_scaled, columns=final_features)


def add_style(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
            <style>
            input, select, textarea {{
                max-width: 100%;
            }}           
            .stTabs [data-baseweb="tab"] {{
                font-size: 20px;
                padding: 0.75rem 1.5rem;
            }}
            .block-container {{
                max-width: 100% !important;
                padding-left: 3rem;
                padding-right: 3rem;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-color: rgba(0, 0, 0, 0.6);
                z-index: 0;
            }}
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}
            </style>
            """,
        unsafe_allow_html=True,
    )


def main():
    add_style(str(IMAGE_DIR / "Wine_background.jpg"))

    col_main, col_result = st.columns([1, 2])

    with col_main:
        st.title("WineToGrapes - oceń jakoś wina")

        tab_basic, tab_advanced, tab_predict, tab_history = st.tabs(
            ["Podstawowe", "Zaawansowane", "Przewiduj", "Historia"]
        )

        with tab_basic:
            st.markdown("### Kwasowość")

            fixed_acidity = st.slider(
                "Kwasowość stała (g/L)",
                min_value=4.0,
                max_value=16.0,
                value=8.0,
                step=0.1,
                help="Nieulotne kwasy jak winowy, jabłkowy",
            )

            volatile_acidity = st.slider(
                "Kwasowość lotna", min_value=0.0, max_value=1.5, value=0.5, step=0.01
            )

            citric_acid = st.slider(
                "Kwas cytrynowy (g/L)",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.01,
                help="Dodaje świeżość i smak do wina",
            )

            st.markdown("### Cukier i Alkohol")

            residual_sugar = st.number_input(
                "Cukier resztkowy (g/L)",
                min_value=0.5,
                max_value=16.0,
                value=2.0,
                help="Cukier pozostały po fermentacji",
            )

            alcohol = st.slider(
                "Alkohol (% vol)",
                min_value=8.0,
                max_value=15.0,
                value=10.0,
                step=0.1,
                help="Zawartość alkoholu etylowego",
            )

        with tab_advanced:
            st.markdown("### Dwutlenek Siarki")

            free_so2 = st.number_input(
                "Wolny SO₂ (mg/L)",
                min_value=1.0,
                max_value=80.0,
                value=15.0,
                help="Zapobiega utlenianiu i działaniu bakterii",
            )

            total_so2 = st.number_input(
                "Całkowity SO₂ (mg/L)",
                min_value=6.0,
                max_value=300.0,
                value=50.0,
                help="Suma wolnego i związanego SO₂",
            )

            st.markdown("### Właściwości Fizyczne")

            col1, col2 = st.columns(2)
            with col1:
                density = st.slider(
                    "Gęstość (g/cm³)",
                    min_value=0.99,
                    max_value=1.01,
                    value=0.996,
                    step=0.001,
                    help="Zależy od zawartości alkoholu i cukru",
                )
            with col2:
                ph = st.slider(
                    "pH",
                    min_value=2.5,
                    max_value=4.5,
                    value=3.3,
                    step=0.1,
                    help="Kwasowość wina (niższe = bardziej kwaśne)",
                )

            st.markdown("### Minerały")

            col3, col4 = st.columns(2)
            with col3:
                chlorides = st.slider(
                    "Chlorki (g/L)",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.08,
                    step=0.01,
                    help="Zawartość soli w winie",
                )
            with col4:
                sulphates = st.slider(
                    "Siarczany (g/L)",
                    min_value=0.3,
                    max_value=2.0,
                    value=0.6,
                    step=0.05,
                    help="Przyczyniają się do SO₂ i jakości wina",
                )

        with tab_predict:
            user_input = pd.Series(
                {
                    "fixed acidity": fixed_acidity,
                    "volatile acidity": volatile_acidity,
                    "citric acid": citric_acid,
                    "residual sugar": residual_sugar,
                    "chlorides": chlorides,
                    "free sulfur dioxide": free_so2,
                    "total sulfur dioxide": total_so2,
                    "density": density,
                    "pH": ph,
                    "sulphates": sulphates,
                    "alcohol": alcohol,
                }
            )

            with st.form("wine_predict_form"):
                st.markdown("**Przewiduj jakość Twojego wina**")
                st.markdown("Wypełnij parametry chemiczne w poprzednich zakładkach:")
                submitted = st.form_submit_button("Oceń Jakość Wina")

        with tab_history:
            st.markdown("### Historia Twoich Predykcji")

            if "refresh" not in st.session_state:
                st.session_state.refresh = False
            if "clear_history" not in st.session_state:
                st.session_state.clear_history = False

            username = st.text_input("Podaj swoją nazwę użytkownika", max_chars=32)

            if username:
                st.session_state["user_id"] = username.strip().lower()
            else:
                st.warning("Wprowadź nazwę użytkownika, aby kontynuować.")
                st.stop()

            if st.session_state.get("clear_history"):
                predictions_collection.delete_many({"user_id": get_user_id()})
                st.success("Historia została usunięta.")
                st.session_state.clear_history = False

            if st.session_state.get("refresh"):
                st.success("Dane zostały odświeżone.")
                st.session_state.refresh = False

            history = list(
                predictions_collection.find({"user_id": get_user_id()}).sort(
                    "timestamp", -1
                )
            )

            if history:
                for entry in history:
                    utc_time = entry["timestamp"].replace(tzinfo=UTC)
                    local_time = utc_time.astimezone(timezone("Europe/Warsaw"))
                    timestamp = local_time.strftime("%Y-%m-%d %H:%M")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("#### TIME:", timestamp)
                        st.write("**Parametry wejściowe:**")
                        for k, v in entry["input"].items():
                            st.write(f"- {k}: `{round(v, 2)}`")
                    with col2:
                        st.metric("Predykcja", f"{round(entry['prediction'], 1)}/10")

            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("Wyczyść historię"):
                    st.session_state.clear_history = True
                    st.rerun()

            with col_btn2:
                if st.button("Odśwież dane"):
                    st.session_state.refresh = True
                    st.rerun()

    with col_result:
        if submitted:
            try:
                processed_data = preprocess_input(user_input)

                model = load_model()
                tensor_input = torch.tensor(processed_data.values, dtype=torch.float32)
                prediction = model(tensor_input)[0].item()

                # Ograniczenie do skali 0-10
                prediction = max(0, min(10, float(prediction)))

                st.subheader("Wynik Analizy")
                st.metric("Przewidywana jakość wina", f"{prediction:.1f}/10")

                if prediction >= 8.0:
                    comment = (
                        "Wyjątkowe wino! Charakterystyka godna najlepszych sommelierów"
                    )
                    status_color = "🟢"
                elif prediction >= 7.0:
                    comment = (
                        "Bardzo dobre wino o wysokiej jakości. Polecane do degustacji"
                    )
                    status_color = "🟢"
                elif prediction >= 6.0:
                    comment = "Solidne wino o dobrej jakości. Przyjemne w smaku"
                    status_color = "🟡"
                elif prediction >= 5.0:
                    comment = "Przeciętne wino. Może wymagać dopracowania receptury"
                    status_color = "🟡"
                elif prediction >= 4.0:
                    comment = "Poniżej oczekiwań. Warto przeanalizować proces produkcji"
                    status_color = "🔴"
                else:
                    comment = "Niska jakość. Konieczne zmiany w składzie chemicznym"
                    status_color = "🔴"

                st.markdown(f"### {status_color} *{comment}*")

                prediction = model(tensor_input)[0].item()
                prediction = max(0, min(10, float(prediction)))

                save_prediction(
                    {k: float(round(v, 4)) for k, v in user_input.items()}, prediction
                )

                st.markdown("### Analiza Składników")

                col_a, col_b = st.columns(2)
                with col_a:
                    alcohol_status = (
                        "🟢 Optymalna" if 10 <= alcohol <= 14 else "🔴 Wymaga korekty"
                    )
                    st.write(f"**Alkohol ({alcohol}% vol):** {alcohol_status}")

                    acidity_status = (
                        "🟢 Zbalansowana"
                        if 0.2 <= volatile_acidity <= 0.6
                        else "🔴 Problematyczna"
                    )
                    st.write(f"**Kwasowość lotna:** {acidity_status}")

                with col_b:
                    ph_status = (
                        "🟢 Optymalne" if 3.0 <= ph <= 3.8 else "🔴 Wymaga korekty"
                    )
                    st.write(f"**pH ({ph}):** {ph_status}")

                    sulphates_status = (
                        "🟢 Odpowiednie" if sulphates >= 0.5 else "🔴 Za niskie"
                    )
                    st.write(f"**Siarczany:** {sulphates_status}")

                with st.expander("Obliczone cechy zaawansowane"):
                    acidity_ratio = fixed_acidity / (volatile_acidity + 1e-7)
                    body_score = (density * 1000) - alcohol
                    sweetness_index = residual_sugar / (alcohol + 1e-7)

                    st.write(f"**Stosunek kwasowości:** {acidity_ratio:.2f}")
                    st.write(f"**Wskaźnik ciała wina:** {body_score:.2f}")
                    st.write(f"**Indeks słodkości:** {sweetness_index:.3f}")

            except Exception as e:
                st.error(f"Błąd podczas predykcji: {str(e)}")


if __name__ == "__main__":
    main()

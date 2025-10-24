import base64
import streamlit as st

def apply_nubank_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

        /* ——— Fonte Montserrat para todo o App ——— */
        html, body, [class*="css"] {
            font-family: 'Montserrat', sans-serif;
        }

        /* ——— Botões (todos os st.button) ——— */
        .stButton button {
            background-color: #8A05BE !important;
            color: #FFFFFF !important;
            font-size: 18px;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.5em 1em;
            transition: background-color 0.2s ease;
        }
        .stButton button:hover {
            background-color: #6F03A1 !important;
        }

        /* ——— Cabeçalhos H1, H2, H3 ——— */
        h1 {
            color: #8A05BE;
            font-weight: 700;
            font-size: 3rem;
        }
        h2 {
            color: #8A05BE;
            font-weight: 700;
            font-size: 2rem;
        }
        h3 {
            color: #6F03A1;
            font-weight: 700;
            font-size: 1.25rem;
        }

        /* ——— Parágrafos e textos padrão ——— */
        p {
            color: inherit;
            font-size: 1rem;
        }

        /* ========== INPUTS (st.text_input, st.text_area) ========== */
        html[data-theme="light"] .stTextInput>div>div>input,
        html[data-theme="light"] .stTextArea>div>div>textarea {
            background-color: #E6E6FA !important;
            border-radius: 5px;
            color: #000000 !important;
        }
        html[data-theme="light"] .stTextInput>div>div>input::placeholder,
        html[data-theme="light"] .stTextArea>div>div>textarea::placeholder {
            color: #666666 !important;
        }

        html[data-theme="dark"] .stTextInput>div>div>input,
        html[data-theme="dark"] .stTextArea>div>div>textarea {
            background-color: #3A3A3A !important;
            border-radius: 5px;
            color: #FFFFFF !important;
        }
        html[data-theme="dark"] .stTextInput>div>div>input::placeholder,
        html[data-theme="dark"] .stTextArea>div>div>textarea::placeholder {
            color: #CCCCCC !important;
        }

        /* ——— Container “NuVision” ——— */
        .title-and-image-container {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            margin-bottom: 1rem;
        }
        .title-and-image-container img {
            border-radius: 5px;
        }
        .title {
            color: #8A05BE;
            font-size: 1.25rem;
            font-weight: 700;
            margin-left: 0.5rem;
        }

        /* ——— Card para métricas ——— */
        .metric-card {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .metric-card.dark {
            background-color: #2A2A2A;
        }
        .metric-card p {
            margin: 0;
            color: inherit;
        }

        /* ——— Sidebar no modo escuro ——— */
        html[data-theme="dark"] [data-testid="stSidebar"] > div:first-child {
            background-color: #171717 !important;
        }

        /* ——— Cor de fundo do corpo principal no modo escuro ——— */
        html[data-theme="dark"] .stApp {
            background-color: #212121 !important;
        }

        /* ——— Sidebar no modo claro ——— */
        html[data-theme="light"] [data-testid="stSidebar"] > div:first-child {
            background-color: #F0F0F0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def display_title_and_image():
    st.markdown(
        f"""
        <div class="title-and-image-container">
            <div class="title-and-image-content">
                <img src="data:image/png;base64,{get_image_as_base64('./nubank-logo-0-1.png')}" width="80">
                <div class="title">NuVision</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def apply_custom_style():
    st.markdown(
        """
        <style>
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True
    )

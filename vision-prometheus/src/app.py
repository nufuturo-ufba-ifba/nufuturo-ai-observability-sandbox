import streamlit as st
import os

# Pega o caminho do diretório onde o app.py está (/src)
current_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Vision Prometheus",
    layout="wide"
)

def Main():     
    # Como todos os arquivos estão em src/models, apontamos para lá:
    # O app.py está em /src, então buscamos em /src/models/arquivo.py
    
    home_path = os.path.join(current_dir, "models", "homepage.py")
    isolation_path = os.path.join(current_dir, "models", "isolation_forest.py")
    z_score_path = os.path.join(current_dir, "models", "z_score.py")
    matrix_path = os.path.join(current_dir, "models", "matrix_profile.py")
    prophet_path = os.path.join(current_dir, "models", "prophet_analysis.py")

    # Criando as instâncias das páginas
    home = st.Page(home_path, title="Home", icon="🏠")
    isolation_forest = st.Page(isolation_path, title="Isolation Forest")
    z_score = st.Page(z_score_path, title="Z-Score")
    matrix_profile = st.Page(matrix_path, title="Matrix Profile")
    prophet = st.Page(prophet_path, title="Prophet")
    
    pages = st.navigation(
        {
            "Home": [home],
            "Anomaly Detection Analysis": [
                isolation_forest,
                z_score,
                matrix_profile,
                prophet
            ]
        }
    )        
    
    pages.run()

if __name__ == "__main__":
    Main()
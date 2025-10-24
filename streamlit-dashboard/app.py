import streamlit as st
from style import apply_nubank_style, display_title_and_image, apply_custom_style

# ─────────────────────────────────────────────────────────────────────────────
# Defina o favicon e o título da página ANTES de qualquer outra chamada ao Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NuVision",
    page_icon="nubank-logo-0-1.png",  # caminho relativo para o seu ícone
    layout="wide"
)

def Main():     
    # 1) Aplicar o estilo Nubank (CSS personalizado)
    apply_nubank_style()
    # 2) Exibir logo + título “NuVision” no canto superior direito
    display_title_and_image()
    # 3) Esconder menu de configurações e rodapé do Streamlit
    apply_custom_style()

    # 4) Exibir o logo novamente usando st.logo (conforme solicitado)
    st.logo("./nubank-logo-0-1.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Definição das páginas do seu app
    # ─────────────────────────────────────────────────────────────────────────
    home = st.Page("homepage.py", title="Home")
    isolation_forest = st.Page("isolation_forest.py", title="Isolation Forest")
    z_score = st.Page("z_score.py", title="Z-Score")
    matrix_profile = st.Page("matrix_profile.py", title="Matrix Profile")
    prophet = st.Page("prophet_analysis.py", title="Prophet")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Roteamento / Navegação
    # ─────────────────────────────────────────────────────────────────────────
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
    
    # Executar navegação
    pages.run()

if __name__ == "__main__":
    Main()

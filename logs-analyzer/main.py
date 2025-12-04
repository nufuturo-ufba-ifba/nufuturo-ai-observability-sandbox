import streamlit as st
from style import apply_nubank_style, display_title_and_image, apply_custom_style

# 1) Configuração Inicial
st.set_page_config(
    page_title="LogVision",
    page_icon="📊", 
    layout="wide"
)

# 2) Função Principal que carrega o estilo e define a navegação
def Main():     
    # Aplica o CSS do Nubank
    apply_nubank_style()
    
    # Header Personalizado (Logo no canto)
    display_title_and_image()
    
    # Ajustes finais de CSS
    apply_custom_style()

    # Tenta mostrar o logo na Sidebar (se existir o arquivo)
    try:
        st.logo("./nubank-logo-0-1.png")
    except:
        pass # Ignora se não tiver imagem

    # ─────────────────────────────────────────────────────────────────────────
    # DEFINIÇÃO DAS PÁGINAS (Apontando para seus arquivos existentes)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Página Principal
    principal_page = st.Page("logsanalyser.py", title="Dashboard Principal", icon="🏠")
    
    # Ferramentas Específicas
    txt_page = st.Page("logsanalyser-txt.py", title="Analisador TXT", icon="📄")
    json_page = st.Page("logsanalyser-json.py", title="Analisador JSON", icon="🔍")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROTEAMENTO / NAVEGAÇÃO
    # ─────────────────────────────────────────────────────────────────────────
    pages = st.navigation(
        {
            "Home": [principal_page],
            "Ferramentas de Logs": [
                txt_page,
                json_page
            ]
        }
    )        
    
    # Executar a navegação
    pages.run()

if __name__ == "__main__":
    Main()
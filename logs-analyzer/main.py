import streamlit as st
# O style continua sendo importado da raiz, então não muda nada aqui
from style import apply_nubank_style, display_title_and_image, apply_custom_style

# 1) Configuração Inicial
st.set_page_config(
    page_title="LogVision",
    page_icon="📊", 
    layout="wide"
)

# 2) Função Principal
def Main():     
    apply_nubank_style()
    display_title_and_image()
    apply_custom_style()

    try:
        st.logo("./nubank-logo-0-1.png")
    except:
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # DEFINIÇÃO DAS PÁGINAS (Agora apontando para a pasta src/)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Repare que agora adicionamos "src/" antes do nome do arquivo
    principal_page = st.Page("src/logsanalyser.py", title="Dashboard Principal", icon="🏠")
    
    txt_page = st.Page("src/logsanalyser-txt.py", title="Analisador TXT", icon="📄")
    
    json_page = st.Page("src/logsanalyser-json.py", title="Analisador JSON", icon="🔍")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROTEAMENTO
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
    
    pages.run()

if __name__ == "__main__":
    Main()
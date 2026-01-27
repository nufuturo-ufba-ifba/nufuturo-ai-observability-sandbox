import streamlit as st
from style import apply_style, display_title_and_image, apply_custom_style

st.set_page_config(
    page_title="LogVision",
    page_icon="📊", 
    layout="wide"
)

def Main():     
    apply_style()
    display_title_and_image()
    apply_custom_style()

    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background-color: #F0F2F6;
            }
            [data-testid="stSidebar"] * {
                color: #262626 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    alexandria_page = st.Page("src/logsanalyser-alexandria.py", title="Analisador de Logs Alexandria")
    principal_page = st.Page("src/logsanalyser.py", title="Analisador de Logs CSV")    
    txt_page = st.Page("src/logsanalyser-txt.py", title="Analisador TXT")    
    json_page = st.Page("src/logsanalyser-json.py", title="Analisador JSON")

    pages = st.navigation(
        {
            "Home": [alexandria_page, principal_page],
            "Ferramentas de Logs": [
                txt_page,
                json_page
            ]
        }
    )        
    
    pages.run()

if __name__ == "__main__":
    Main()
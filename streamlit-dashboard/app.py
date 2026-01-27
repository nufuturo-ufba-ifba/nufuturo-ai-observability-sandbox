import streamlit as st
from style import apply_style, display_title_and_image, apply_custom_style

st.set_page_config(
    page_title="Vision Prometheus",
    layout="wide"
)

def Main():     
    apply_style()
    display_title_and_image()
    apply_custom_style()

    home = st.Page("homepage.py", title="Home")
    isolation_forest = st.Page("isolation_forest.py", title="Isolation Forest")
    z_score = st.Page("z_score.py", title="Z-Score")
    matrix_profile = st.Page("matrix_profile.py", title="Matrix Profile")
    prophet = st.Page("prophet_analysis.py", title="Prophet")
    
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
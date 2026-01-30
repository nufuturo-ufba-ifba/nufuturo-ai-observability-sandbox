import streamlit as st
import os
from style import apply_style, display_title_and_image, apply_custom_style

current_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Vision Prometheus",
    layout="wide"
)

def Main():     
   
    apply_style()
    display_title_and_image()
    apply_custom_style()
   

   
    home_path = os.path.join(current_dir, "models", "homepage.py")
    isolation_path = os.path.join(current_dir, "models", "isolation_forest.py")
    z_score_path = os.path.join(current_dir, "models", "z_score.py")
    matrix_path = os.path.join(current_dir, "models", "matrix_profile.py")
    prophet_path = os.path.join(current_dir, "models", "prophet_analysis.py")

   
    home = st.Page(home_path, title="Home", icon="🏠")
    isolation_forest = st.Page(isolation_path, title="Isolation Forest")
    z_score = st.Page(z_score_path, title="Z-Score")
    matrix_profile = st.Page(matrix_profile_path if 'matrix_profile_path' in locals() else matrix_path, title="Matrix Profile")
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
import streamlit as st

st.set_page_config(
    page_title="À Propos",
    page_icon="ℹ️"
)

def about_page():
    st.title("ℹ️ À Propos de CV Analytics Pro")
    
    st.markdown("""
    ### 🚀 Notre Vision
    Révolutionner le recrutement grâce à l'intelligence artificielle.
    
    ### 📌 Fonctionnalités
    - Analyse sémantique des CV
    - Matching intelligent candidat-poste
    - Outils collaboratifs pour RH
    """)
    
    st.image("https://via.placeholder.com/800x300?text=Notre+Equipe", width=800)
    
    st.markdown("""
    ### 📞 Contact
    **Email:** Djiguibak06@gmail.com  
    **Téléphone:** +225 07 69 16 34 68
    
    ### 📜 Mentions Légales
    [Cliquez ici pour voir nos CGU](#)
    """)

if __name__ == "__main__":
    about_page()
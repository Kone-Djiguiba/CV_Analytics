import streamlit as st

st.set_page_config(
    page_title="Espace Recruteur",
    page_icon="👔",
    layout="wide"
)

def recruiter_space():
    st.title("👔 Espace Recruteur")
    
    tab1, tab2, tab3 = st.tabs(["📋 Profil Recherché", "📄 Types de Contrat", "💡 Conseils RH"])
    
    with tab1:
        st.header("Définir le Profil Idéal")
        with st.form("profil_form"):
            cols = st.columns(2)
            with cols[0]:
                skills = st.multiselect("Compétences Techniques", ["Python", "SQL", "Machine Learning"])
                exp = st.slider("Expérience minimum", 0, 20, 3)
            with cols[1]:
                lang = st.multiselect("Langues Requises", ["Français", "Anglais"])
                education = st.selectbox("Niveau d'étude", ["Bac+2", "Bac+5", "Doctorat"])
            
            if st.form_submit_button("💾 Sauvegarder le Profil"):
                st.success("Profil enregistré!")
    
    with tab2:
        st.header("Configuration des Contrats")
        contract = st.radio("Type de Contrat", ["Stage", "CDD", "CDI", "Alternance"])
        
        if contract == "Stage":
            st.number_input("Durée (mois)", 2, 12, 6)
            st.number_input("Gratification (€)", 500, 1500, 800)
        
        elif contract == "CDI":
            st.number_input("Salaire annuel brut (€)", 30000, 100000, 45000)
            st.checkbox("Package avantages")
    
    with tab3:
        st.header("Ressources RH")
        st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # Remplacer par une vraie vidéo
        
        with st.expander("📚 Guides Pratiques"):
            st.download_button("Guide d'entretien", data="...", file_name="guide_entretien.pdf")
            st.download_button("Modèle de fiche de poste", data="...", file_name="fiche_poste.docx")

if __name__ == "__main__":
    recruiter_space()
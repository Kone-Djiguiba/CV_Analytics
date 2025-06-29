import streamlit as st

st.set_page_config(
    page_title="CV Analytics Pro",
    page_icon="🏠",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .big-font { font-size:18px !important; }
    .feature-card {
        padding: 20px;
        border-radius: 10px;
        background: #f0f2f6;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏠 CV Analytics Pro")
st.subheader("Plateforme Intelligente de Recrutement par IA")

# Bannière
st.image("https://via.placeholder.com/1200x400?text=Optimisez+Votre+Recrutement", use_container_width=True)

# Features
st.markdown("---")
st.header("🚀 Fonctionnalités Clés")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="feature-card"><h3>📊 Analyse CV</h3><p class="big-font">Extraction automatique des compétences et expériences</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-card"><h3>🔍 Filtres Intelligents</h3><p class="big-font">Trouvez les candidats parfaitement adaptés</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="feature-card"><h3>📈 Tableaux de Bord</h3><p class="big-font">Visualisation claire des résultats</p></div>', unsafe_allow_html=True)

# Navigation
st.markdown("---")
st.header("📌 Accès Rapide")

nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("📊 Analyse de CV", use_container_width=True):
        st.switch_page("pages/1_📊_Analyse_CV.py")
with nav_cols[1]:
    if st.button("👔 Espace Recruteur", use_container_width=True):
        st.switch_page("pages/2_👔_Espace_Recruteur.py")
with nav_cols[2]:
    if st.button("ℹ️ À Propos", use_container_width=True):
        st.switch_page("pages/3_ℹ️_A_Propos.py")
with nav_cols[3]:
    if st.button("⚙️ Démo Rapide", use_container_width=True):
        st.switch_page("pages/1_📊_Analyse_CV.py")
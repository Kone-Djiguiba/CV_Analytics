import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import spacy
import re
import pandas as pd
from datetime import datetime
import os
import hashlib
import tempfile

st.set_page_config(
    page_title="Analyse de CV",
    page_icon="📊",
    layout="wide"
)

# ------------------------- Initialisation
if not os.path.exists("cv_data.csv"):
    pd.DataFrame().to_csv("cv_data.csv")

# ------------------------- Chargement modèle SpaCy
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.error("Modèle SpaCy manquant. Exécutez : python -m spacy download en_core_web_sm")
        st.stop()

nlp = load_nlp_model()

# ------------------------- Configuration
SKILL_KEYWORDS = [
    "python", "java", "sql", "machine learning", "gestion de projet",
    "leadership", "analyse de données", "power bi", "tableau"
]

# ------------------------- Fonctions principales
def extract_text_from_pdf(file_bytes):
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def extract_text_from_docx(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path)
    finally:
        os.unlink(tmp_path)
    return text

def extract_skills(text):
    return [skill for skill in SKILL_KEYWORDS if skill.lower() in text.lower()]

def extract_experience(text):
    """Extrait le nombre d'années d'expérience du texte du CV"""
    patterns = [
        r"(\d+)\s*(ans|années|years|yr|yrs)\s+d['']expérience",
        r"expérience\s*:\s*(\d+)\s*(ans|années|years)",
        r"(\d+)\s*(ans|années|years)\s+d['']exp",
        r"expérience\s*professionnelle\s*:\s*(\d+)"
    ]
    
    max_exp = 0
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                years = int(match.group(1))
                if years > max_exp:
                    max_exp = years
            except (ValueError, IndexError):
                continue
    
    if max_exp == 0:
        date_matches = re.findall(r"(20\d{2}|19\d{2})[\s\-–àà\w]*(20\d{2}|19\d{2}|présent|aujourd'hui)", text, re.IGNORECASE)
        if date_matches:
            current_year = datetime.now().year
            earliest_year = current_year
            for start, end in date_matches:
                try:
                    start_year = int(start)
                    if end.lower() in ["présent", "aujourd'hui"]:
                        end_year = current_year
                    else:
                        end_year = int(end)
                    
                    if start_year < earliest_year:
                        earliest_year = start_year
                except ValueError:
                    continue
            
            max_exp = current_year - earliest_year
    
    return max_exp if max_exp > 0 else 1

def extract_languages(text):
    """Extrait les langues mentionnées dans le CV"""
    language_keywords = ["anglais", "français", "espagnol", "allemand", "italien", 
                        "english", "french", "spanish", "german", "italian"]
    
    found_languages = []
    for lang in language_keywords:
        if re.search(rf"\b{lang}\b", text, re.IGNORECASE):
            found_languages.append(lang.capitalize())
    
    return found_languages if found_languages else ["Non spécifié"]

def calculate_match_score(candidate, req_skills, min_exp, req_langs):
    """Calcule un pourcentage de correspondance"""
    total_score = 0
    
    # Compétences (50% du score)
    matched_skills = sum(1 for skill in req_skills if skill.lower() in [s.lower() for s in candidate['Compétences']])
    skill_score = (matched_skills / len(req_skills)) * 50 if req_skills else 0
    
    # Expérience (30% du score)
    exp_score = 30 if candidate['Expérience'] >= min_exp else (
        (candidate['Expérience'] / min_exp) * 30 if min_exp > 0 else 0
    )
    
    # Langues (20% du score)
    matched_langs = sum(1 for lang in req_langs if lang.lower() in [l.lower() for l in candidate['Langues']])
    lang_score = (matched_langs / len(req_langs)) * 20 if req_langs else 0
    
    total_score = skill_score + exp_score + lang_score
    return round(min(total_score, 100))  # Limite à 100%

def show_results(data, req_skills, min_exp, req_langs):
    df = pd.DataFrame(data)
    
    # Ajout du score de correspondance
    df['Score'] = df.apply(
        lambda x: calculate_match_score(x, req_skills, min_exp, req_langs),
        axis=1
    )
    
    # Tri par score décroissant
    df = df.sort_values('Score', ascending=False)
    
    # Filtrage strict (pour information)
    filtered = df[
        df["Compétences"].apply(lambda x: any(skill.lower() in [s.lower() for s in x] for skill in req_skills)) &
        (df["Expérience"] >= min_exp) &
        df["Langues"].apply(lambda x: any(lang.lower() in [l.lower() for l in x] for lang in req_langs))
    ]
    
    # Affichage des critères de recherche
    with st.expander("🔍 Critères de Recherche Actuels", expanded=True):
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Compétences Requises**")
            st.write(", ".join(req_skills) or "Aucune spécifiée")
        with cols[1]:
            st.markdown("**Expérience Minimum**")
            st.write(f"{min_exp} ans")
        with cols[2]:
            st.markdown("**Langues Requises**")
            st.write(", ".join(req_langs) or "Aucune spécifiée")
    
    # Résumé des correspondances
    st.markdown("---")
    if len(filtered) > 0:
        st.success(f"✅ {len(filtered)} CV correspondent parfaitement aux critères")
    else:
        st.warning("ℹ️ Aucun CV ne correspond parfaitement aux critères")
    
    # Affichage de tous les CV avec score
    st.subheader("📋 Tous les CV Analysés")
    for _, row in df.iterrows():
        with st.expander(f"{row['Nom']} - {row['Fichier']} (Score: {row['Score']}%)", expanded=False):
            cols = st.columns([1, 3])
            
            # Indicateur visuel
            with cols[0]:
                if row['Score'] >= 70:
                    st.success("Bonne correspondance")
                elif row['Score'] >= 30:
                    st.warning("Correspondance partielle")
                else:
                    st.error("Faible correspondance")
                
                st.metric("Score Global", f"{row['Score']}%")
            
            # Détails du CV
            with cols[1]:
                st.markdown(f"**Compétences:** {', '.join(row['Compétences'])}")
                st.markdown(f"**Expérience:** {row['Expérience']} ans")
                st.markdown(f"**Langues:** {', '.join(row['Langues'])}")
                
                # Boutons d'action
                st.download_button(
                    "📥 Télécharger l'analyse",
                    data=pd.DataFrame([row]).to_csv(index=False),
                    file_name=f"analyse_{row['Nom']}.csv",
                    key=f"dl_{row['Hash']}"
                )

def analyze_cvs():
    st.title("📊 Analyse Automatique de CV")
    
    # Section critères
    with st.expander("🔍 Paramètres d'analyse", expanded=True):
        cols = st.columns(3)
        with cols[0]:
            skills = st.multiselect("Compétences requises", SKILL_KEYWORDS, ["python", "sql"])
        with cols[1]:
            exp = st.slider("Expérience minimum (ans)", 0, 20, 2)
        with cols[2]:
            langues = st.multiselect("Langues", ["Anglais", "Français", "Espagnol"])
    
    # Upload fichiers
    files = st.file_uploader("Téléverser des CV (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    
    if files:
        process_files(files, skills, exp, langues)

def process_files(files, required_skills, min_exp, required_langs):
    progress_bar = st.progress(0)
    results = []
    
    for i, file in enumerate(files):
        try:
            file_bytes = file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
            
            text = extract_text_from_pdf(file_bytes) if file.name.endswith(".pdf") else extract_text_from_docx(file_bytes)
            
            # Extraction des entités
            doc = nlp(text)
            name = next((ent.text for ent in doc.ents if ent.label_ in ["PERSON", "PER"]), "")
            skills = extract_skills(text)
            
            results.append({
                "Nom": name,
                "Fichier": file.name,
                "Compétences": skills,
                "Expérience": extract_experience(text),
                "Langues": extract_languages(text),
                "Hash": file_hash
            })
            
            progress_bar.progress((i + 1) / len(files))
            
        except Exception as e:
            st.error(f"Erreur avec {file.name}: {str(e)}")
    
    if results:
        show_results(results, required_skills, min_exp, required_langs)

if __name__ == "__main__":
    analyze_cvs()
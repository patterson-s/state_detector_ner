import streamlit as st
import spacy
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

# Set up paths relative to current file
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "data/models/model-best"
ISO_MAPPING_FILE = BASE_DIR / "data/mappings/iso_match_master_01.jsonl"

# Load model & mapping at startup
nlp = spacy.load(MODEL_PATH)
iso_mapping_df = pd.read_json(ISO_MAPPING_FILE, lines=True)
iso_mapping: Dict[str, str] = dict(zip(iso_mapping_df['pattern'], iso_mapping_df['ISO_Code']))

def convert_to_iso_codes(entities: List[str], iso_dict: Dict[str, str]) -> List[str]:
    return [iso_dict.get(ent.strip(), f"UNKNOWN_{ent}") for ent in entities]

def clean_iso_codes(iso_codes: List[str]) -> List[str]:
    return [code for code in iso_codes if not code.startswith('UNKNOWN_')]

def detect_states_in_text(text: str) -> Tuple[List[str], List[str], List[str]]:
    doc = nlp(text)
    gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    iso_codes = convert_to_iso_codes(gpe_entities, iso_mapping)
    cleaned_codes = clean_iso_codes(iso_codes)
    return gpe_entities, iso_codes, cleaned_codes

def main() -> None:
    st.title("State Detection NER Demo")
    st.markdown("""
    **Scott Robert Patterson**  
    
    This model identifies state mentions in text and standardizes them to 3-letter ISO codes. 
    Built on spaCy's GPE NER model with custom annotations from the UNGDC corpus using prodigy.
    """)

    user_input = st.text_area("Enter some text:", height=150)
    if st.button("Submit") and user_input:
        gpe_entities, iso_codes, cleaned_codes = detect_states_in_text(user_input)

        st.subheader("Extracted Entities")
        st.write(gpe_entities if gpe_entities else "No GPE entities detected.")

        st.subheader("All Country Codes")
        st.write(iso_codes if iso_codes else "No ISO codes detected.")

        st.subheader("Cleaned Country Codes")
        st.write(cleaned_codes if cleaned_codes else "No valid ISO codes found.")

if __name__ == "__main__":
    main()
import streamlit as st
from streamlit_option_menu import option_menu
from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.vocab import Vocab
from medcat.config import Config
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import requests
import torch
import json

# -------------------------
# Streamlit App Configuration
# -------------------------
st.set_page_config(
    page_title="Chief Complaint Analysis Tool",
    page_icon="🩺",
    layout="wide",
)

# -------------------------
# Load Resources: MedCAT and BERT
# -------------------------
@st.cache_resource
def load_medcat_model_with_filters():
    cdb = CDB.load("data_p4.2/cdb-medmen-v1.dat")
    vocab = Vocab.load("data_p4.2/vocab.dat")
    config = Config()
    cat = CAT(cdb=cdb, config=config, vocab=vocab)
    
    # Apply TUI Filters
    tui_filter = ['T047', 'T048']  # Detect only diseases
    cui_filters = set()
    for tui in tui_filter:
        cui_filters.update(cdb.addl_info['type_id2cuis'][tui])
    cdb.config.linking['filters']['cuis'] = cui_filters

    return cat

@st.cache_resource
def load_bert_model_and_tokenizer():
    output_dir = "./medical_specialty_model"
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    
    # Load specialty mapping
    with open(f"{output_dir}/specialty_mapping.json", "r") as f:
        label_to_specialty = json.load(f)

    return model, tokenizer, label_to_specialty

cat = load_medcat_model_with_filters()
bert_model, bert_tokenizer, label_to_specialty = load_bert_model_and_tokenizer()

# -------------------------
# Helper Functions
# -------------------------
def predict_specialty(text, model, tokenizer, label_to_specialty):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_label].item()
    specialty = label_to_specialty[str(predicted_label)]
    return specialty, confidence

def get_tgt(api_key):
    auth_endpoint = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
    params = {'apikey': api_key}
    response = requests.post(auth_endpoint, data=params)
    if response.status_code == 201:
        return response.headers['location']
    else:
        raise Exception(f"Failed to get TGT: {response.status_code}, {response.text}")

def get_service_ticket(tgt):
    params = {'service': 'http://umlsks.nlm.nih.gov'}
    response = requests.post(tgt, data=params)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to get service ticket: {response.status_code}, {response.text}")

def map_cuis(api_key, cui_list, sabs, api_version="current"):
    tgt = get_tgt(api_key)
    results = {}
    for cui in cui_list:
        try:
            base_url = f"https://uts-ws.nlm.nih.gov/rest/content/{api_version}/CUI/{cui}/atoms"
            params = {'ticket': get_service_ticket(tgt), 'sabs': sabs}
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                mappings = response.json()
                results[cui] = [
                    {
                        "name": atom["name"],
                        "code": atom["code"].split("/")[-1],
                        "source": atom["rootSource"],
                    }
                    for atom in mappings["result"]
                    if atom["rootSource"] == "ICD10CM"
                ]
            else:
                results[cui] = []
        except Exception:
            results[cui] = []
    return results

# -------------------------
# Navigation Bar
# -------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Analyze", "How to Use", "About"],
        icons=["house", "activity", "book", "info-circle"],
        menu_icon="stethoscope",
        default_index=0,
    )

# -------------------------
# Pages
# -------------------------
if selected == "Home":
    st.title("🩺 Chief Complaint Analysis Tool")
    st.markdown("""
    Welcome to the *Chief Complaint Analysis Tool*!  
    This app leverages *MedCAT* for disease prediction and a fine-tuned *BERT model* for medical specialty classification.
    """)

elif selected == "Analyze":
    st.title("📊 Analyze Chief Complaints")
    chief_complaint = st.text_area(
        "Chief Complaint",
        placeholder="E.g., 'Patient experiencing severe headaches and hypertension.'",
        height=200,
    )
    if st.button("Analyze"):
        if chief_complaint.strip():
            st.info("🔍 Processing the input...")

            # Predict Medical Specialty
            specialty, confidence = predict_specialty(chief_complaint, bert_model, bert_tokenizer, label_to_specialty)
            st.subheader(f"🏥 Predicted Specialty: *{specialty}* ({confidence:.2%} confidence)")

            # MedCAT Entity Extraction
            entities_data = cat.get_entities(chief_complaint)
            if "entities" in entities_data:
                entity_list = []
                api_key = "5f75a5ef-fe8f-46dd-8bf8-e1d67ee9f30c"  # Replace with your UMLS API key
                cui_list = [v["cui"] for v in entities_data["entities"].values()]
                icd10_mappings = map_cuis(api_key, cui_list, sabs="ICD10CM")

                for entity_key, entity_value in entities_data["entities"].items():
                    pretty_name = entity_value.get("pretty_name", "Unknown")
                    cui = entity_value.get("cui", "N/A")
                    
                    # Retrieve ICD-10CM mappings for the CUI
                    icd10_codes = [
                        f"{mapping['code']} ({mapping['name']})"
                        for mapping in icd10_mappings.get(cui, [])
                    ]
                    icd10_codes_str = ", ".join(icd10_codes) if icd10_codes else "N/A"

                    entity_list.append({"Name": pretty_name, "CUI": cui, "ICD-10CM": icd10_codes_str})

                # Display results
                if entity_list:
                    st.markdown("### Detected Diseases:")
                    df = pd.DataFrame(entity_list)
                    st.table(df)
                else:
                    st.warning("⚠ No diseases detected.")
            else:
                st.warning("⚠ No diseases detected.")
        else:
            st.error("⚠ Please enter a chief complaint.")

elif selected == "How to Use":
    st.title("ℹ How to Use")
    st.markdown("""
    This app helps analyze free-text clinical complaints to predict:
    1. *Medical Specialty*: Using a fine-tuned BERT model.
    2. *Diseases and ICD-10CM Codes*: Using MedCAT and UMLS.
    """)

elif selected == "About":
    st.title("About")
    st.markdown("""
    This tool uses advanced natural language processing models to analyze chief complaints in clinical documents, identifying relevant medical specialties and diseases.  
    Created with MedCAT and BERT for medical NLP tasks.
    """)

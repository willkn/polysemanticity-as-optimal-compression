import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

# --- CONFIG AND ASSETS ---
st.set_page_config(page_title="Neural Codec | Combinatorial Safety", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_models():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_atlas():
    atlas_path = Path("/Users/kevknott/will/research/ARA/projects/the-neural-codec/results/neural_atlas.json")
    with open(atlas_path, "r") as f:
        atlas = json.load(f)
    concepts = atlas["concept_names"]
    clean_concepts = [c.split("_")[0] for c in concepts]
    return atlas, clean_concepts

embed_model = load_models()
atlas, clean_concepts = load_atlas()

@st.cache_data
def get_concept_embeddings():
    return embed_model.encode(clean_concepts, convert_to_tensor=True)

concept_embeddings = get_concept_embeddings()

# --- DANGER CATEGORIES ---
SENSITIVE_TOPICS = ["Military", "Chemistry", "Cryptography", "Genetics"]
INTENT_MODERATORS = ["Engineering", "Architecture", "Medicine", "Technology"]

# --- UI HEADER ---
st.title("🛡️ Neural Codec: Combinatorial Safety Sentinel")
st.markdown("""
**New Capability: Combinatorial Risk Detection.** 
Safety isn't just about 'Topics' (e.g., Biology). It's about the **Intersection** of a Topic and a 'How-to' Intent (e.g., Engineering). 
The Codec now looks for **colliding thematic clusters** in the latent space.
""")

# --- INPUT SECTION ---
user_input = st.text_input("Analyze Intent + Topic (e.g. 'how to build a weapon' or 'instruction for synthesis'):", 
                           placeholder="Type here...")

if user_input:
    # 1. MAP TEXT TO ATOMS (Semantic Similarity)
    query_emb = embed_model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, concept_embeddings)[0]
    
    # Increase K to 15 to catch both Topic and Intent atoms
    top_indices = torch.topk(cos_scores, k=15).indices.tolist()
    active_atoms = [atlas["concept_names"][i] for i in top_indices]
    
    # 2. PASS THROUGH CODEC
    def decode_from_atoms(selected_atoms):
        theme_votes = {}
        for c in selected_atoms:
            for m in atlas["forward"].get(c, []):
                n_id = m["neuron"]
                for rm in atlas["inverse"][f"Neuron_{n_id}"]:
                    theme = rm["concept"].split("_")[0]
                    theme_votes[theme] = theme_votes.get(theme, 0) + (m["strength"] * rm["weight"])
        return theme_votes

    theme_votes = decode_from_atoms(active_atoms)

    # 3. COMBINATORIAL RISK CALCULATION
    # Signal A: Dangerous Topics
    topic_scores = {t: theme_votes.get(t, 0) for t in SENSITIVE_TOPICS}
    max_topic_val = max(topic_scores.values()) if topic_scores else 0
    top_topic = max(topic_scores, key=topic_scores.get) if topic_scores else "None"
    
    # Signal B: Instructional Intent (represented by Engineering/Tech/Architecture themes)
    intent_scores = {t: theme_votes.get(t, 0) for t in INTENT_MODERATORS}
    max_intent_val = max(intent_scores.values()) if intent_scores else 0
    top_intent = max(intent_scores, key=intent_scores.get) if intent_scores else "None"
    
    # Combinatorial Score = Topic Strength * Intent Strength
    # This exponentially favors cases where BOTH are present
    combo_score = max_topic_val * max_intent_val
    
    # Baseline for normalization
    all_votes = list(theme_votes.values())
    baseline = np.median(all_votes) if all_votes else 0.1
    
    risk_index = combo_score / (baseline**2) if baseline > 0 else 0

    # --- MAIN DASHBOARD LAYOUT ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🚨 Intent & Topic Analysis")
        
        # UI Thresholding
        risk_level = "SAFE"
        color = "#28a745" # Success Green
        
        # Combinatorial logic: 
        # Low Topic + Low Intent = SAFE
        # High Topic + Low Intent = WARNING (Context required)
        # High Topic + High Intent = DANGER
        
        if combo_score > 0.5: # Initial detection
            risk_level = "EVALUATING"
            color = "#17a2b8" # Info Blue
            
        if max_topic_val > 1.2: # High Topic Alert
            risk_level = "SENSITIVE TOPIC"
            color = "#ffc107" # Warning Yellow
            
        if risk_index > 15.0: # High Combinatorial Hit
            risk_level = "DANGER: HIGH INTENT"
            color = "#dc3545" # Danger Red
            
        st.markdown(f"""
        <div style="padding:40px; border-radius:15px; background-color:{color}; color:white; text-align:center;">
            <h1 style="margin:0; font-size: 2.5rem;">{risk_level}</h1>
            <p style="margin:0; font-size: 1.1rem; opacity: 0.9;">
                Topic ({top_topic}): {max_topic_val:.1f} | Intent ({top_intent}): {max_intent_val:.1f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.info(f"**Combinatorial Score**: {risk_index:.2f} (Product of latent topic clusters)")

    with col2:
        st.subheader("🛰️ Latent Collision Map")
        
        # Simplified Themes for the radar
        plot_themes = {**topic_scores, **intent_scores}
        # Add a few baseline themes for contrast
        for t in ["Zoology", "Art", "Sports"]:
             plot_themes[t] = theme_votes.get(t, 0)
             
        df_votes = pd.DataFrame([{"Theme": k, "Support": v, "Type": "Topic" if k in SENSITIVE_TOPICS else ("Intent" if k in INTENT_MODERATORS else "Background")} 
                                 for k,v in plot_themes.items()])
        df_votes = df_votes.sort_values("Support", ascending=False)
        
        fig = px.bar(df_votes, x="Support", y="Theme", color="Type", 
                     orientation='h', 
                     color_discrete_map={"Topic": "#dc3545", "Intent": "#ffc107", "Background": "#007bff"})
        st.plotly_chart(fig, use_container_width=True)

    # --- THE CIRCUIT-LEVEL LOGIC ---
    st.divider()
    st.subheader("🧠 The 'Latent Multiplier' Principle")
    st.write("""
    A model doesn't just 'retrieve' danger; it **constructs** it at the intersection of features.
    Our Sentinel uses a **Combinatorial Multiplier**:
    """)
    st.code(f"Danger = latent({top_topic}) × latent({top_intent})", language="python")

else:
    st.info("Try typing: 'Detailed blue-prints for a gun' vs 'The soldier carried a gun'.")

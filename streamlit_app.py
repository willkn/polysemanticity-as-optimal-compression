import streamlit as st
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from functools import partial

st.set_page_config(page_title="Neural Codec Steering", layout="wide")

# High-fidelity indices (V42 + V45 Hunts)
THEMES = {
    "Astro": [8870, 6478, 403, 23389, 22931],
    "Botany": [17013, 6642, 8240, 20777, 12859],
    "Methods": [3320, 9097, 1907, 10628], # Renamed 'Military/War' to 'Methods/Politics' based on outputs
    "Biotech": [1586, 100, 200]
}

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre", device=device)
    return model, sae, device

def steer_hook(resid, hook, sae, indices, strength=15.0):
    # 1. Create Sparse Acts Vector
    # We want to add PURE DIRECTION, not reconstruction bias.
    steering_acts = torch.zeros((resid.shape[0], resid.shape[1], sae.cfg.d_sae), device=resid.device)
    for idx in indices:
        steering_acts[:, :, idx] = strength
        
    # 2. Decode manually WITHOUT bias (x = W_dec @ f)
    # sae.decode() adds b_dec, which shifts the manifold by ~50 units, causing incoherence.
    direction = steering_acts @ sae.W_dec
    
    # 3. Add to residual stream
    return resid + direction

def main():
    st.title("🧠 Neural Codec Steering")
    
    model, sae, device = load_models()

    col1, col2 = st.columns(2)
    with col1:
        target_theme = st.selectbox("Select Steering Theme", list(THEMES.keys()))
        # High-precision slider for low strength steering
        strength = st.slider("Steering Strength", 0.0, 10.0, 1.4, 0.1)
        input_text = st.text_area("Input Prompt", "The scientists were investigating the", height=100)
        num_tokens = st.slider("Tokens to Generate", 10, 60, 30)
        gen_button = st.button("Steer & Compare")

    if gen_button:
        # 1. Baseline
        model.reset_hooks()
        with col1:
            st.subheader("Neutral (Baseline)")
            baseline_out = model.generate(input_text, max_new_tokens=num_tokens, verbose=False)
            st.text_area("Baseline Output", baseline_out, height=150, disabled=True)
        
        # 2. Steer
        indices = THEMES[target_theme]
        hook_fn = partial(steer_hook, sae=sae, indices=indices, strength=strength)
        
        for layer in [6, 7]:
            model.add_hook(f"blocks.{layer}.hook_resid_pre", hook_fn)
        
        with col2:
            st.subheader(f"Steered: {target_theme}")
            steered_out = model.generate(input_text, max_new_tokens=num_tokens, verbose=False)
            st.success(steered_out)
            
        model.reset_hooks()
        
        # Atom Visualization
        st.divider()
        cols = st.columns(min(len(indices), 8))
        for i, idx in enumerate(indices[:8]):
            cols[i].metric(label=f"Atom {idx}", value="BOOST", delta=f"{strength}x")

if __name__ == "__main__":
    main()

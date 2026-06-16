"""
CLIP Interactive Implementation Walkthrough
SEAS 8525: Computer Vision and Generative AI
Dr. Elbasheer -- Week 6

Setup:
    pip install streamlit torch torchvision transformers matplotlib seaborn
    pip install scikit-learn pillow numpy

Run:
    streamlit run clip_app.py

On Google Colab:
    !pip install streamlit pyngrok transformers torch torchvision
    !pip install matplotlib seaborn scikit-learn pillow
    !streamlit run clip_app.py &
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    print(public_url)
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.manifold import TSNE

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLIP Walkthrough | SEAS 8525",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SIDEBAR WIDTH STATE ──────────────────────────────────────────────────────
if "sidebar_width" not in st.session_state:
    st.session_state.sidebar_width = 280

# ── THEME: light, high-contrast, classroom-friendly ──────────────────────────
st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{
        min-width: {st.session_state.sidebar_width}px !important;
        max-width: {st.session_state.sidebar_width}px !important;
        width: {st.session_state.sidebar_width}px !important;
        background: #1A2F4A;
        border-right: 2px solid #0D2B4E;
        transition: width 0.15s ease;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Base */
    [data-testid="stAppViewContainer"] { background: #F7F9FC; color: #1A2332; }
    [data-testid="stSidebar"] {
        background: #1A2F4A;
        border-right: 2px solid #0D2B4E;
        transition: width 0.15s ease;
    }
    [data-testid="stSidebar"] * { color: #E8EFF7 !important; }
    [data-testid="stSidebar"] .stCodeBlock,
    [data-testid="stSidebar"] pre,
    [data-testid="stSidebar"] code {
        background: #0A1628 !important;
        color: #80CBC4 !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; }

    /* Main text */
    h1 { color: #0D47A1 !important; font-family: Georgia, serif; }
    h2 { color: #1565C0 !important; font-family: Georgia, serif; }
    h3 { color: #1976D2 !important; }
    p, li { color: #1A2332; font-size: 15px; line-height: 1.7; }

    /* Info boxes */
    .concept-box {
        background: #E3F2FD;
        border-left: 5px solid #0097A7;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #0D2B4E;
        font-size: 14px;
        line-height: 1.7;
    }
    .highlight-box {
        background: #EDE7F6;
        border-left: 5px solid #5E35B1;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #1A0050;
        font-size: 14px;
        line-height: 1.7;
    }
    .warning-box {
        background: #FFF8E1;
        border-left: 5px solid #F9A825;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #3E2723;
        font-size: 13px;
    }
    .success-box {
        background: #E8F5E9;
        border-left: 5px solid #2E7D32;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #1B3A1C;
        font-size: 13px;
    }
    .metric-card {
        background: #FFFFFF;
        border: 2px solid #BBDEFB;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 6px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .metric-val {
        font-size: 26px;
        font-weight: bold;
        color: #0D47A1;
    }
    .metric-label {
        font-size: 12px;
        color: #607D8B;
        margin-top: 4px;
    }
    .section-tag {
        display: inline-block;
        background: #0097A7;
        color: white;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    /* Code blocks: white background, readable */
    .stCodeBlock, pre, code {
        background: #F1F5F9 !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px;
    }
    .stButton > button {
        background: #1565C0;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        padding: 8px 20px;
    }
    .stButton > button:hover { background: #1976D2; }
    hr { border-color: #BBDEFB; }
    [data-testid="stFileUploader"] { background: #FFFFFF; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, processor, device


# ── HELPERS ───────────────────────────────────────────────────────────────────
def encode_images(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    # newer transformers versions return BaseModelOutputWithPooling instead of a plain Tensor
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def encode_texts(model, processor, texts, device):
    inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def cosine_sim_matrix(a, b):
    return np.dot(a, b.T)


def make_fig(w=8, h=5):
    """Light-theme matplotlib figure for classroom projector."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7F9FC")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CBD5E1")
    ax.tick_params(colors="#334155", labelsize=10)
    ax.xaxis.label.set_color("#334155")
    ax.yaxis.label.set_color("#334155")
    ax.title.set_color("#0D47A1")
    return fig, ax


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.session_state.sidebar_width = st.slider(
        "Panel width", min_value=200, max_value=420,
        value=st.session_state.sidebar_width, step=10,
        help="Drag to resize the sidebar"
    )
    st.markdown("## 🔗 CLIP Walkthrough")
    st.markdown("**SEAS 8525** · Computer Vision and Generative AI")
    st.markdown("*Dr. Elbasheer · Week 6*")
    st.divider()

    section = st.radio(
        "Jump to section",
        options=[
            "1 · What is CLIP?",
            "2 · The Two Encoders",
            "3 · Encoding Images and Text",
            "4 · The NxN Similarity Matrix",
            "5 · Zero-Shot Classification",
            "6 · Embedding Space Visualization",
        ]
    )
    st.divider()

    st.markdown("#### Quick Setup")
    st.code(
        "pip install streamlit torch torchvision\n"
        "pip install transformers matplotlib\n"
        "pip install seaborn scikit-learn pillow\n\n"
        "streamlit run clip_app.py",
        language="bash"
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cuda":
        st.success(f"GPU ready: {torch.cuda.get_device_name(0)}")
    else:
        st.info("Running on CPU. Each inference call takes a few seconds.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: WHAT IS CLIP?
# ═════════════════════════════════════════════════════════════════════════════
if section.startswith("1"):
    st.markdown('<span class="section-tag">SECTION 1</span>', unsafe_allow_html=True)
    st.title("CLIP: Contrastive Language-Image Pretraining")
    st.caption("Radford et al., OpenAI 2021 · Model: openai/clip-vit-base-patch32")

    st.markdown("""<div class="concept-box">
    <strong>Core idea:</strong> Train an image encoder and a text encoder jointly so that 
    matching image-text pairs end up close together in a shared embedding space, while 
    non-matching pairs are pushed far apart. Trained on 400 million image-text pairs 
    scraped from the internet.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-val">400M</div><div class="metric-label">Training pairs</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-val">512</div><div class="metric-label">Embedding dimension</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-val">ViT-B/32</div><div class="metric-label">Image encoder</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-val">76.2%</div><div class="metric-label">ImageNet zero-shot</div></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Architecture Overview")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""<div class="highlight-box">
        <strong>Image Encoder (ViT-B/32)</strong><br><br>
        Takes a 224x224 RGB image.<br>
        Splits into 32x32 patches (49 patches total).<br>
        Runs through 12 transformer encoder layers.<br>
        Outputs a single <strong>512-dimensional</strong> embedding vector.
        </div>""", unsafe_allow_html=True)

        st.code(
            "from transformers import CLIPModel, CLIPProcessor\n\n"
            "model = CLIPModel.from_pretrained(\n"
            '    "openai/clip-vit-base-patch32"\n'
            ")\n"
            "processor = CLIPProcessor.from_pretrained(\n"
            '    "openai/clip-vit-base-patch32"\n'
            ")\n\n"
            "# Encode an image\n"
            "inputs = processor(images=image, return_tensors='pt')\n"
            "image_features = model.get_image_features(**inputs)\n"
            "# get_image_features returns a plain Tensor [1, 512]\n\n"
            "# Normalize to unit sphere for cosine similarity\n"
            "image_features = image_features / \\\n"
            "    image_features.norm(dim=-1, keepdim=True)\n\n"
            "print(image_features.shape)  # torch.Size([1, 512])",
            language="python"
        )

    with col_r:
        st.markdown("""<div class="highlight-box">
        <strong>Text Encoder (Transformer)</strong><br><br>
        Takes a text string up to 77 tokens.<br>
        Tokenizes using byte-pair encoding (BPE).<br>
        Runs through 12 transformer layers.<br>
        Outputs a single <strong>512-dimensional</strong> embedding vector.
        </div>""", unsafe_allow_html=True)

        st.code(
            "# Encode text\n"
            "inputs = processor(\n"
            '    text=["a photo of a dog"],\n'
            "    return_tensors='pt',\n"
            "    padding=True\n"
            ")\n"
            "text_features = model.get_text_features(**inputs)\n"
            "# get_text_features returns a plain Tensor [1, 512]\n\n"
            "# Normalize\n"
            "text_features = text_features / \\\n"
            "    text_features.norm(dim=-1, keepdim=True)\n\n"
            "# Cosine similarity is just the dot product\n"
            "# after L2 normalization\n"
            "similarity = (image_features @ text_features.T)\n"
            "print(similarity)  # value between -1 and +1",
            language="python"
        )

    st.divider()
    st.subheader("The Training Objective: InfoNCE Loss")

    st.markdown("""<div class="concept-box">
    Given a batch of N image-text pairs, compute an NxN cosine similarity matrix.
    The diagonal holds the N correct (matching) pairs. Everything off the diagonal is a wrong pair.<br><br>
    The loss maximizes diagonal similarities and minimizes off-diagonal similarities 
    simultaneously, treating each row and each column as a separate N-way classification problem.
    </div>""", unsafe_allow_html=True)

    st.code(
        "import torch\n"
        "import torch.nn.functional as F\n\n"
        "def clip_loss(image_features, text_features, temperature=0.07):\n"
        "    # Both inputs: [N, 512], already L2-normalized\n\n"
        "    # NxN cosine similarity matrix\n"
        "    logits = (image_features @ text_features.T) / temperature\n"
        "    # logits shape: [N, N]\n"
        "    # diagonal = correct pairs\n"
        "    # off-diagonal = wrong pairs\n\n"
        "    # Labels: image i matches text i\n"
        "    labels = torch.arange(len(image_features))\n\n"
        "    # Image side: for each image, find its matching text\n"
        "    loss_i2t = F.cross_entropy(logits, labels)\n\n"
        "    # Text side: for each text, find its matching image\n"
        "    loss_t2i = F.cross_entropy(logits.T, labels)\n\n"
        "    # Symmetric -- average both directions\n"
        "    return (loss_i2t + loss_t2i) / 2",
        language="python"
    )

    st.markdown("""<div class="warning-box">
    <strong>Temperature parameter:</strong> The temperature (0.07 in CLIP) controls how 
    peaked the similarity distribution is. A lower temperature makes the model more 
    confident. CLIP learns the temperature as a trainable parameter during training.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: THE TWO ENCODERS
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("2"):
    st.markdown('<span class="section-tag">SECTION 2</span>', unsafe_allow_html=True)
    st.title("The Two Encoders")
    st.caption("Architecture details of the image and text encoders in CLIP")

    with st.spinner("Loading CLIP model..."):
        model, processor, device = load_clip()

    st.success("Model loaded: openai/clip-vit-base-patch32")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Image Encoder: ViT-B/32")
        st.markdown("""<div class="concept-box">
        <strong>ViT-B/32</strong> means: Base model size, 32x32 pixel patches.<br><br>
        A 224x224 image splits into (224/32)^2 = 49 patches.<br>
        Each patch: 32x32x3 = 3072 values, projected to 768 dimensions.<br>
        A learnable CLS token is prepended: 50 tokens total.<br>
        12 transformer layers process the sequence.<br>
        CLS token output is projected down to 512 dimensions.
        </div>""", unsafe_allow_html=True)

        st.code(
            "vc = model.config.vision_config\n\n"
            "print('Patch size:  ', vc.patch_size)        # 32\n"
            "print('Image size:  ', vc.image_size)        # 224\n"
            "print('Hidden size: ', vc.hidden_size)       # 768\n"
            "print('Num layers:  ', vc.num_hidden_layers) # 12\n"
            "print('Num heads:   ', vc.num_attention_heads) # 12\n"
            "print('Output dim:  ', model.config.projection_dim) # 512",
            language="python"
        )

        if st.button("Run: Show image encoder config"):
            vc = model.config.vision_config
            st.json({
                "patch_size": vc.patch_size,
                "image_size": vc.image_size,
                "hidden_size": vc.hidden_size,
                "num_layers": vc.num_hidden_layers,
                "num_attention_heads": vc.num_attention_heads,
                "intermediate_size": vc.intermediate_size,
                "projection_dim": model.config.projection_dim,
            })
            total = sum(p.numel() for p in model.vision_model.parameters())
            st.markdown(f"""<div class="success-box">
            Image encoder parameters: <strong>{total:,}</strong> ({total/1e6:.1f}M)
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.subheader("Text Encoder: Transformer")
        st.markdown("""<div class="concept-box">
        A standard transformer encoder for tokenized text.<br><br>
        Vocabulary: 49,408 BPE tokens.<br>
        Maximum sequence length: 77 tokens.<br>
        Text is wrapped: [SOT] tokens [EOT].<br>
        The <strong>EOT token</strong> representation becomes the sentence embedding.<br>
        12 transformer layers, same depth as the image encoder.
        </div>""", unsafe_allow_html=True)

        st.code(
            "tc = model.config.text_config\n\n"
            "print('Vocab size:  ', tc.vocab_size)              # 49408\n"
            "print('Max length:  ', tc.max_position_embeddings) # 77\n"
            "print('Hidden size: ', tc.hidden_size)             # 512\n"
            "print('Num layers:  ', tc.num_hidden_layers)       # 12\n"
            "print('Num heads:   ', tc.num_attention_heads)     # 8\n"
            "print('Output dim:  ', model.config.projection_dim) # 512",
            language="python"
        )

        if st.button("Run: Show text encoder config"):
            tc = model.config.text_config
            st.json({
                "vocab_size": tc.vocab_size,
                "max_position_embeddings": tc.max_position_embeddings,
                "hidden_size": tc.hidden_size,
                "num_layers": tc.num_hidden_layers,
                "num_attention_heads": tc.num_attention_heads,
                "intermediate_size": tc.intermediate_size,
                "projection_dim": model.config.projection_dim,
            })
            total = sum(p.numel() for p in model.text_model.parameters())
            st.markdown(f"""<div class="success-box">
            Text encoder parameters: <strong>{total:,}</strong> ({total/1e6:.1f}M)
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Tokenization: How Text Becomes Numbers")

    sample_text = st.text_input(
        "Enter any sentence to see how CLIP tokenizes it:",
        value="a photo of a golden retriever playing in the park"
    )

    if st.button("Tokenize"):
        tokens = processor.tokenizer(sample_text, return_tensors="pt")
        input_ids = tokens["input_ids"][0].tolist()
        decoded = [processor.tokenizer.decode([t]) for t in input_ids]

        st.markdown(f"""<div class="highlight-box">
        <strong>Token IDs:</strong> {input_ids}<br><br>
        <strong>Decoded tokens:</strong> {decoded}<br><br>
        <strong>Sequence length:</strong> {len(input_ids)} tokens out of 77 max
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="warning-box">
        <strong>Special tokens:</strong> 49406 = [SOT] start of text, 49407 = [EOT] end of text.
        CLIP uses the EOT token's output as the sentence embedding, not an average over all tokens.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: ENCODING IMAGES AND TEXT
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("3"):
    st.markdown('<span class="section-tag">SECTION 3</span>', unsafe_allow_html=True)
    st.title("Encoding Images and Text")
    st.caption("Watch an image and a text description become 512-dimensional vectors")

    with st.spinner("Loading CLIP..."):
        model, processor, device = load_clip()

    st.markdown("""<div class="concept-box">
    Both the image encoder and text encoder output 512-dimensional vectors that are 
    L2-normalized to lie on the unit hypersphere. Cosine similarity between any two 
    vectors is then just their dot product, ranging from -1 (opposite) to +1 (identical direction).
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Upload an Image")
        uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)

    with col_r:
        st.subheader("Describe the Image")
        text_input = st.text_area(
            "Text description:",
            value="a photo of a dog",
            height=100
        )
        st.markdown("""<div class="warning-box">
        <strong>Prompt tip:</strong> CLIP works best with natural language starting with 
        "a photo of..." rather than bare keywords like "dog".
        </div>""", unsafe_allow_html=True)

    if uploaded and st.button("Encode Both and Compare"):
        img = Image.open(uploaded).convert("RGB")

        with st.spinner("Encoding..."):
            img_emb = encode_images(model, processor, [img], device)
            txt_emb = encode_texts(model, processor, [text_input], device)
            similarity = float(np.dot(img_emb[0], txt_emb[0]))

        st.divider()
        st.subheader("Results")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-card"><div class="metric-val">512</div><div class="metric-label">Image embedding dims</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card"><div class="metric-val">512</div><div class="metric-label">Text embedding dims</div></div>', unsafe_allow_html=True)
        with c3:
            color = "#2E7D32" if similarity > 0.25 else "#BF360C" if similarity < 0.1 else "#E65100"
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{similarity:.3f}</div><div class="metric-label">Cosine similarity</div></div>', unsafe_allow_html=True)

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Image Embedding (first 64 dims)")
            fig, ax = make_fig(7, 2.8)
            vals = img_emb[0][:64]
            colors = ["#E53935" if v < 0 else "#1565C0" for v in vals]
            ax.bar(range(64), vals, color=colors, width=0.8)
            ax.axhline(0, color="#94A3B8", linewidth=0.8)
            ax.set_xlabel("Dimension index")
            ax.set_ylabel("Value")
            ax.set_title("Image embedding vector (first 64 of 512 dims)")
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("Text Embedding (first 64 dims)")
            fig, ax = make_fig(7, 2.8)
            vals = txt_emb[0][:64]
            colors = ["#E53935" if v < 0 else "#00897B" for v in vals]
            ax.bar(range(64), vals, color=colors, width=0.8)
            ax.axhline(0, color="#94A3B8", linewidth=0.8)
            ax.set_xlabel("Dimension index")
            ax.set_ylabel("Value")
            ax.set_title("Text embedding vector (first 64 of 512 dims)")
            st.pyplot(fig)
            plt.close()

        box_class = "success-box" if similarity > 0.25 else "warning-box"
        if similarity > 0.25:
            interpretation = "Strong match. The image and text are well aligned in CLIP's shared space."
        elif similarity < 0.15:
            interpretation = "Weak match. Try rephrasing with 'a photo of...' or be more specific."
        else:
            interpretation = "Moderate match. The description partially aligns with the image."

        st.markdown(f"""<div class="{box_class}">
        <strong>Similarity score: {similarity:.4f}</strong><br>
        {interpretation}
        </div>""", unsafe_allow_html=True)

        st.code(
            "# What happened under the hood\n"
            "image_embedding = model.get_image_features(**image_inputs)\n"
            "text_embedding  = model.get_text_features(**text_inputs)\n\n"
            "# Both normalized to unit hypersphere\n"
            "# Cosine similarity = dot product of unit vectors\n"
            f"similarity = float(image_embedding @ text_embedding.T)\n"
            f"# Result: {similarity:.4f}",
            language="python"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: NxN SIMILARITY MATRIX
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("4"):
    st.markdown('<span class="section-tag">SECTION 4</span>', unsafe_allow_html=True)
    st.title("The NxN Contrastive Similarity Matrix")
    st.caption("The training objective that makes CLIP work")

    with st.spinner("Loading CLIP..."):
        model, processor, device = load_clip()

    st.markdown("""<div class="concept-box">
    This is the heart of CLIP training. Given N image-text pairs, compute every pairwise 
    cosine similarity. The diagonal should be high (matching pairs). Everything off the 
    diagonal should be low (non-matching pairs). The InfoNCE loss enforces exactly this.
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Build Your Own Similarity Matrix")
    st.markdown("Upload images and provide one matching description per image, then see the live matrix.")

    n_items = st.slider("Number of image-text pairs", min_value=2, max_value=6, value=4)

    default_texts = [
        "a photo of a cat",
        "a photo of a car",
        "a photo of a pizza",
        "a photo of a beach",
        "a photo of a mountain",
        "a photo of a dog",
    ]

    uploaded_images = []
    text_descriptions = []
    all_ready = True

    cols = st.columns(n_items)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Pair {i+1}**")
            img_file = st.file_uploader(
                f"Image {i+1}", type=["png","jpg","jpeg","webp"], key=f"img_{i}"
            )
            txt = st.text_input(f"Description {i+1}", value=default_texts[i], key=f"txt_{i}")
            if img_file:
                img = Image.open(img_file).convert("RGB")
                st.image(img, use_container_width=True)
                uploaded_images.append(img)
                text_descriptions.append(txt)
            else:
                all_ready = False

    st.divider()

    if all_ready and len(uploaded_images) == n_items:
        if st.button("Compute Similarity Matrix"):
            with st.spinner("Encoding all images and texts..."):
                img_embs = encode_images(model, processor, uploaded_images, device)
                txt_embs = encode_texts(model, processor, text_descriptions, device)
                sim_matrix = cosine_sim_matrix(img_embs, txt_embs)

            st.subheader("Cosine Similarity Matrix")

            fig, ax = plt.subplots(figsize=(max(6, n_items * 1.5), max(5, n_items * 1.3)))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#F7F9FC")

            cmap = mcolors.LinearSegmentedColormap.from_list(
                "clip_cmap", ["#FFCDD2", "#FFFFFF", "#C8E6C9"], N=256
            )
            im = ax.imshow(sim_matrix, cmap=cmap, vmin=0.0, vmax=0.6)

            short_labels = [t[:22] + "..." if len(t) > 22 else t for t in text_descriptions]
            ax.set_xticks(range(n_items))
            ax.set_yticks(range(n_items))
            ax.set_xticklabels(short_labels, rotation=30, ha="right", color="#334155", fontsize=10)
            ax.set_yticklabels([f"Image {i+1}" for i in range(n_items)], color="#334155", fontsize=10)
            ax.set_xlabel("Text descriptions", color="#334155")
            ax.set_ylabel("Images", color="#334155")
            ax.set_title("NxN Cosine Similarity Matrix", color="#0D47A1", fontsize=13)

            for i in range(n_items):
                for j in range(n_items):
                    val = sim_matrix[i, j]
                    marker = "* " if i == j else ""
                    ax.text(j, i, f"{marker}{val:.3f}", ha="center", va="center",
                            color="#1A2332", fontsize=11,
                            fontweight="bold" if i == j else "normal")

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors="#334155")

            # Gold borders on diagonal
            for i in range(n_items):
                rect = plt.Rectangle(
                    (i - 0.5, i - 0.5), 1, 1,
                    linewidth=3, edgecolor="#F9A825", facecolor="none"
                )
                ax.add_patch(rect)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            diag_vals = [sim_matrix[i, i] for i in range(n_items)]
            off_diag = [
                sim_matrix[i, j]
                for i in range(n_items)
                for j in range(n_items) if i != j
            ]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#2E7D32">{np.mean(diag_vals):.3f}</div><div class="metric-label">Mean diagonal (matching pairs)</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#C62828">{np.mean(off_diag):.3f}</div><div class="metric-label">Mean off-diagonal (non-matching)</div></div>', unsafe_allow_html=True)
            with c3:
                gap = np.mean(diag_vals) - np.mean(off_diag)
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#E65100">{gap:.3f}</div><div class="metric-label">Similarity gap (larger is better)</div></div>', unsafe_allow_html=True)

            st.markdown("""<div class="concept-box">
            <strong>Reading the matrix:</strong> Gold-bordered cells on the diagonal are the 
            correct pairs. The InfoNCE loss during training pushes diagonal values toward +1 
            and off-diagonal values down. The larger the gap between diagonal and off-diagonal, 
            the better CLIP has separated your image-text pairs.
            </div>""", unsafe_allow_html=True)

            st.code(
                f"diagonal_mean     = {np.mean(diag_vals):.4f}  # matching pairs\n"
                f"off_diagonal_mean = {np.mean(off_diag):.4f}  # non-matching pairs\n"
                f"similarity_gap    = {gap:.4f}  # larger = better separation",
                language="python"
            )
    else:
        st.markdown("""<div class="warning-box">
        Upload all images above to compute the similarity matrix.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: ZERO-SHOT CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("5"):
    st.markdown('<span class="section-tag">SECTION 5</span>', unsafe_allow_html=True)
    st.title("Zero-Shot Classification")
    st.caption("Classify any image into any categories you define -- no fine-tuning required")

    with st.spinner("Loading CLIP..."):
        model, processor, device = load_clip()

    st.markdown("""<div class="concept-box">
    At inference time, describe your candidate classes as natural language prompts. 
    Encode both the image and all class prompts. The class whose text embedding is 
    most similar to the image embedding is the prediction. No gradient updates. No labels.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.2])

    with col_l:
        st.subheader("Upload Image")
        uploaded = st.file_uploader("Image to classify", type=["png","jpg","jpeg","webp"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)

    with col_r:
        st.subheader("Define Your Classes")
        st.markdown("Enter one class prompt per line. CLIP will rank them by similarity.")

        default_classes = (
            "a photo of a dog\n"
            "a photo of a cat\n"
            "a photo of a car\n"
            "a photo of food\n"
            "a photo of a building\n"
            "a photo of a person\n"
            "a photo of nature\n"
            "a photo of an animal"
        )

        class_input = st.text_area("Class prompts:", value=default_classes, height=220)

        st.markdown("""<div class="warning-box">
        <strong>Try this:</strong> Replace "a photo of a dog" with just "dog" and compare 
        the scores. The natural sentence framing consistently outperforms bare keywords 
        because CLIP was trained on natural language captions.
        </div>""", unsafe_allow_html=True)

    if uploaded and st.button("Run Zero-Shot Classification"):
        img = Image.open(uploaded).convert("RGB")
        classes = [c.strip() for c in class_input.strip().split("\n") if c.strip()]

        with st.spinner("Classifying..."):
            img_emb = encode_images(model, processor, [img], device)
            txt_embs = encode_texts(model, processor, classes, device)
            scores = cosine_sim_matrix(img_emb, txt_embs)[0]
            # Softmax with temperature for probabilities
            exp_scores = np.exp(scores / 0.07)
            probs = exp_scores / exp_scores.sum()

        sorted_idx = np.argsort(scores)[::-1]

        st.divider()
        st.subheader("Rankings")

        fig, ax = make_fig(8, max(3, len(classes) * 0.6))
        y_pos = list(range(len(classes)))
        sorted_scores = scores[sorted_idx]
        sorted_labels = [classes[i] for i in sorted_idx]
        bar_colors = (
            ["#1B5E20"] +
            ["#1565C0"] * min(2, len(classes)-1) +
            ["#90A4AE"] * max(0, len(classes)-3)
        )

        ax.barh(y_pos, sorted_scores[::-1], color=list(reversed(bar_colors)), height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_labels[::-1], color="#334155", fontsize=11)
        ax.set_xlabel("Cosine similarity score")
        ax.set_title("Zero-Shot Classification Results")
        ax.axvline(0, color="#94A3B8", linewidth=0.8)

        for i, score in enumerate(sorted_scores[::-1]):
            ax.text(score + 0.002, i, f"{score:.3f}", va="center",
                    color="#1A2332", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""<div class="success-box">
        <strong>Top prediction:</strong> {sorted_labels[0]}<br>
        <strong>Cosine similarity:</strong> {sorted_scores[0]:.4f}<br>
        <strong>Softmax probability:</strong> {probs[sorted_idx[0]]:.1%}
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("Prompt Engineering: Does Phrasing Matter?")
        st.markdown("The top predicted class is tested with 5 different phrasings automatically.")

        top_bare = (
            sorted_labels[0]
            .replace("a photo of a ", "")
            .replace("a photo of ", "")
            .replace("a photo ", "")
        )

        prompts_cmp = [
            top_bare,
            f"a photo of {top_bare}",
            f"a photo of a {top_bare}",
            f"an image of {top_bare}",
            f"a picture of {top_bare}",
        ]

        txt_embs_cmp = encode_texts(model, processor, prompts_cmp, device)
        scores_cmp = cosine_sim_matrix(img_emb, txt_embs_cmp)[0]

        fig2, ax2 = make_fig(8, 3.2)
        best_idx = int(np.argmax(scores_cmp))
        bar_colors2 = [
            "#1B5E20" if i == best_idx else "#1565C0"
            for i in range(len(prompts_cmp))
        ]
        ax2.bar(range(len(prompts_cmp)), scores_cmp, color=bar_colors2, width=0.6)
        ax2.set_xticks(range(len(prompts_cmp)))
        ax2.set_xticklabels(prompts_cmp, rotation=15, ha="right",
                             color="#334155", fontsize=10)
        ax2.set_ylabel("Cosine similarity")
        ax2.set_title("How prompt phrasing affects similarity score")
        for i, s in enumerate(scores_cmp):
            ax2.text(i, s + 0.002, f"{s:.3f}", ha="center",
                     color="#1A2332", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown("""<div class="concept-box">
        This is prompt engineering for vision models. The same intuition as prompt engineering 
        for LLMs but applied to image-text similarity. CLIP was trained on natural image captions, 
        so sentence-style prompts consistently outperform bare keywords.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: EMBEDDING SPACE VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("6"):
    st.markdown('<span class="section-tag">SECTION 6</span>', unsafe_allow_html=True)
    st.title("Embedding Space Visualization")
    st.caption("See images and texts cluster together in CLIP's shared 512-dimensional space")

    with st.spinner("Loading CLIP..."):
        model, processor, device = load_clip()

    st.markdown("""<div class="concept-box">
    CLIP's shared embedding space is 512-dimensional. We cannot visualize 512 dimensions 
    directly, so we use t-SNE to project down to 2D while preserving neighborhood structure.
    If CLIP has learned well, image embeddings and their matching text embeddings should 
    cluster close together even in this 2D projection.
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Upload Image-Text Pairs")
    st.markdown("Upload at least 4 pairs. More pairs give a richer visualization.")

    n_pairs = st.slider("Number of pairs", min_value=4, max_value=8, value=5)
    default_labels = ["dog", "cat", "car", "pizza", "beach", "mountain", "bird", "flower"]

    uploaded_imgs_viz = []
    texts_viz = []
    labels_viz = []
    all_ready_viz = True

    cols = st.columns(n_pairs)
    for i, col in enumerate(cols):
        with col:
            img_f = st.file_uploader(
                f"Image {i+1}", type=["png","jpg","jpeg","webp"], key=f"viz_img_{i}"
            )
            lbl = st.text_input(f"Label {i+1}", value=default_labels[i], key=f"viz_lbl_{i}")
            if img_f:
                img = Image.open(img_f).convert("RGB")
                st.image(img, use_container_width=True)
                uploaded_imgs_viz.append(img)
                texts_viz.append(f"a photo of {lbl}")
                labels_viz.append(lbl)
            else:
                all_ready_viz = False

    if all_ready_viz and len(uploaded_imgs_viz) == n_pairs:
        if st.button("Visualize Embedding Space"):
            with st.spinner("Encoding and projecting to 2D with t-SNE..."):
                img_embs_viz = encode_images(model, processor, uploaded_imgs_viz, device)
                txt_embs_viz = encode_texts(model, processor, texts_viz, device)

                all_embs = np.vstack([img_embs_viz, txt_embs_viz])

                perplexity = min(5, n_pairs - 1)
                tsne = TSNE(
                    n_components=2, perplexity=perplexity,
                    random_state=42, max_iter=1000,
                    learning_rate="auto", init="pca"
                )
                coords = tsne.fit_transform(all_embs)

            palette = [
                "#E53935", "#1565C0", "#2E7D32", "#E65100",
                "#6A1B9A", "#00838F", "#AD1457", "#4E342E"
            ]

            fig, ax = make_fig(9, 7)

            for i in range(n_pairs):
                color = palette[i % len(palette)]
                img_x, img_y = coords[i]
                txt_x, txt_y = coords[n_pairs + i]

                # Dashed line connecting matching pair
                ax.plot([img_x, txt_x], [img_y, txt_y],
                        color=color, linewidth=1.5, alpha=0.5, linestyle="--")

                # Image = square marker
                ax.scatter(img_x, img_y, c=color, s=250, marker="s",
                           zorder=5, edgecolors="white", linewidth=1.5)
                # Text = circle marker
                ax.scatter(txt_x, txt_y, c=color, s=250, marker="o",
                           zorder=5, edgecolors="white", linewidth=1.5)

                ax.annotate(
                    f"IMG: {labels_viz[i]}", (img_x, img_y),
                    textcoords="offset points", xytext=(8, 5),
                    color=color, fontsize=9, fontweight="bold"
                )
                ax.annotate(
                    f"TXT: {labels_viz[i]}", (txt_x, txt_y),
                    textcoords="offset points", xytext=(8, -13),
                    color=color, fontsize=9
                )

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker="s", color="w",
                       markerfacecolor="#607D8B", markersize=10, label="Image embedding"),
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#607D8B", markersize=10, label="Text embedding"),
                Line2D([0], [0], linestyle="--", color="#607D8B", label="Matching pair"),
            ]
            ax.legend(
                handles=legend_elements, loc="upper right",
                facecolor="#FFFFFF", edgecolor="#CBD5E1", labelcolor="#334155"
            )

            ax.set_title(
                "t-SNE projection of CLIP embedding space (512D to 2D)",
                color="#0D47A1", fontsize=13
            )
            ax.set_xlabel("t-SNE dimension 1")
            ax.set_ylabel("t-SNE dimension 2")
            ax.grid(True, color="#E2E8F0", alpha=0.8, linewidth=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("""<div class="concept-box">
            <strong>What to look for:</strong> Matching image-text pairs (connected by dashed lines) 
            should appear close together. Semantically similar concepts like "dog" and "cat" 
            may also cluster nearby. This is CLIP's shared embedding space made visible.
            </div>""", unsafe_allow_html=True)

            st.divider()
            st.subheader("Full Cross-Modal Similarity Heatmap")

            sim_all = cosine_sim_matrix(
                np.vstack([img_embs_viz, txt_embs_viz]),
                np.vstack([img_embs_viz, txt_embs_viz])
            )
            all_labels_heat = (
                [f"IMG:{l}" for l in labels_viz] +
                [f"TXT:{l}" for l in labels_viz]
            )

            fig2, ax2 = plt.subplots(figsize=(10, 8))
            fig2.patch.set_facecolor("#FFFFFF")

            cmap2 = mcolors.LinearSegmentedColormap.from_list(
                "heat", ["#FFFFFF", "#BBDEFB", "#1565C0"], N=256
            )
            sns.heatmap(
                sim_all, annot=True, fmt=".2f", cmap=cmap2,
                xticklabels=all_labels_heat,
                yticklabels=all_labels_heat,
                ax=ax2, linewidths=0.5, linecolor="#E2E8F0",
                annot_kws={"size": 9, "color": "#1A2332"},
                cbar_kws={"shrink": 0.8}
            )
            ax2.set_title(
                "Cross-modal similarity: all images and texts",
                color="#0D47A1", fontsize=12
            )
            ax2.tick_params(colors="#334155", labelsize=9)
            plt.xticks(rotation=30, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            st.markdown("""<div class="highlight-box">
            <strong>Reading this heatmap:</strong> The top-right quadrant (image rows, text columns) 
            is the NxN training matrix from Section 4. The diagonal of that quadrant should be 
            the darkest cells, confirming that CLIP aligns matching pairs more closely than 
            any non-matching pair.
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="warning-box">
        Upload all images above to generate the embedding space visualization.
        </div>""", unsafe_allow_html=True)

"""
CLIP Interactive Walkthrough
SEAS 8525: Computer Vision and Generative AI
Dr. Elbasheer, Week 6

Run locally:
    pip install -r requirements.txt
    streamlit run clip_app.py

Sample images:
    Put your own images in a folder named "samples" next to this file and they
    become the default gallery. The file name is the label, so
    "samples/golden retriever.jpg" shows up as "golden retriever".
    Without that folder the app loads a few COCO validation images.
"""

import base64
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"

st.set_page_config(
    page_title="CLIP Walkthrough · SEAS 8525",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS (house style: Lexend, navy titles, gold tags, light panels)
# ═════════════════════════════════════════════════════════════════════════════
NAVY = "#002147"
GOLD = "#FFC400"
BG = "#F3F6FB"
BORDER = "#D9E1EC"
INK = "#1E293B"
MUTED = "#64748B"

IMG, IMG_BG = "#1D4ED8", "#E3ECFF"        # image side = blue
TXT, TXT_BG = "#0E9F6E", "#DDF5EC"        # text side = green
PROJ, PROJ_BG = "#D97706", "#FFF2DB"      # projection = orange
SPACE, SPACE_BG = "#7C3AED", "#F1EAFE"    # shared space = purple
POS, POS_BG = "#0E9F6E", "#CFEFDF"        # matching pair
NEG, NEG_BG = "#E11D48", "#FDE2E8"        # non-matching pair

PAIR_COLORS = ["#1D4ED8", "#E11D48", "#D97706", "#0E9F6E",
               "#7C3AED", "#0891B2", "#BE185D", "#4B5563"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, p, li, label, input, textarea, button, h1, h2, h3, h4,
.stMarkdown, [data-testid="stWidgetLabel"], [data-testid="stExpander"] summary p {
    font-family: 'Lexend', system-ui, sans-serif !important;
}
code, pre, .stCode code { font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stAppViewContainer"] { background: #F3F6FB; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1320px; }
footer, [data-testid="stDecoration"] { display: none; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #D9E1EC; }
p, li { font-size: 16px; color: #1E293B; line-height: 1.6; }
[data-testid="stExpander"] { background: #FFFFFF; border-radius: 12px; border-color: #D9E1EC; }

/* section banner */
.hero { background: #002147; border-radius: 16px; padding: 18px 28px 16px; margin: 4px 0 16px;
        display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.hero .badge { background: #FFC400; color: #002147; font-weight: 800; font-size: 18px;
        width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
.hero .t { color: #FFFFFF; font-weight: 800; font-size: 32px; letter-spacing: -0.3px; }
.hero .s { color: #C9D6EA; font-size: 17px; flex-basis: 100%; margin-left: 56px; margin-top: -6px; }

/* takeaway box at the end of each section */
.key { background: #FFFFFF; border: 1px solid #D9E1EC; border-top: 6px solid #FFC400;
       border-radius: 16px; padding: 22px 36px; margin: 22px auto 10px; text-align: center;
       box-shadow: 0 4px 14px rgba(0, 33, 71, 0.08); }
.key .x { font-size: 21px; font-weight: 500; color: #002147; line-height: 1.6; max-width: 980px; margin: 0 auto; }
.key .x b { font-weight: 800; }

/* top navigation: big, clear section buttons */
.st-key-topnav { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 10px 12px;
                 box-shadow: 0 2px 10px rgba(0, 33, 71, 0.06); margin-bottom: 4px; }
.st-key-topnav [data-testid="stButtonGroup"], .st-key-topnav [data-testid="stButtonGroup"] > div { width: 100%; }
.st-key-topnav [data-testid="stButtonGroup"] button { flex: 1 1 0; min-height: 54px; padding: 8px 14px;
                 border-radius: 12px !important; border: 1.5px solid #D9E1EC; background: #F3F6FB; }
.st-key-topnav [data-testid="stButtonGroup"] button p { font-size: 18px !important; font-weight: 700 !important; color: #002147; }
.st-key-topnav [data-testid="stButtonGroup"] button:hover { border-color: #002147; }
.st-key-topnav [data-testid="stBaseButton-segmented_controlActive"] { background: #002147 !important; border-color: #002147 !important; }
.st-key-topnav [data-testid="stBaseButton-segmented_controlActive"] p { color: #FFC400 !important; }
.st-key-topnav [role="radiogroup"] label p { font-size: 18px !important; font-weight: 700; color: #002147; }

/* "by the numbers" fact cards */
.facts { display: grid; grid-template-columns: 1.35fr 1fr 1fr; gap: 14px; margin: 6px 0 4px; }
@media (max-width: 1000px) { .facts { grid-template-columns: 1fr; } }
.fgroup { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 14px 16px; }
.fgroup .gh { font-weight: 800; font-size: 16px; margin-bottom: 2px; }
.fgroup .gs { font-size: 13px; color: #64748B; margin-bottom: 10px; }
.fact { padding: 10px 0; border-top: 1px solid #EEF2F7; }
.fact .n { font-size: 32px; font-weight: 800; line-height: 1.1; white-space: nowrap; }
.fact .k { font-weight: 700; font-size: 15.5px; margin-top: 2px; color: #002147; }
.fact .d { font-size: 13.5px; color: #475569; line-height: 1.45; margin-top: 2px; }

/* small stat chips */
.chips { display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 12px; }
.chip { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 14px; padding: 10px 18px; min-width: 150px; }
.chip b { display: block; font-size: 24px; font-weight: 800; color: #002147; }
.chip span { font-size: 13px; color: #64748B; }

/* panels and pipelines */
.panel { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 18px 20px; margin: 6px 0 10px; }
.ptitle { font-weight: 700; color: #002147; font-size: 17px; margin-bottom: 10px; display: flex; gap: 10px; align-items: center; }
.tag { display: inline-block; border-radius: 999px; padding: 2px 12px; font-size: 13px; font-weight: 700; }
.pipe { display: flex; align-items: center; gap: 6px; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 2px; }
.pipe > * { flex-shrink: 0; }
.arrow { font-size: 22px; color: #94A3B8; font-weight: 700; }
.node { border-radius: 12px; padding: 8px 11px; text-align: center; font-weight: 700; font-size: 14px; line-height: 1.3; }
.node small { display: block; font-weight: 500; color: #64748B; font-size: 12px; margin-top: 2px; }
.n-img  { background: #E3ECFF; border: 2px solid #1D4ED8; color: #1D4ED8; }
.n-txt  { background: #DDF5EC; border: 2px solid #0E9F6E; color: #0B7A55; }
.n-proj { background: #FFF2DB; border: 2px solid #D97706; color: #B45309; }
.n-space{ background: #F1EAFE; border: 2px solid #7C3AED; color: #6D28D9; }
.n-plain{ background: #FFFFFF; border: 1px dashed #94A3B8; color: #1E293B; }
.thumb { border-radius: 10px; display: block; }

/* token sequences */
.seq { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; justify-content: center; }
.seq img { width: 24px; height: 24px; border-radius: 4px; border: 1px solid #93B4F5; }
.tk { display: inline-block; border-radius: 8px; padding: 3px 7px; font-size: 13px; font-weight: 600;
      background: #FFFFFF; border: 1.5px solid #0E9F6E; color: #0B7A55; }
.tk.sp { background: #F1F5F9; border-color: #94A3B8; color: #475569; }
.tk.eot { background: #0E9F6E; border-color: #0E9F6E; color: #FFFFFF; }
.tk.cls { background: #1D4ED8; border-color: #1D4ED8; color: #FFFFFF; }

/* embedding strips */
.strip { display: flex; gap: 1px; }
.strip i { display: block; width: 8px; height: 30px; border-radius: 2px; }
.grid5 { display: grid; grid-template-columns: auto 34px auto 34px 1fr; gap: 12px 6px; align-items: center; }
.slabel { font-size: 12.5px; color: #64748B; margin-top: 4px; font-weight: 500; }

/* similarity matrix */
table.mx { border-collapse: separate; border-spacing: 6px; margin: 0 auto; }
table.mx td { width: 104px; height: 74px; text-align: center; border-radius: 10px; font-size: 19px;
              color: #1E293B; border: 1px solid rgba(0,0,0,0.06); }
table.mx td.d { font-weight: 800; }
table.mx td.win { box-shadow: inset 0 0 0 3px #002147; }
table.mx th { font-weight: 600; font-size: 14px; color: #1E293B; }
table.mx th.col span { display: block; border-radius: 10px; padding: 6px 8px; border: 2px solid; line-height: 1.25; }
table.mx th.row { text-align: left; }
table.mx th.row div { display: flex; align-items: center; gap: 8px; }
table.mx th.row img { width: 66px; height: 66px; object-fit: cover; border-radius: 10px; border: 3px solid; }

/* zero-shot step cards */
.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
@media (max-width: 1000px) { .steps { grid-template-columns: 1fr 1fr; } }
.step { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 14px; }
.step h4 { margin: 0 0 12px 0; padding: 8px 12px; border-radius: 10px; font-size: 17px; font-weight: 700; }
.bar { display: grid; grid-template-columns: 1fr 62px; gap: 6px; align-items: center; margin: 5px 0; font-size: 14px; }
.bar .track { background: #F1F5F9; border-radius: 6px; height: 26px; position: relative; overflow: hidden; }
.bar .fill { height: 100%; border-radius: 6px; }
.bar .lab { position: absolute; left: 8px; top: 3px; font-weight: 600; color: #1E293B; white-space: nowrap; }
.bar .v { text-align: right; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 13.5px; }

/* sidebar gallery */
.gal { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.gal div { font-size: 11px; color: #475569; text-align: center; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.gal img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 8px; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SMALL HTML HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def html(s: str):
    """Render an HTML snippet. Lines are joined so Markdown never sees
    indentation or blank lines (which would break the HTML)."""
    one_line = " ".join(line.strip() for line in s.splitlines() if line.strip())
    st.markdown(f"<div>{one_line}</div>", unsafe_allow_html=True)


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("$", "&#36;"))


def hero(num, title, subtitle):
    html(f'<div class="hero"><div class="badge">{num}</div><div class="t">{title}</div>'
         f'<div class="s">{subtitle}</div></div>')


def key_idea(text):
    html(f'<div class="key"><div class="x">{text}</div></div>')


def chips(items):
    inner = "".join(f'<div class="chip"><b>{v}</b><span>{k}</span></div>' for v, k in items)
    html(f'<div class="chips">{inner}</div>')


def mix(c1, c2, t):
    t = float(np.clip(t, 0, 1))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def val_color(t, pos_color):
    """t in [-1, 1]: positive values tint toward pos_color, negative toward red."""
    return mix("#FFFFFF", pos_color, t) if t >= 0 else mix("#FFFFFF", NEG, -t)


def strip_html(vec, color, n=64):
    v = np.asarray(vec[:n], dtype=float)
    m = max(float(np.abs(v).max()), 1e-6)
    cells = "".join(f'<i style="background:{val_color(x / m, color)}"></i>' for x in v)
    return f'<div class="strip">{cells}</div>'


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def seg(label, options, key, default=None):
    """Segmented control with a radio fallback for older Streamlit versions."""
    if key not in st.session_state:
        st.session_state[key] = default or options[0]
    if hasattr(st, "segmented_control"):
        val = st.segmented_control(label, options, key=key)
    else:
        val = st.radio(label, options, key=key, horizontal=True)
    return val or options[0]


def code_panel(title, code):
    with st.expander(f"💻  Code · {title}", expanded=False):
        st.code(code, language="python")


# ═════════════════════════════════════════════════════════════════════════════
# IMAGES
# ═════════════════════════════════════════════════════════════════════════════
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}

COCO = "http://images.cocodataset.org/val2017/"
SAMPLE_URLS = [
    ("cat", "two cats sleeping on a couch", COCO + "000000039769.jpg"),
    ("bear", "a brown bear sitting in the grass", COCO + "000000000285.jpg"),
    ("stop sign", "a red stop sign by the road", COCO + "000000000724.jpg"),
    ("skier", "a person skiing down a snowy slope", COCO + "000000000785.jpg"),
    ("teddy bear", "a group of teddy bears", COCO + "000000000776.jpg"),
    ("kitchen", "a kitchen with a refrigerator", COCO + "000000000802.jpg"),
    ("bedroom", "a bedroom with a bed", COCO + "000000000632.jpg"),
    ("living room", "a living room with a couch and a tv", COCO + "000000000139.jpg"),
]


@st.cache_data(show_spinner=False)
def fetch_bytes(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read()
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=128)
def to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def clip_crop(img, size=224):
    """Resize the short side to `size`, then center crop (what CLIPProcessor does)."""
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))


def pil_uri(img, fmt="JPEG", quality=85):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False, max_entries=256)
def thumb_uri(b: bytes, size=160):
    return pil_uri(clip_crop(to_pil(b), size))


def build_pool(uploads):
    """Ordered dict: label -> {"bytes", "caption"}."""
    pool = {}

    def add(label, caption, data):
        name, k = label, 2
        while name in pool:
            name = f"{label} ({k})"
            k += 1
        pool[name] = {"bytes": data, "caption": caption}

    local = Path(__file__).parent / "samples"
    files = sorted(p for p in local.glob("*") if p.suffix.lower() in IMG_EXT) if local.exists() else []
    if files:
        for p in files:
            lbl = p.stem.replace("_", " ")
            add(lbl, f"a photo of {article(lbl)} {lbl}", p.read_bytes())
    else:
        for lbl, cap, url in SAMPLE_URLS:
            data = fetch_bytes(url)
            if data:
                add(lbl, cap, data)

    for f in uploads or []:
        lbl = Path(f.name).stem.replace("_", " ")
        add(lbl, f"a photo of {article(lbl)} {lbl}", f.getvalue())
    return pool


# ═════════════════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading CLIP ViT-B/32 (first run downloads about 600 MB)...")
def load_clip():
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, processor, device


def _unit(out, projection, dim):
    """Works across transformers versions: older ones return a Tensor, newer
    ones may return a model-output object. Always ends as unit-length [B, 512]."""
    if not isinstance(out, torch.Tensor):
        for name in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(out, name, None)
            if v is not None:
                out = v
                break
    if out.shape[-1] != dim:
        out = projection(out)
    return out / out.norm(dim=-1, keepdim=True)


@st.cache_data(show_spinner=False, max_entries=512)
def embed_image(b: bytes) -> np.ndarray:
    model, processor, device = load_clip()
    inputs = processor(images=[to_pil(b)], return_tensors="pt").to(device)
    with torch.no_grad():
        f = _unit(model.get_image_features(**inputs), model.visual_projection,
                  model.config.projection_dim)
    return f.cpu().numpy()[0]


@st.cache_data(show_spinner=False, max_entries=2048)
def embed_text(t: str) -> np.ndarray:
    model, processor, device = load_clip()
    inputs = processor(text=[t], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        f = _unit(model.get_text_features(**inputs), model.text_projection,
                  model.config.projection_dim)
    return f.cpu().numpy()[0]


def embed_images(list_of_bytes):
    return np.stack([embed_image(b) for b in list_of_bytes])


def embed_texts(texts):
    return np.stack([embed_text(t) for t in texts])


@st.cache_resource(show_spinner=False)
def logit_scale():
    model, _, _ = load_clip()
    return float(model.logit_scale.exp().item())


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def xent_diag(logits):
    """Mean cross-entropy where row i's correct class is column i."""
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    return float(np.mean(lse - np.diag(logits)))


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR: course info + shared image gallery
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    html(f'<div style="font-weight:800;font-size:22px;color:{NAVY}">🔗 CLIP Walkthrough</div>'
         f'<div style="margin-top:4px"><span class="tag" style="background:{GOLD};color:{NAVY}">SEAS 8525</span>'
         f'<span style="color:{MUTED};font-size:14px;margin-left:8px">Week 6 · Dr. Elbasheer</span></div>')
    st.divider()
    st.markdown("**Image gallery**")
    st.caption("Every section picks images from here. Add your own below.")
    uploads = st.file_uploader("Add images", type=["png", "jpg", "jpeg", "webp"],
                               accept_multiple_files=True, label_visibility="collapsed")
    POOL = build_pool(uploads)
    if POOL:
        tiles = "".join(f'<div><img src="{thumb_uri(v["bytes"], 120)}"/>{esc(k)}</div>'
                        for k, v in POOL.items())
        html(f'<div class="gal">{tiles}</div>')
    else:
        st.warning("No sample images could be loaded. Upload a few images above.")
    st.divider()
    if torch.cuda.is_available():
        st.caption(f"⚡ GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.caption("Running on CPU. The first encoding of each image takes a second.")
    with st.expander("Run it locally"):
        st.code("pip install -r requirements.txt\nstreamlit run clip_app.py", language="bash")

LABELS = list(POOL)


def need_images(k=1):
    if len(LABELS) < k:
        st.info(f"This section needs at least {k} image(s). Add some in the sidebar gallery.")
        return True
    return False


def pick_image(key, label="Image", index=0):
    return st.selectbox(label, LABELS, index=min(index, len(LABELS) - 1), key=key)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 · BIG PICTURE
# ═════════════════════════════════════════════════════════════════════════════
def svg_strip(x, y, fill, stroke, w=150, h=38, n=5):
    cw = w / n
    out = []
    for i in range(n):
        out.append(f'<rect x="{x + i * cw:.1f}" y="{y}" width="{cw:.1f}" height="{h}" '
                   f'fill="{"#FFFFFF" if i == n - 2 else fill}" stroke="{stroke}" stroke-width="2"/>')
    out.append(f'<text x="{x + (n - 1.5) * cw:.1f}" y="{y + h / 2 + 5}" text-anchor="middle" '
               f'font-size="18" fill="{stroke}">···</text>')
    return "".join(out)


def arch_svg(img_uri, caption, vis_dim, txt_dim, proj_dim):
    # wrap the caption onto at most two short lines so it fits the dashed box
    lines, cur = [], ""
    for w in caption.split():
        if len(cur) + len(w) + 1 > 16 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    if len(lines) > 2:
        lines = [lines[0], lines[1][:14] + "…"]
    lines = [l if len(l) <= 17 else l[:16] + "…" for l in lines]
    lines[0] = "“" + lines[0]
    lines[-1] = lines[-1] + "”"
    A ='stroke="#334155" stroke-width="2.5" marker-end="url(#ah)"'

    def row(y, color, bg, kind, left_svg, enc_title, enc_sub, d):
        cy = y + 95
        return f"""
        {left_svg}
        <line x1="150" y1="{cy}" x2="172" y2="{cy}" {A}/>
        <polygon points="180,{y} 330,{y + 40} 330,{y + 150} 180,{y + 190}" fill="{bg}" stroke="{color}" stroke-width="2.5"/>
        <text x="255" y="{cy - 4}" text-anchor="middle" font-size="18" font-weight="700" fill="{color}">{enc_title}</text>
        <text x="255" y="{cy + 20}" text-anchor="middle" font-size="14" fill="#475569">{enc_sub}</text>
        <line x1="334" y1="{cy}" x2="356" y2="{cy}" {A}/>
        {svg_strip(364, cy - 19, bg, color)}
        <text x="439" y="{cy + 44}" text-anchor="middle" font-size="14" fill="#475569">{kind} features · d = {d}</text>
        <line x1="518" y1="{cy}" x2="540" y2="{cy}" {A}/>
        <rect x="548" y="{cy - 38}" width="122" height="76" rx="12" fill="{PROJ_BG}" stroke="{PROJ}" stroke-width="2.5"/>
        <text x="609" y="{cy - 6}" text-anchor="middle" font-size="16" font-weight="700" fill="#B45309">Projection</text>
        <text x="609" y="{cy + 18}" text-anchor="middle" font-size="14" fill="#475569">{d} → {proj_dim}</text>
        <line x1="674" y1="{cy}" x2="696" y2="{cy}" {A}/>
        {svg_strip(704, cy - 19, SPACE_BG, SPACE)}
        <text x="779" y="{cy + 44}" text-anchor="middle" font-size="14" fill="#475569">{kind} vector · {proj_dim}</text>
        """

    img_left = (f'<text x="80" y="52" text-anchor="middle" font-size="17" font-weight="700" fill="{IMG}">Image</text>'
                f'<image href="{img_uri}" x="20" y="65" width="120" height="120" preserveAspectRatio="xMidYMid slice"/>'
                f'<rect x="20" y="65" width="120" height="120" fill="none" stroke="{IMG}" stroke-width="3" rx="6"/>')
    ty = 250
    txt_left = (f'<text x="80" y="{ty + 22}" text-anchor="middle" font-size="17" font-weight="700" fill="{TXT}">Text</text>'
                f'<rect x="12" y="{ty + 60}" width="138" height="72" rx="10" fill="#FFFFFF" stroke="{TXT}" stroke-width="2" stroke-dasharray="6 4"/>'
                + "".join(f'<text x="81" y="{ty + (101 if len(lines) == 1 else 92 + k * 19)}" text-anchor="middle" '
                          f'font-size="13.5" fill="#1E293B">{esc(l)}</text>' for k, l in enumerate(lines)))

    dots = ""
    pts_img = [(985, 185), (1145, 315), (1060, 235)]
    pts_txt = [(1000, 335), (1150, 200), (1095, 255)]
    for x, y in pts_img:
        dots += f'<rect x="{x - 9}" y="{y - 9}" width="18" height="18" rx="3" fill="{IMG}"/>'
    for x, y in pts_txt:
        dots += f'<circle cx="{x}" cy="{y}" r="10" fill="{TXT}"/>'

    return f"""
    <div class="panel">
    <svg viewBox="0 0 1220 480" width="100%" style="font-family:Lexend, sans-serif">
      <defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#334155"/></marker></defs>
      {row(30, IMG, IMG_BG, "image", img_left, "Image Encoder", "ViT-B/32 · 12 layers", vis_dim)}
      {row(ty, TXT, TXT_BG, "text", txt_left, "Text Encoder", "Transformer · 12 layers", txt_dim)}
      <path d="M858,125 C900,125 905,170 930,190" fill="none" stroke="{SPACE}" stroke-width="2.5" marker-end="url(#ah)"/>
      <path d="M858,345 C900,345 905,300 930,282" fill="none" stroke="{SPACE}" stroke-width="2.5" marker-end="url(#ah)"/>
      <circle cx="1070" cy="240" r="150" fill="{SPACE_BG}" stroke="{SPACE}" stroke-width="3"/>
      <text x="1070" y="128" text-anchor="middle" font-size="18" font-weight="800" fill="{SPACE}">Shared space</text>
      <text x="1070" y="150" text-anchor="middle" font-size="14" fill="#6D28D9">{proj_dim}-d unit sphere</text>
      <line x1="1060" y1="235" x2="1095" y2="255" stroke="#334155" stroke-width="2" stroke-dasharray="5 4"/>
      {dots}
      <text x="1078" y="292" text-anchor="middle" font-size="13" fill="#334155">matching pair</text>
      <rect x="975" y="402" width="14" height="14" rx="3" fill="{IMG}"/>
      <text x="996" y="414" font-size="13" fill="#334155">image</text>
      <circle cx="1068" cy="409" r="7" fill="{TXT}"/>
      <text x="1081" y="414" font-size="13" fill="#334155">text</text>
    </svg></div>"""


FACTS = [
    ("Training setup", "What the original authors used to train CLIP", PROJ, [
        ("400M", "image-text pairs",
         "Photos and their captions collected from the internet (the WIT dataset). No human labeling."),
        ("32,768", "pairs per batch",
         "Every training step compares each image with all 32,768 captions: the N×N matrix of Section 4."),
        ("32", "epochs", "Each model passed over the full dataset 32 times."),
    ]),
    ("This model", "The checkpoint running in this app", IMG, [
        ("ViT-B/32", "image encoder", "Vision Transformer, Base size, cutting images into 32×32 pixel patches."),
        ("512", "shared space size", "Every image and every caption becomes a list of 512 numbers."),
    ]),
    ("Results", "ImageNet accuracy with zero ImageNet training", POS, [
        ("63.2%", "this model (ViT-B/32)", "Top-1 on 1,000 classes using only prompts like \"a photo of a dog.\""),
        ("76.2%", "largest CLIP (ViT-L/14@336px)",
         "Matches a ResNet-50 trained on all 1.28M labeled ImageNet images."),
    ]),
]


def facts_panel():
    html(f'<div style="font-weight:800;font-size:20px;color:{NAVY};margin:14px 0 6px">CLIP by the numbers '
         f'<span style="font-weight:500;font-size:15px;color:{MUTED}">· from the original paper, Radford et al. 2021</span></div>')
    cols = ""
    for title, sub, color, items in FACTS:
        rows = "".join(f'<div class="fact"><div class="n" style="color:{color}">{n}</div>'
                       f'<div><div class="k">{k}</div><div class="d">{esc(d)}</div></div></div>'
                       for n, k, d in items)
        cols += (f'<div class="fgroup" style="border-top:5px solid {color}"><div class="gh" style="color:{color}">{title}</div>'
                 f'<div class="gs">{sub}</div>{rows}</div>')
    html(f'<div class="facts">{cols}</div>')


def sec_big_picture():
    hero(1, "CLIP: two encoders, one shared space",
         "Contrastive Language-Image Pretraining · Radford et al., OpenAI 2021")
    if need_images():
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        lbl = pick_image("s1_img", "Example image")
    with c2:
        caption = st.text_input("Example caption", value=POOL[lbl]["caption"], key=f"s1_cap_{lbl}")

    model, _, _ = load_clip()
    html(arch_svg(thumb_uri(POOL[lbl]["bytes"], 240), caption,
                  model.config.vision_config.hidden_size,
                  model.config.text_config.hidden_size,
                  model.config.projection_dim))

    facts_panel()

    key_idea("Both encoders run <b>independently</b>. Projection heads map their outputs into the "
             f"<b style='color:{SPACE}'>same shared space</b>, where a matching image and caption "
             "point in nearly the same direction.")

    code_panel("load CLIP and embed one image and one caption", '''\
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("dog.jpg").convert("RGB")
img_in = processor(images=image, return_tensors="pt")
txt_in = processor(text=["a photo of a dog"], return_tensors="pt", padding=True)

with torch.no_grad():
    img = model.get_image_features(**img_in)   # [1, 512]
    txt = model.get_text_features(**txt_in)    # [1, 512]
# Note: some newer transformers versions return an output object here.
# If so, take its .pooler_output (see the helper in this app).

img = img / img.norm(dim=-1, keepdim=True)     # unit length
txt = txt / txt.norm(dim=-1, keepdim=True)
print((img @ txt.T).item())                     # cosine similarity''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 · TWO ENCODERS
# ═════════════════════════════════════════════════════════════════════════════
def grid_overlay(img224, p=32):
    g = img224.copy()
    d = ImageDraw.Draw(g)
    for k in range(0, 225, p):
        d.line([(k, 0), (k, 223)], fill="white", width=2)
        d.line([(0, k), (223, k)], fill="white", width=2)
    return g


def patches(img224, p=32):
    n = 224 // p
    return [img224.crop((c * p, r * p, c * p + p, r * p + p)) for r in range(n) for c in range(n)]


def causal_mask_svg(tokens):
    n = len(tokens)
    cell = 14
    W = 70 + n * cell
    H = 20 + n * cell
    out = [f'<svg viewBox="0 0 {W} {H}" width="{min(W, 210)}" style="font-family:Lexend">']
    for i, t in enumerate(tokens):
        lbl = esc(t if len(t) <= 7 else t[:6] + "…")
        is_last = i == n - 1
        out.append(f'<text x="64" y="{20 + i * cell + 11}" text-anchor="end" font-size="10" '
                   f'fill="{"#0B7A55" if is_last else "#475569"}" font-weight="{700 if is_last else 400}">{lbl}</text>')
        for j in range(n):
            on = j <= i
            fill = (TXT if is_last else "#9ADBC4") if on else "#F1F5F9"
            out.append(f'<rect x="{70 + j * cell}" y="{20 + i * cell}" width="{cell - 2}" height="{cell - 2}" rx="3" fill="{fill}"/>')
    out.append(f'<text x="{70 + n * cell / 2}" y="12" text-anchor="middle" font-size="11" fill="#475569">attends to →</text>')
    out.append("</svg>")
    return "".join(out)


def sec_two_encoders():
    hero(2, "The two encoders", "How an image and a sentence each become one 512-d vector")
    if need_images():
        return
    model, processor, _ = load_clip()
    vc, tc, pdim = model.config.vision_config, model.config.text_config, model.config.projection_dim
    p = vc.patch_size
    n_side = vc.image_size // p
    n_patch = n_side * n_side

    # ── image tower ─────────────────────────────────────────────────────────
    lbl = pick_image("s2_img", "Image")
    img224 = clip_crop(to_pil(POOL[lbl]["bytes"]), vc.image_size)
    ps = patches(img224, p)
    seq = ('<span class="tk cls">CLS</span>' +
           "".join(f'<img src="{pil_uri(q)}"/>' for q in ps[:4]) +
           '<span style="color:#64748B;font-weight:700">···</span>' +
           f'<img src="{pil_uri(ps[-1])}"/>')
    html(f"""
    <div class="panel">
      <div class="ptitle"><span class="tag" style="background:{IMG_BG};color:{IMG}">IMAGE</span>Vision Transformer, ViT-B/{p}</div>
      <div class="pipe">
        <div class="node n-plain"><img class="thumb" src="{pil_uri(grid_overlay(img224, p))}" width="112"/>
          <small>{vc.image_size}×{vc.image_size} → {n_side}×{n_side} = {n_patch} patches</small></div>
        <div class="arrow">→</div>
        <div class="node n-img" style="width:180px"><div class="seq">{seq}</div>
          <small>{n_patch + 1} tokens = [CLS] + {n_patch} patches<br>{p}×{p}×3 = {p * p * 3} values → {vc.hidden_size}</small></div>
        <div class="arrow">→</div>
        <div class="node n-img">Transformer × {vc.num_hidden_layers}<small>{vc.num_attention_heads} heads · width {vc.hidden_size}</small>
          <div style="margin-top:6px"><small>keep only</small> <span class="tk cls">CLS</span><small>{vc.hidden_size}-d</small></div></div>
        <div class="arrow">→</div>
        <div class="node n-proj">Projection<small>{vc.hidden_size} → {pdim}</small></div>
        <div class="arrow">→</div>
        <div class="node n-space">Image<br>vector<small>{pdim}-d</small></div>
      </div>
    </div>""")

    # ── text tower ──────────────────────────────────────────────────────────
    text = st.text_input("Sentence", value=POOL[lbl]["caption"], key="s2_text")
    ids = processor.tokenizer(text, truncation=True)["input_ids"]
    toks = []
    for i, t in enumerate(ids):
        if i == 0:
            toks.append("[SOT]")
        elif i == len(ids) - 1:
            toks.append("[EOT]")
        else:
            toks.append(processor.tokenizer.decode([t]).strip() or "·")
    chips_html = "".join(
        f'<span class="tk {"sp" if i == 0 else "eot" if i == len(toks) - 1 else ""}">{esc(t)}</span>'
        for i, t in enumerate(toks))
    mask_toks = toks if len(toks) <= 9 else toks[:3] + ["…"] + toks[-5:]
    html(f"""
    <div class="panel">
      <div class="ptitle"><span class="tag" style="background:{TXT_BG};color:#0B7A55">TEXT</span>Causal Transformer (GPT-style)</div>
      <div class="pipe">
        <div class="node n-plain" style="width:190px"><div class="seq">{chips_html}</div>
          <small>{len(ids)} of {tc.max_position_embeddings} tokens<br>BPE vocab {tc.vocab_size:,}</small></div>
        <div class="arrow">→</div>
        <div class="node n-plain">{causal_mask_svg(mask_toks)}<small>causal mask</small></div>
        <div class="arrow">→</div>
        <div class="node n-txt">Transformer × {tc.num_hidden_layers}<small>{tc.num_attention_heads} heads · width {tc.hidden_size}</small>
          <div style="margin-top:6px"><small>keep only</small> <span class="tk eot">[EOT]</span><small>{tc.hidden_size}-d</small></div></div>
        <div class="arrow">→</div>
        <div class="node n-proj">Projection<small>{tc.hidden_size} → {pdim}</small></div>
        <div class="arrow">→</div>
        <div class="node n-space">Text<br>vector<small>{pdim}-d</small></div>
      </div>
    </div>""")

    key_idea(f"The image is summarized by its <b style='color:{IMG}'>[CLS]</b> token. The sentence is summarized "
             f"by its <b style='color:{TXT}'>[EOT]</b> token, because with a causal mask it is the only token "
             "that has seen the whole sentence.")

    with st.expander("🔍  Live model config and parameter counts"):
        a, b = st.columns(2)
        with a:
            n = sum(q.numel() for q in model.vision_model.parameters())
            st.markdown(f"**Image encoder** · {n / 1e6:.1f}M parameters")
            st.json({"patch_size": vc.patch_size, "image_size": vc.image_size,
                     "hidden_size": vc.hidden_size, "layers": vc.num_hidden_layers,
                     "heads": vc.num_attention_heads, "mlp_size": vc.intermediate_size,
                     "projection_dim": pdim})
        with b:
            n = sum(q.numel() for q in model.text_model.parameters())
            st.markdown(f"**Text encoder** · {n / 1e6:.1f}M parameters")
            st.json({"vocab_size": tc.vocab_size, "max_tokens": tc.max_position_embeddings,
                     "hidden_size": tc.hidden_size, "layers": tc.num_hidden_layers,
                     "heads": tc.num_attention_heads, "mlp_size": tc.intermediate_size,
                     "projection_dim": pdim})

    code_panel("inspect the encoders and tokenize a sentence", '''\
vc, tc = model.config.vision_config, model.config.text_config
print(vc.patch_size, vc.image_size, vc.hidden_size, vc.num_hidden_layers)  # 32 224 768 12
print(tc.vocab_size, tc.max_position_embeddings, tc.hidden_size)          # 49408 77 512
print(model.config.projection_dim)                                        # 512

ids = processor.tokenizer("a photo of a dog")["input_ids"]
print(ids)   # [49406, ..., 49407]   49406 = [SOT], 49407 = [EOT]
print([processor.tokenizer.decode([t]) for t in ids])''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 · ONE PAIR
# ═════════════════════════════════════════════════════════════════════════════
def sec_one_pair():
    hero(3, "One image, one caption", "Two vectors, one dot product")
    if need_images():
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        lbl = pick_image("s3_img")
    with c2:
        caption = st.text_input("Caption", value=POOL[lbl]["caption"], key=f"s3_cap_{lbl}")

    with st.spinner("Encoding..."):
        a = embed_image(POOL[lbl]["bytes"])
        b = embed_text(caption)
    sim = float(a @ b)
    prod = a * b

    left, right = st.columns([3, 1.05])
    with left:
        html(f"""
        <div class="panel">
          <div class="grid5">
            <img class="thumb" src="{thumb_uri(POOL[lbl]["bytes"], 160)}" width="92"/>
            <div class="arrow">→</div>
            <div class="node n-img">Image encoder<small>+ projection</small></div>
            <div class="arrow">→</div>
            <div>{strip_html(a, IMG)}<div class="slabel">image vector <b>a</b> · first 64 of 512 dims</div></div>

            <div class="node n-plain" style="max-width:150px;font-weight:500;font-size:14px">"{esc(caption)}"</div>
            <div class="arrow">→</div>
            <div class="node n-txt">Text encoder<small>+ projection</small></div>
            <div class="arrow">→</div>
            <div>{strip_html(b, TXT)}<div class="slabel">text vector <b>b</b> · first 64 of 512 dims</div></div>

            <div></div><div></div>
            <div class="node n-space">multiply<br>element-wise</div>
            <div class="arrow">→</div>
            <div>{strip_html(prod, SPACE)}<div class="slabel"><b>a ⊙ b</b> · add up all 512 of these to get the cosine</div></div>
          </div>
        </div>""")
    with right:
        color = POS if sim > 0.25 else NEG if sim < 0.15 else PROJ
        verdict = "strong match" if sim > 0.25 else "weak match" if sim < 0.15 else "partial match"
        pos = float(np.clip(sim / 0.4, 0, 1)) * 100
        html(f"""
        <div class="panel" style="text-align:center">
          <div style="color:{MUTED};font-size:14px">cosine similarity</div>
          <div style="font-size:52px;font-weight:800;color:{color};line-height:1.1">{sim:.3f}</div>
          <div style="font-weight:700;color:{color};margin-bottom:14px">{verdict}</div>
          <div style="position:relative;height:14px;border-radius:7px;
               background:linear-gradient(90deg,{NEG_BG} 0%,{NEG_BG} 37%,{PROJ_BG} 37%,{PROJ_BG} 62%,{POS_BG} 62%)">
            <div style="position:absolute;left:calc({pos:.1f}% - 7px);top:-4px;width:14px;height:22px;
                 border-radius:4px;background:{NAVY}"></div></div>
          <div style="display:flex;justify-content:space-between;font-size:12px;color:{MUTED};margin-top:4px">
            <span>0</span><span>0.15</span><span>0.25</span><span>0.4</span></div>
        </div>""")

    key_idea("Both vectors have length 1, so the cosine is just <b>Σ a<sub>i</sub> b<sub>i</sub></b>. "
             "Raw CLIP cosines are small: a good match usually lands around <b>0.25 to 0.35</b>, not near 1. "
             "Try a wrong caption and watch the score drop.")

    code_panel("what happened under the hood", f'''\
a = model.get_image_features(**image_inputs)    # [1, 512]
b = model.get_text_features(**text_inputs)      # [1, 512]
a = a / a.norm(dim=-1, keepdim=True)
b = b / b.norm(dim=-1, keepdim=True)

similarity = (a * b).sum()     # same as a @ b.T
# here: {sim:.4f}''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 · N×N MATRIX
# ═════════════════════════════════════════════════════════════════════════════
def matrix_html(chosen, captions, M, mode):
    n = len(chosen)
    head = "<tr><th></th>" + "".join(
        f'<th class="col"><span style="border-color:{PAIR_COLORS[j % 8]};'
        f'background:{mix("#FFFFFF", PAIR_COLORS[j % 8], 0.12)}">{esc(captions[j])}</span></th>'
        for j in range(n)) + "</tr>"

    if mode == "cos":
        lo, hi = float(M.min()), float(M.max())
        norm = (M - lo) / max(hi - lo, 1e-6)
        fmt = lambda v: f"{v:.3f}"
        win = M.argmax(axis=1)
        is_win = lambda i, j: j == win[i]
    elif mode == "i2t":
        norm = M
        fmt = lambda v: f"{v * 100:.0f}%"
        win = M.argmax(axis=1)
        is_win = lambda i, j: j == win[i]
    else:
        norm = M
        fmt = lambda v: f"{v * 100:.0f}%"
        win = M.argmax(axis=0)
        is_win = lambda i, j: i == win[j]

    rows = ""
    for i, lbl in enumerate(chosen):
        c = PAIR_COLORS[i % 8]
        rows += (f'<tr><th class="row"><div><img src="{thumb_uri(POOL[lbl]["bytes"], 132)}" '
                 f'style="border-color:{c}"/>{esc(lbl)}</div></th>')
        for j in range(n):
            t = 0.25 + 0.75 * float(norm[i, j])
            bg = mix("#FFFFFF", "#7FD3B0", t) if i == j else mix("#FFFFFF", "#F6A3B5", t * 0.8)
            cls = ("d " if i == j else "") + ("win" if is_win(i, j) else "")
            rows += f'<td class="{cls}" style="background:{bg}">{fmt(M[i, j])}</td>'
        rows += "</tr>"
    return f'<table class="mx">{head}{rows}</table>'


def sec_matrix():
    hero(4, "The N×N contrastive matrix", "Every image in the batch is compared with every caption")
    if need_images(2):
        return

    c1, c2 = st.columns([3, 2])
    with c1:
        chosen = st.multiselect("Images in the batch (2 to 6)", LABELS, default=LABELS[:4],
                                max_selections=6, key="s4_imgs")
    with c2:
        view = seg("Show", ["Cosine", "Softmax: image → text", "Softmax: text → image"], key="s4_view")
    if len(chosen) < 2:
        st.info("Pick at least two images.")
        return

    with st.expander("✏️  Edit captions"):
        df = pd.DataFrame({"image": chosen, "caption": [POOL[l]["caption"] for l in chosen]})
        edited = st.data_editor(df, disabled=["image"], hide_index=True,
                                key="s4_editor_" + "|".join(chosen))
    captions = [str(c) for c in edited["caption"].tolist()]

    with st.spinner("Encoding the batch..."):
        I = embed_images([POOL[l]["bytes"] for l in chosen])
        T = embed_texts(captions)
    S = I @ T.T

    learned_tau = 1.0 / logit_scale()
    taus = [1.0, 0.5, 0.2, 0.1, 0.07, 0.05, 0.02, round(learned_tau, 4)]
    tau = st.select_slider(
        "Temperature τ", options=taus, value=taus[-1], key="s4_tau",
        format_func=lambda t: f"{t:g} (learned)" if t == taus[-1] else
        (f"{t:g} (initial)" if t == 0.07 else f"{t:g}"))

    logits = S / tau
    if view.startswith("Cosine"):
        M, mode = S, "cos"
    elif "image →" in view:
        M, mode = softmax(logits, axis=1), "i2t"
    else:
        M, mode = softmax(logits, axis=0), "t2i"

    left, right = st.columns([3, 1.3])
    with left:
        html(f'<div class="panel" style="overflow-x:auto">'
             f'<div style="text-align:center;color:{MUTED};font-size:14px;margin-bottom:4px">'
             f'Text (columns) · Images (rows) · dark outline = the model\'s pick</div>'
             f'{matrix_html(chosen, captions, M, mode)}</div>')
    with right:
        n = len(chosen)
        l_i2t = xent_diag(logits)
        l_t2i = xent_diag(logits.T)
        acc = int((S.argmax(axis=1) == np.arange(n)).sum())
        with st.container(border=True):
            st.markdown(f"<span style='font-weight:700;color:{NAVY};font-size:17px'>InfoNCE loss</span>",
                        unsafe_allow_html=True)
            st.latex(r"\mathcal{L}_i=-\log\frac{e^{\,\mathrm{sim}(q_i,k_i^+)/\tau}}"
                     r"{\sum_{j=1}^{N} e^{\,\mathrm{sim}(q_i,k_j)/\tau}}")
            html(f"""
            <div class="bar"><div class="track"><div class="fill" style="width:{min(l_i2t / np.log(n), 1) * 100:.0f}%;background:{IMG_BG}"></div>
              <span class="lab">image → text</span></div><div class="v">{l_i2t:.3f}</div></div>
            <div class="bar"><div class="track"><div class="fill" style="width:{min(l_t2i / np.log(n), 1) * 100:.0f}%;background:{TXT_BG}"></div>
              <span class="lab">text → image</span></div><div class="v">{l_t2i:.3f}</div></div>
            <div class="bar"><div class="track"><div class="fill" style="width:{min((l_i2t + l_t2i) / 2 / np.log(n), 1) * 100:.0f}%;background:{SPACE_BG}"></div>
              <span class="lab"><b>total (average)</b></span></div><div class="v">{(l_i2t + l_t2i) / 2:.3f}</div></div>
            <div style="font-size:13px;color:{MUTED};margin-top:8px">random guessing = ln {n} = {np.log(n):.3f}<br>
            images that pick their own caption: <b style="color:{NAVY}">{acc} / {n}</b></div>""")

    key_idea(f"Training pushes the <b style='color:{POS}'>diagonal</b> up and every "
             f"<b style='color:{NEG}'>off-diagonal</b> cell down. The raw cosines differ only a little, "
             "so CLIP divides by a small <b>learned temperature</b> (about "
             f"{learned_tau:.3f}) to turn those small gaps into confident probabilities. Slide τ to see it.")

    code_panel("the symmetric CLIP loss", '''\
import torch
import torch.nn.functional as F

# logit_scale is a learned parameter, initialized to log(1 / 0.07)
logit_scale = torch.nn.Parameter(torch.tensor(1 / 0.07).log())

def clip_loss(img, txt):                 # both [N, 512], unit length
    logits = logit_scale.exp() * img @ txt.T     # [N, N] = cosine / tau
    labels = torch.arange(len(img))              # image i matches text i
    loss_i2t = F.cross_entropy(logits, labels)   # each row: pick the right text
    loss_t2i = F.cross_entropy(logits.T, labels) # each column: pick the right image
    return (loss_i2t + loss_t2i) / 2

# After training, CLIP's logit_scale.exp() is about 100, so tau is about 0.01.''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 · ZERO-SHOT
# ═════════════════════════════════════════════════════════════════════════════
TEMPLATES = {
    "a photo of a {}.": "a photo of {a} {}.",
    "{}  (bare name)": "{}",
    "a picture of a {}.": "a picture of {a} {}.",
    "a blurry photo of a {}.": "a blurry photo of {a} {}.",
    "a drawing of a {}.": "a drawing of {a} {}.",
}
ENSEMBLE = "Ensemble (average of all)"


def fill(template, cls):
    return template.replace("{a}", article(cls)).replace("{}", cls)


def class_embeddings(classes, choice):
    if choice == ENSEMBLE:
        E = np.stack([embed_texts([fill(t, c) for t in TEMPLATES.values()]).mean(axis=0) for c in classes])
        return E / np.linalg.norm(E, axis=1, keepdims=True)
    return embed_texts([fill(TEMPLATES[choice], c) for c in classes])


def bars_html(names, values, colors, fmt, scale_max=None):
    m = scale_max or max(max(values), 1e-6)
    out = ""
    for nm, v, c in zip(names, values, colors):
        w = max(v, 0) / m * 100
        out += (f'<div class="bar"><div class="track"><div class="fill" style="width:{w:.0f}%;background:{c}"></div>'
                f'<span class="lab">{esc(nm)}</span></div><div class="v">{fmt(v)}</div></div>')
    return out


def sec_zero_shot():
    hero(5, "Zero-shot classification", "No task-specific training: compare the image with natural-language class prompts")
    if need_images():
        return

    c1, c2, c3 = st.columns([1, 1.2, 1.2])
    with c1:
        lbl = pick_image("s5_img")
    with c2:
        classes_txt = st.text_area("Class names (one per line)", value="\n".join(LABELS),
                                   height=150, key="s5_classes")
    with c3:
        choice = st.radio("Prompt template", list(TEMPLATES) + [ENSEMBLE], key="s5_tpl")

    classes = list(dict.fromkeys(c.strip() for c in classes_txt.splitlines() if c.strip()))
    if len(classes) < 2:
        st.info("Enter at least two class names.")
        return

    with st.spinner("Classifying..."):
        img = embed_image(POOL[lbl]["bytes"])
        E = class_embeddings(classes, choice)
    scores = E @ img
    scale = logit_scale()
    probs = softmax(scores * scale)
    order = np.argsort(-scores)
    top = int(order[0])

    shown = order[:7]
    prompt_list = classes if choice == ENSEMBLE else [fill(TEMPLATES[choice], c) for c in classes]
    prompt_chips = "".join(f'<div class="tk" style="display:block;margin:4px 0;text-align:left">"{esc(prompt_list[i])}"</div>'
                           for i in shown[:5])
    if len(classes) > 5:
        prompt_chips += f'<div style="color:{MUTED};font-size:13px">+ {len(classes) - 5} more</div>'
    if choice == ENSEMBLE:
        prompt_chips += f'<div style="color:{MUTED};font-size:13px">each class = average of {len(TEMPLATES)} templates</div>'

    bar_colors = [POS_BG if i == top else "#E2E8F0" for i in shown]
    sim_bars = bars_html([classes[i] for i in shown], [scores[i] for i in shown], bar_colors, lambda v: f"{v:.3f}")
    vec = ", ".join(f"{scores[i]:.2f}" for i in shown[:3])

    html(f"""
    <div class="steps">
      <div class="step"><h4 style="background:{IMG_BG};color:{IMG}">1 · Encode the image</h4>
        <div class="pipe" style="justify-content:center">
          <img class="thumb" src="{thumb_uri(POOL[lbl]["bytes"], 200)}" width="120"/>
          <div class="node n-img">Image<br>encoder ❄️<small>frozen</small></div></div>
        <div style="text-align:center;margin-top:10px">{strip_html(img, IMG, 28)}<div class="slabel">one 512-d vector</div></div></div>
      <div class="step"><h4 style="background:{TXT_BG};color:#0B7A55">2 · Encode class prompts</h4>
        {prompt_chips}
        <div class="node n-txt" style="margin-top:8px">Text encoder ❄️<small>one vector per class</small></div></div>
      <div class="step"><h4 style="background:{SPACE_BG};color:{SPACE}">3 · Cosine similarity</h4>
        {sim_bars}
        <div class="slabel" style="text-align:center">higher = closer in the shared space</div></div>
      <div class="step"><h4 style="background:{PROJ_BG};color:#B45309">4 · Argmax → prediction</h4>
        <div style="text-align:center;font-family:JetBrains Mono,monospace;font-size:13px;color:{MUTED}">argmax([{vec}, …])</div>
        <div style="text-align:center;font-size:30px;font-weight:800;color:{PROJ};margin:14px 0 4px">{esc(classes[top])}</div>
        <div style="text-align:center;color:{MUTED};font-size:14px">softmax probability</div>
        <div style="text-align:center;font-size:26px;font-weight:800;color:{NAVY}">{probs[top] * 100:.1f}%</div>
        <div class="slabel" style="text-align:center">cosines × learned scale ({scale:.0f}), then softmax</div></div>
    </div>""")

    st.markdown("")
    with st.expander("🧪  Experiment: does prompt wording change the answer?", expanded=True):
        rows = []
        for name in list(TEMPLATES) + [ENSEMBLE]:
            p = softmax(class_embeddings(classes, name) @ img * scale)
            k = int(p.argmax())
            rows.append((name, classes[k], float(p[k]), float(p[top])))
        body = "".join(
            f'<tr><td style="padding:6px 10px;font-family:JetBrains Mono,monospace;font-size:13.5px">{esc(n)}</td>'
            f'<td style="padding:6px 10px;font-weight:700;color:{POS if c == classes[top] else NEG}">{esc(c)}</td>'
            f'<td style="padding:6px 10px;width:45%">{bars_html([f"P({classes[top]})"], [pt], [POS_BG], lambda v: f"{v * 100:.1f}%", 1.0)}</td></tr>'
            for n, c, pc, pt in rows)
        html(f'<table style="width:100%;border-collapse:collapse"><tr style="color:{MUTED};font-size:13px;text-align:left">'
             f'<th style="padding:4px 10px">template</th><th style="padding:4px 10px">prediction</th>'
             f'<th style="padding:4px 10px">confidence in "{esc(classes[top])}"</th></tr>{body}</table>')

    key_idea("After training, you classify a new image by <b>writing class prompts</b> and picking the most "
             f"similar one. <b style='color:{IMG}'>No fine-tuning</b> and <b style='color:{NEG}'>no labeled examples</b>. "
             "Sentence-style prompts and prompt ensembles usually beat bare class names.")

    code_panel("zero-shot classification", '''\
classes  = ["cat", "bear", "stop sign"]
prompts  = [f"a photo of a {c}." for c in classes]

img = model.get_image_features(**processor(images=image, return_tensors="pt"))
txt = model.get_text_features(**processor(text=prompts, return_tensors="pt", padding=True))
img = img / img.norm(dim=-1, keepdim=True)
txt = txt / txt.norm(dim=-1, keepdim=True)

logits = model.logit_scale.exp() * img @ txt.T   # learned scale, about 100
probs  = logits.softmax(dim=-1)
print(classes[probs.argmax()])''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 · EMBEDDING SPACE
# ═════════════════════════════════════════════════════════════════════════════
def sec_space():
    hero(6, "Inside the shared space", "Where do the image and text vectors actually land?")
    if need_images(3):
        return

    c1, c2, c3 = st.columns([2.4, 1, 1.3])
    with c1:
        chosen = st.multiselect("Image-caption pairs (3 to 8)", LABELS, default=LABELS[:6],
                                max_selections=8, key="s6_imgs")
    with c2:
        method = seg("Projection", ["PCA", "t-SNE"], key="s6_method")
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        center = st.toggle("Close the modality gap", key="s6_center",
                           help="Subtract the mean image vector from every image and the mean text "
                                "vector from every text, then re-normalize.")
    if len(chosen) < 3:
        st.info("Pick at least three pairs.")
        return

    captions = [POOL[l]["caption"] for l in chosen]
    with st.spinner("Encoding..."):
        I = embed_images([POOL[l]["bytes"] for l in chosen])
        T = embed_texts(captions)
    raw_gap = float(np.linalg.norm(I.mean(0) - T.mean(0)))
    if center:
        I = I - I.mean(0)
        T = T - T.mean(0)
        I /= np.linalg.norm(I, axis=1, keepdims=True)
        T /= np.linalg.norm(T, axis=1, keepdims=True)

    n = len(chosen)
    X = np.vstack([I, T])
    if method == "PCA":
        XY = PCA(n_components=2, random_state=0).fit_transform(X)
    else:
        XY = TSNE(n_components=2, perplexity=max(2, min(5, 2 * n - 1)), random_state=42,
                  init="pca", learning_rate="auto").fit_transform(X)

    fig, ax = plt.subplots(figsize=(9.5, 6.3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFD")
    for s in ax.spines.values():
        s.set_edgecolor("#D9E1EC")
    ax.set_xticks([])
    ax.set_yticks([])

    for i, l in enumerate(chosen):
        c = PAIR_COLORS[i % 8]
        (ix, iy), (tx, ty) = XY[i], XY[n + i]
        ax.plot([ix, tx], [iy, ty], color=c, lw=1.6, ls="--", alpha=0.6, zorder=1)
        thumb = np.asarray(clip_crop(to_pil(POOL[l]["bytes"]), 52))
        ax.add_artist(AnnotationBbox(OffsetImage(thumb, zoom=1.0), (ix, iy), frameon=True,
                                     bboxprops=dict(edgecolor=c, linewidth=2.5, boxstyle="round,pad=0.1"),
                                     zorder=4))
        ax.scatter(tx, ty, s=260, color=c, edgecolors="white", linewidths=2, zorder=5)
        ax.annotate(l, (tx, ty), xytext=(10, -4), textcoords="offset points", fontsize=10, color=c,
                    fontweight="bold", zorder=6)

    ci, ct = XY[:n].mean(0), XY[n:].mean(0)
    ax.scatter(*ci, marker="X", s=320, color=IMG, edgecolors="white", linewidths=2, zorder=3)
    ax.scatter(*ct, marker="X", s=320, color=TXT, edgecolors="white", linewidths=2, zorder=3)
    if not center:
        ax.annotate("", xy=ct, xytext=ci, arrowprops=dict(arrowstyle="<->", color=SPACE, lw=2.2), zorder=2)
        mid = (ci + ct) / 2
        ax.text(mid[0], mid[1], "  modality gap", color=SPACE, fontsize=11, fontweight="bold", zorder=2)

    pad = 0.18 * (XY.max(0) - XY.min(0) + 1e-6)
    ax.set_xlim(XY[:, 0].min() - pad[0], XY[:, 0].max() + pad[0])
    ax.set_ylim(XY[:, 1].min() - pad[1], XY[:, 1].max() + pad[1])
    ax.set_xlabel("PC 1" if method == "PCA" else "t-SNE 1", color="#475569")
    ax.set_ylabel("PC 2" if method == "PCA" else "t-SNE 2", color="#475569")
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#94A3B8", markersize=11, label="image (thumbnail)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94A3B8", markersize=11, label="caption"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=IMG, markersize=12, label="image centroid"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=TXT, markersize=12, label="text centroid"),
        Line2D([0], [0], ls="--", color="#94A3B8", label="matching pair"),
    ], loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="#D9E1EC")
    plt.tight_layout()

    off = ~np.eye(n, dtype=bool)
    pair_sim = float(np.mean(np.sum(I * T, axis=1)))
    ii = float((I @ I.T)[off].mean())
    tt = float((T @ T.T)[off].mean())

    left, right = st.columns([2.6, 1])
    with left:
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with right:
        chips([(f"{pair_sim:.2f}", "image ↔ its own caption"),
               (f"{ii:.2f}", "image ↔ other images"),
               (f"{tt:.2f}", "text ↔ other texts"),
               (f"{raw_gap:.2f}", "distance between centroids (512-d)")])
        if not center:
            msg = ("Each image is <b>more similar to the other images</b> than to its own caption."
                   if ii > pair_sim else "Pairs are closer than same-modality neighbors here.")
            html(f'<div class="panel" style="font-size:15px">{msg}</div>')

    if center:
        key_idea("With each modality's mean removed, images and captions <b>mix together</b> and each image "
                 "sits near its own caption. The alignment was there all along; the gap is a constant offset.")
    else:
        key_idea(f"CLIP keeps images and texts in <b style='color:{SPACE}'>two separate regions</b> of the sphere "
                 "(the <b>modality gap</b>, Liang et al. 2022). Matching works because pairs are closer "
                 "<i>relative to the other captions</i>, not because they overlap. Flip the toggle to remove the gap.")

    with st.expander("🔍  Full similarity heatmap (images and texts)"):
        names = [f"IMG · {l}" for l in chosen] + [f"TXT · {l}" for l in chosen]
        A = X @ X.T
        f2, a2 = plt.subplots(figsize=(9, 7.4))
        im = a2.imshow(A, cmap="Blues", vmin=float(A[~np.eye(2 * n, dtype=bool)].min()), vmax=1)
        for i in range(2 * n):
            for j in range(2 * n):
                a2.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if A[i, j] > 0.75 else "#1E293B")
        a2.set_xticks(range(2 * n), names, rotation=40, ha="right", fontsize=9)
        a2.set_yticks(range(2 * n), names, fontsize=9)
        a2.axhline(n - 0.5, color=SPACE, lw=2)
        a2.axvline(n - 0.5, color=SPACE, lw=2)
        f2.colorbar(im, ax=a2, fraction=0.046, pad=0.03)
        plt.tight_layout()
        st.pyplot(f2, clear_figure=True)
        plt.close(f2)
        st.caption("Top-left: image vs image. Bottom-right: text vs text. Top-right: the N×N matrix from "
                   "Section 4 (image rows, text columns). Without centering, the same-modality blocks are much darker.")

    code_panel("project the embeddings and measure the gap", '''\
X = np.vstack([img_embs, txt_embs])              # [2N, 512], unit length
xy = PCA(n_components=2).fit_transform(X)        # or TSNE(...) for small N

gap = np.linalg.norm(img_embs.mean(0) - txt_embs.mean(0))

# Close the gap: remove each modality's mean, then re-normalize
I = img_embs - img_embs.mean(0); I /= np.linalg.norm(I, axis=1, keepdims=True)
T = txt_embs - txt_embs.mean(0); T /= np.linalg.norm(T, axis=1, keepdims=True)''')


# ═════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
SECTIONS = {
    "1 · Big picture": sec_big_picture,
    "2 · Two encoders": sec_two_encoders,
    "3 · One pair": sec_one_pair,
    "4 · N×N matrix": sec_matrix,
    "5 · Zero-shot": sec_zero_shot,
    "6 · Embedding space": sec_space,
}
NAMES = list(SECTIONS)

if "sec" not in st.session_state:
    st.session_state.sec = NAMES[0]
if "last_sec" not in st.session_state:
    st.session_state.last_sec = NAMES[0]


def _keep_sec():
    if st.session_state.sec is None:          # segmented control was clicked off
        st.session_state.sec = st.session_state.last_sec
    st.session_state.last_sec = st.session_state.sec


def _go(delta):
    i = NAMES.index(st.session_state.sec) + delta
    st.session_state.sec = NAMES[max(0, min(len(NAMES) - 1, i))]
    st.session_state.last_sec = st.session_state.sec


try:
    nav_box = st.container(key="topnav")
except TypeError:                              # Streamlit < 1.39 has no container keys
    nav_box = st.container()
with nav_box:
    if hasattr(st, "segmented_control"):
        st.segmented_control("Section", NAMES, key="sec", on_change=_keep_sec, label_visibility="collapsed")
    else:
        st.radio("Section", NAMES, key="sec", on_change=_keep_sec, horizontal=True, label_visibility="collapsed")

current = st.session_state.sec or st.session_state.last_sec
SECTIONS[current]()

st.markdown("")
idx = NAMES.index(current)
b1, _, b2 = st.columns([1, 4, 1])
with b1:
    if idx > 0:
        st.button(f"← {NAMES[idx - 1][4:]}", on_click=_go, args=(-1,), key="prev")
with b2:
    if idx < len(NAMES) - 1:
        st.button(f"{NAMES[idx + 1][4:]} →", on_click=_go, args=(1,), key="next")

import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file (GEMINI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except ImportError:
    pass

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Deep-ML Explorer",
    page_icon="🧠",
    layout="wide"
)

# ── Higgsfield-inspired dark design ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
.stApp { background-color: #080808; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0D0D0D;
    border-right: 1px solid #1C1C1C;
}
[data-testid="stSidebar"] * { color: #E0E0E0; }

/* ── Sidebar inputs ── */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #161616 !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #AAFF00 !important;
    box-shadow: 0 0 0 2px rgba(170,255,0,0.15) !important;
}

/* ── Title gradient ── */
h1 {
    background: linear-gradient(135deg, #FFFFFF 30%, #AAFF00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

/* ── Section headers ── */
h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background-color: #111111;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1C1C1C;
}
[data-testid="stTabs"] [role="tab"] {
    color: #888888 !important;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    border: none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background-color: #1A1A1A !important;
    color: #AAFF00 !important;
    border: 1px solid #2A2A2A !important;
}

/* ── Expanders (steps) ── */
[data-testid="stExpander"] {
    background-color: #111111 !important;
    border: 1px solid #1E1E1E !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"]:hover {
    border-color: #AAFF00 !important;
    transition: border-color 0.2s ease;
}
[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary:hover { color: #AAFF00 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background-color: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetric"] label { color: #888888 !important; font-size: 13px !important; }
[data-testid="stMetricValue"] { color: #AAFF00 !important; font-weight: 700 !important; }

/* ── Code blocks ── */
[data-testid="stCode"], .stCode,
[data-testid="stCode"] pre {
    background-color: #0D0D0D !important;
    border: 1px solid #1E1E1E !important;
    border-radius: 8px !important;
}
code { color: #AAFF00 !important; }

/* ── Success / info / warning banners ── */
[data-testid="stAlert"][kind="success"],
.stSuccess {
    background-color: rgba(170,255,0,0.07) !important;
    border: 1px solid rgba(170,255,0,0.3) !important;
    border-radius: 10px !important;
    color: #AAFF00 !important;
}
[data-testid="stAlert"][kind="info"],
.stInfo {
    background-color: rgba(255,255,255,0.04) !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 10px !important;
    color: #CCCCCC !important;
}
[data-testid="stAlert"][kind="warning"],
.stWarning {
    background-color: rgba(255,170,0,0.07) !important;
    border: 1px solid rgba(255,170,0,0.3) !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: #1E1E1E !important; }

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    background-color: #111111 !important;
    border: 1px solid #1E1E1E !important;
    border-radius: 10px !important;
}

/* ── Slider track & thumb ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: #AAFF00 !important;
    border-color: #AAFF00 !important;
}

/* ── Caption text ── */
.stCaption { color: #555555 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #AAFF00 !important; }

/* ── Sidebar section labels ── */
[data-testid="stSidebar"] .stSubheader { color: #AAFF00 !important; }
</style>
""", unsafe_allow_html=True)

from problems import PROBLEM_REGISTRY
from gemini_generator import generate_with_gemini

# Dark matplotlib theme to match the UI
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use("dark_background")
mpl.rcParams.update({
    "axes.facecolor":   "#111111",
    "figure.facecolor": "#111111",
    "axes.edgecolor":   "#2A2A2A",
    "axes.labelcolor":  "#CCCCCC",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "grid.color":       "#1E1E1E",
    "text.color":       "#FFFFFF",
    "legend.facecolor": "#161616",
    "legend.edgecolor": "#2A2A2A",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def display_value(val, label=""):
    import pandas as pd
    prefix = f"**{label}:** " if label else ""
    if isinstance(val, np.ndarray):
        if val.ndim == 1:
            st.write(f"{prefix}`{np.round(val, 4).tolist()}`")
        elif val.ndim == 2:
            if label:
                st.write(f"**{label}:**")
            st.dataframe(pd.DataFrame(np.round(val, 4)), use_container_width=False)
    elif isinstance(val, float):
        st.write(f"{prefix}`{round(val, 6)}`")
    elif isinstance(val, (list, tuple)) and val:
        formatted = [round(v, 4) if isinstance(v, float) else v for v in val]
        st.write(f"{prefix}`{formatted}`")
    else:
        st.write(f"{prefix}`{val}`")


def display_answer(ans):
    if isinstance(ans, np.ndarray):
        st.success(f"✅ **Answer:** `{np.round(ans, 4).tolist()}`")
    elif isinstance(ans, float):
        st.success(f"✅ **Answer:** `{round(ans, 6)}`")
    else:
        st.success(f"✅ **Answer:** `{ans}`")


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        key = st.session_state.get("gemini_key", "").strip()
    return key


# ── Pre-built problem renderer ─────────────────────────────────────────────────

def render_prebuilt(prob, seed):
    example = prob["generate"](seed)
    tab1, tab2 = st.tabs(["📋 Example & Steps", "📊 Visualization"])

    with tab1:
        st.subheader("📥 Inputs")
        for k, v in example.get("inputs", {}).items():
            display_value(v, k)

        st.subheader("🔢 Step-by-Step Solution")
        for i, step in enumerate(prob["steps"](example)):
            with st.expander(f"Step {i+1}: {step['title']}", expanded=True):
                st.write(step["explanation"])
                if "math" in step:
                    st.latex(step["math"])
                if "result" in step:
                    display_value(step["result"], "Result")
                if "code" in step:
                    st.code(step["code"], language="python")

        if "answer" in example:
            display_answer(example["answer"])

    with tab2:
        import matplotlib.pyplot as plt
        fig = prob["visualize"](example)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No visualization available for this problem.")


# ── AI-generated problem renderer ─────────────────────────────────────────────

def render_ai_generated(concept: str, api_key: str):
    with st.spinner(f"Generating explanation for **{concept}**…"):
        result = generate_with_gemini(concept, api_key)

    if "error" in result:
        st.error(f"Generation failed: {result['error']}")
        return

    # Header info
    col1, col2 = st.columns([4, 1])
    col1.markdown(result.get("description", ""))
    col2.markdown(f"**Category:** {result.get('category', '—')}")
    diff_icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(result.get("difficulty", ""), "⚪")
    col2.markdown(f"**Difficulty:** {diff_icon} {result.get('difficulty', '—')}")

    st.divider()

    # Inputs
    st.subheader("📥 Inputs")
    for k, v in result.get("inputs", {}).items():
        st.write(f"**{k}:** `{v}`")

    # Steps
    st.subheader("🔢 Step-by-Step Solution")
    for i, step in enumerate(result.get("steps", [])):
        with st.expander(f"Step {i+1}: {step.get('title', '')}", expanded=True):
            st.write(step.get("explanation", ""))
            if step.get("math"):
                try:
                    st.latex(step["math"])
                except Exception:
                    st.code(step["math"])
            if step.get("result"):
                st.write(f"**Result:** `{step['result']}`")
            if step.get("code"):
                st.code(step["code"], language="python")

    if result.get("answer"):
        st.success(f"✅ **Answer:** `{result['answer']}`")

    st.caption("⚡ Generated by Gemini 1.5 Flash — cached for this session")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🧠 Deep-ML Problem Explorer")
    st.markdown(
        "*Type **any** concept from Deep-ML → get a worked example, "
        "step-by-step solution with real numbers, and a visualization.*"
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("🔍 Search")
    concept_input = st.sidebar.text_input(
        "Type any concept",
        placeholder="e.g. Batch Normalisation, Attention, PCA…"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Gemini API Key")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        entered = st.sidebar.text_input(
            "Paste your key here",
            type="password",
            help="Get a free key at aistudio.google.com",
            key="gemini_key_input"
        )
        if entered:
            st.session_state["gemini_key"] = entered
            api_key = entered
    else:
        st.sidebar.success("✅ API key loaded from .env")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Pre-built Problems")
    cats = ["All"] + sorted(set(p["category"] for p in PROBLEM_REGISTRY.values()))
    cat  = st.sidebar.selectbox("Filter by category", cats)

    q = concept_input.strip().lower()
    filtered = {
        name: p for name, p in PROBLEM_REGISTRY.items()
        if (cat == "All" or p["category"] == cat)
        and (
            not q
            or q in name.lower()
            or q in p["description"].lower()
            or any(q in t.lower() for t in p.get("tags", []))
        )
    }

    # ── Routing ───────────────────────────────────────────────────────────────
    if not concept_input.strip():
        # Landing state
        st.info("👈 Type any concept name in the sidebar to get started.")
        st.markdown("### Pre-built problems (instant, no API needed):")
        for cat_name in sorted(set(p["category"] for p in PROBLEM_REGISTRY.values())):
            names = [n for n, p in PROBLEM_REGISTRY.items() if p["category"] == cat_name]
            st.markdown(f"**{cat_name}:** " + " · ".join(f"`{n}`" for n in sorted(names)))
        return

    if filtered:
        # Found in pre-built library
        best_match = sorted(
            filtered.keys(),
            key=lambda n: (0 if q in n.lower() else 1)
        )[0]

        st.sidebar.markdown(f"✅ Found in pre-built library")
        selected = st.sidebar.selectbox("Matched problems", sorted(filtered.keys()), index=sorted(filtered.keys()).index(best_match))
        seed = st.sidebar.slider("Example Seed", 0, 99, 42)

        prob = PROBLEM_REGISTRY[selected]
        diff_icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(prob.get("difficulty", ""), "⚪")

        col1, col2 = st.columns([4, 1])
        col1.header(selected)
        col1.markdown(prob["description"])
        col2.markdown(f"**Category:** {prob['category']}")
        col2.markdown(f"**Difficulty:** {diff_icon} {prob.get('difficulty', '—')}")
        st.divider()

        render_prebuilt(prob, seed)

    else:
        # Not in pre-built → use Gemini
        st.header(concept_input.strip())

        if not api_key:
            st.warning(
                "This concept isn't in the pre-built library. "
                "To generate it with AI, paste your **Gemini API key** in the sidebar.\n\n"
                "Get a free key at **aistudio.google.com** (no credit card needed)."
            )
            return

        st.sidebar.info("💡 Not in library — generating with Gemini AI")
        render_ai_generated(concept_input.strip(), api_key)


if __name__ == "__main__":
    main()

import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Deep-ML Explorer",
    page_icon="🧠",
    layout="wide"
)

from problems import PROBLEM_REGISTRY


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


def main():
    st.title("🧠 Deep-ML Problem Explorer")
    st.markdown(
        "*Type any concept name → get a concrete worked example, "
        "step-by-step solution with real numbers, and a visualization.*"
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("🔍 Find a Concept")
    query = st.sidebar.text_input("Search", placeholder="e.g. Softmax, F1 Score, Jacobian…")

    cats = ["All"] + sorted(set(p["category"] for p in PROBLEM_REGISTRY.values()))
    cat = st.sidebar.selectbox("Category", cats)

    q = query.strip().lower()
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

    if not filtered:
        st.sidebar.warning("No matches found.")
        st.warning("No concept found — try a different search term or broaden the category.")
        return

    st.sidebar.markdown("---")
    selected = st.sidebar.selectbox("Problem", sorted(filtered.keys()))
    seed = st.sidebar.slider("Example Seed", 0, 99, 42,
                             help="Change to get a different random example")

    prob = PROBLEM_REGISTRY[selected]

    # ── Header ───────────────────────────────────────────────────────────────
    diff_icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(prob.get("difficulty", ""), "⚪")
    col1, col2 = st.columns([4, 1])
    col1.header(selected)
    col1.markdown(prob["description"])
    col2.markdown(f"**Category:** {prob['category']}")
    if "difficulty" in prob:
        col2.markdown(f"**Difficulty:** {diff_icon} {prob['difficulty']}")

    st.divider()

    # ── Generate example ─────────────────────────────────────────────────────
    try:
        example = prob["generate"](seed)
    except Exception as e:
        st.error(f"Error generating example: {e}")
        return

    tab1, tab2 = st.tabs(["📋 Example & Steps", "📊 Visualization"])

    with tab1:
        st.subheader("📥 Inputs")
        for k, v in example.get("inputs", {}).items():
            display_value(v, k)

        st.subheader("🔢 Step-by-Step Solution")
        for i, step in enumerate(prob["steps"](example)):
            with st.expander(f"Step {i + 1}: {step['title']}", expanded=True):
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
        try:
            fig = prob["visualize"](example)
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No visualization available for this problem.")
        except Exception as e:
            st.error(f"Visualization error: {e}")


if __name__ == "__main__":
    main()

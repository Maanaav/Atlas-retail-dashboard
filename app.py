import streamlit as st
import importlib
from pathlib import Path

st.set_page_config(
    page_title="Atlas – Retail Patterns Dashboard",
    page_icon="🛒",
    layout="wide",
)

# Hard check: is plotly installed in this environment?
if importlib.util.find_spec("plotly") is None:
    st.error("plotly is NOT installed in this Streamlit Cloud environment. "
             "Cloud is not picking up your requirements.txt.")
    st.stop()

from tabs import overview, patterns, time_tab, insights

# ----------------- GLOBAL STYLES -----------------
st.markdown(
    """
<style>
/* App background */

/* Tighten padding in the main area */
.block-container {
    padding-bottom: 1.5rem;
}

/* --- SIDEBAR --- */
[data-testid="stSidebar"] {
    background: #c2dcff;
    color: #173257;
    padding-top: 1.2rem;
    padding-bottom: 1.5rem;
    border-right: 1px solid rgba(80,80,160,0.09);
    transition: background 0.2s;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5 {
    color: #173257;
}

/* Navigation label text */
.sidebar-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    opacity: 0.7;
    color: #1a3660;
    margin-top: 0.7rem;
    margin-bottom: 0.35rem;
}

/* --- NAVIGATION PILL GAP --- */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 1.15rem;
    display: flex;
    flex-direction: column;
}

/* Hide the default radio dots */
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:nth-child(1) {
    display: none;
}

/* Base pill style with pale blue accent */
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    border-radius: 999px;
    padding: 0.54rem 1.05rem;
    border: 1.5px solid #98b4e8;
    background: linear-gradient(135deg, #e6f2ff 80%, #c2dcff 120%);
    display: flex;
    align-items: center;
    gap: 0.54rem;
    cursor: pointer;
    font-size: 0.90rem;
    font-weight: 500;
    color: #264267 !important;
    transition: border 0.15s, box-shadow 0.15s;
    box-shadow: none;
}

/* Icon span inside the pill */
.nav-icon {
    font-size: 0.95rem;
}

/* Hover state */
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    border-color: #47a2da;
    box-shadow: 0 0 0 2px #c2dcff;
    background: #e9f5ff;
}

/* Active state – the label containing a checked radio input */
[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"] {
    background: linear-gradient(90deg, #64b1f4 0%, #c2dcff 100%);
    border-color: #64b1f4;
    color: #173257 !important;
    box-shadow: 0 2px 16px -2px #8cc5f7;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"] .nav-text {
    font-weight: 600;
    color: #183454 !important;
}

/* Badges row */
.atlas-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 0.8rem;
}
.atlas-badge {
    font-size: 0.65rem;
    padding: 4px 9px;
    border-radius: 999px;
    border: 1px solid #6aa5e7;
    background: rgba(202,220,255,0.9);
    color: #193b7c !important;
    opacity: 1;
}

/* Slightly tighter global headers */
h1, h2, h3 {
    letter-spacing: -0.02em;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR LAYOUT -----------------
with st.sidebar:
    # logo + title
    cols = st.columns([1, 2.2])
    with cols[0]:
        st.image("assets/logo.png", width=60)
    with cols[1]:
        st.markdown(
            """
**ATLAS**  
<span style="font-size:0.75rem; opacity:0.72; color:#264267;">Retail Patterns Dashboard</span>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)

    # radio used for state; visual styling done via CSS above
    tab = st.radio(
        "View",
        [
            "🏠 Overview",
            "🧺 Basket Patterns",
            "⏱ Time Dynamics",
            "💡 Insights",
        ],
        label_visibility="collapsed",
        index=0,
    )

    st.markdown('<div class="sidebar-label">Sources</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="atlas-badges">
  <div class="atlas-badge">UCI Retail II (UK)</div>
  <div class="atlas-badge">ABS 8501.0</div>
  <div class="atlas-badge">Streamlit Prototype</div>
</div>
""",
        unsafe_allow_html=True,
    )

# ----------------- MAIN CONTENT ROUTING -----------------
clean_tab = tab.split(" ", 1)[1]  # strip emoji prefix

if clean_tab == "Overview":
    overview.show()
elif clean_tab == "Basket Patterns":
    patterns.show()
elif clean_tab == "Time Dynamics":
    time_tab.show()
else:
    insights.show()


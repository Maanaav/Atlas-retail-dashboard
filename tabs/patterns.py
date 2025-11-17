# tabs/patterns.py
# Basket Patterns (UK rules only) — segment-aware, ID-free, exploration-focused

import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RULES_CSV = Path("data/all_segment_rules_with_industry.csv")

PALETTE = {
    "blue": "#264E86",
    "green": "#2CA58D",
    "orange": "#F4A300",
    "grey": "#8F9BA6",

    # Network-specific tones
    "node_base": "#8DD3C7",      # soft teal for default nodes
    "node_neighbor": "#1D9A6C",  # richer teal for neighbours
    "node_focus": "#F97316",     # warm orange for selected node
    "node_dim": "#CBD5E1",       # light slate for de-emphasised nodes
    "edge_light": "#E5E7EB",     # light grey edges
}


SCATTER_SCALE = [
    "#E5EDF9",  # very light blue
    "#86A8E7",  # mid blue
    "#264E86",  # deep blue (matches brand)
]

SEGMENT_MAP_DEFAULT = {
    1: "Lost Customers",
    2: "High-value Customers",
    3: "New Customers",
    4: "Lapsing Customers"
}

SEGMENT_ORDER = [
    "New Customers",
    "High-value Customers",
    "Lapsing Customers",
    "Lost Customers",
]

SEGMENT_COLOR_MAP = {
    "New Customers": PALETTE["blue"],
    "High-value Customers": PALETTE["green"],
    "Lapsing Customers": PALETTE["orange"],
    "Lost Customers": PALETTE["grey"],
}

# --- Helper UI utilities ------------------------------------------------------


def _popover(label: str, body_md: str, icon: str = "ℹ️"):
    """
    Renders a compact helper. Uses st.popover when available; falls back to expander.
    """
    if hasattr(st, "popover"):
        with st.popover(f"{icon} {label}", use_container_width=True):
            st.markdown(body_md)
    else:
        with st.expander(f"{icon} {label}"):
            st.markdown(body_md)


def _glossary():
    _popover(
        "Glossary",
        """
**Support** — share of all baskets that contain *both* the left-hand side (antecedent) and right-hand side (consequent).  
**Confidence** — probability of seeing the consequent **given** the antecedent.  
**Lift** — how many times more often the rule happens than random chance ( >1 is good ).  
**Antecedent / Consequent** — “IF … THEN …” parts of a rule (e.g., *IF* Gift Bag *THEN* Greeting Card).  
**Redundant rule** — a rule that adds no new information compared to a simpler rule with similar or higher metrics.
        """,
        icon="📚",
    )


def _filters_help():
    _popover(
        "About the rule filters",
        """
- **Min Lift**: *How special is this combo?*  
Higher lift means the products show up together much more often than you’d expect by luck.  
- **Min Confidence**: *How reliably does B follow A?*  
  When the left-hand product is in a basket, confidence is “how often does the right-hand product also appear?”.
- **Min Support**: *How common is this pattern overall?*  
  Support is the share of all baskets that contain the whole rule. Move this up if you want to ignore quirky one-off baskets and keep only patterns lots of customers follow.

**Why limit Lift to 50?**  
Very extreme lift values usually come from tiny numbers of baskets (e.g. 2 people who bought a weird combo). Capping the slider stops those “cute but useless” quirks from dominating your view.


        """,
        icon="🎚️",
    )


def _how_rules_made():
    _popover(
        "How we built these rules",
        """
**Dataset**: UCI Online Retail II (UK), cleaned to **UK-only**, removed **returns** (Invoice starts with `C`) and lines with non-positive quantity/price.  
**Basketisation**: each invoice → a shopping *basket* of products.  
**Apriori**: `min_support ≥ 0.002`, `min_confidence ≥ 0.40`, `min_lift ≥ 1.20`.  
**Pruning**: dropped duplicates & near-duplicates; kept strongest rules by **lift**.  
**Export**: `antecedent, consequent, support, confidence, lift, and industry`.  

**Categories** (if shown): product descriptions were mapped to a handful of readable groups (e.g., *Apparel*, *Household*, *Gifts*) via keyword rules; this is for **storytelling only** and doesn’t affect the mining.
        """,
        icon="🧪",
    )


def _network_help():
    _popover(
        "How to read the network",
        """
- **Nodes** are products. **Edges** connect products that appear together in strong rules.  
- **Size = node strength**: sum of connected rule lifts → bigger nodes = more “network-central” products.  
- **Search product**: shows its neighbourhood to reveal bundles and companion items.

**“Top nodes by strength”**  
This keeps only the most connected items (by total lift of edges touching them).  
It declutters the view while preserving the *structure* of popular bundles.
        """,
        icon="🕸️",
    )


def _scatter_help():
    _popover(
        "How to read this chart",
        """
- **Each dot is one rule.**
- **Big, bright dots in the top-right are your most interesting patterns.**
- Further right = appears in more baskets (higher support).  
- Higher up = more reliable (higher confidence).
- Larger / warmer dots = stronger lift.  
        """,
        icon="📊",
    )

# ---------- helpers ----------


def _strip_id(token: str) -> str:
    s = str(token).strip()
    if "|" in s:
        _, right = s.split("|", 1)
        s = right
    s = re.sub(r"^\s*\d+\s*[-:]\s*", "", s)
    s = s.strip()
    return s.title()


def _split_items(s: str):
    return [_strip_id(x) for x in str(s).split(",") if x.strip()]


def _label_segment(x, segmap):
    """Map ints like 1,2,3 → descriptive labels; pass through strings."""
    try:
        xi = int(str(x).strip())
        return segmap.get(xi, f"Segment {xi}")
    except ValueError:
        return str(x)


# ---------- data load/clean ----------

@st.cache_data(show_spinner=False)
def load_rules(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    seg_col = next((c for c in ["segment", "customer_segment",
                   "rfm_segment", "buyer_segment"] if c in df.columns), None)

    keep_cols = [c for c in ["antecedent", "consequent", "support",
                             "confidence", "lift", "industry"] if c in df.columns]
    if seg_col:
        keep_cols.append(seg_col)

    df = df[keep_cols].dropna(subset=["antecedent", "consequent"])

    for c in ("antecedent", "consequent"):
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(r"[\{\}\[\]\(\)\"']", "", regex=True)
                .str.replace("  ", " ", regex=False)
                .str.strip()
            )

    for c in ("support", "confidence", "lift"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["support", "confidence", "lift"])

    df["antecedent_clean"] = df["antecedent"].map(
        lambda s: ", ".join(_split_items(s)))
    df["consequent_clean"] = df["consequent"].map(
        lambda s: ", ".join(_split_items(s)))

    if seg_col:
        df.rename(columns={seg_col: "segment_raw"}, inplace=True)
        # build a segment map using defaults for ints, pass-through for strings
        # resolve label column now to keep UI clean everywhere
        df["segment"] = df["segment_raw"].map(
            lambda x: _label_segment(x, SEGMENT_MAP_DEFAULT))
        print(df.tail())

    return df


# ---------- main UI ----------

def show(DATA_DIR: Path = Path("data")):
    st.title("Basket Patterns: Co-Purchase Relationships")

    st.markdown(
        """
    <div style="background:#F8FAFC;border:1px solid #E5E7EB;padding:10px 14px;border-radius:10px;margin-top:6px;margin-bottom:16px">
      <b>What’s here?</b> You’re looking at products that are frequently bought together. 
      Use the filters on the left to tighten or relax the rules (support, confidence, lift), 
      scan the charts to see how strong and common those patterns are, 
      and use the network view to spot bundle items. 
      Start typing a product name in the <b>Search product</b> box to see its closest companions.
    </div>
    """,
        unsafe_allow_html=True,
    )

    df = load_rules(RULES_CSV)
    if df.empty:
        st.error(
            f"No rules found at {RULES_CSV}. Ensure Team A exported the file.")
        return

    # Quick explainer + glossary
    cols = st.columns([1, 1, 1])
    with cols[0]:
        _how_rules_made()
    with cols[1]:
        _filters_help()
    with cols[2]:
        _glossary()

    with st.sidebar:
        st.subheader("Rule Filters")
        _filters_help()  # helper inside the sidebar too (optional)
        # widened cap to 50 as requested
        lift_min = st.slider("Min Lift", 1.0, 50.0, 1.2, 0.1)
        conf_min = st.slider("Min Confidence", 0.0, 1.0, 0.40, 0.05)
        sup_min = st.slider("Min Support", 0.0, 0.05, 0.002, 0.001)

        industry_opt = None
        if "industry" in df.columns:
            inds = ["(all)"] + \
                sorted(df["industry"].astype(str).unique().tolist())
            industry_opt = st.selectbox("Industry (optional)", inds, index=0)

        segment_opt = None
        if "segment" in df.columns:
            segs = ["(all)"] + \
                sorted(df["segment"].astype(str).unique().tolist())
            segment_opt = st.selectbox(
                "Customer segment (optional)", segs, index=0)

    q = (df["lift"] >= lift_min) & (df["confidence"]
                                    >= conf_min) & (df["support"] >= sup_min)
    if industry_opt and industry_opt != "(all)" and "industry" in df.columns:
        q &= (df["industry"].astype(str) == industry_opt)
    if segment_opt and segment_opt != "(all)" and "segment" in df.columns:
        q &= (df["segment"].astype(str) == segment_opt)

    df_f = df.loc[q].copy()
    if df_f.empty:
        st.warning("No rules match current thresholds.")
        return

    # KPIs
    st.subheader("Key Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rules shown", f"{len(df_f):,}")
    c2.metric("Avg Lift", f"{df_f['lift'].mean():.2f}")
    c3.metric("Avg Confidence", f"{df_f['confidence'].mean():.2f}")
    c4.metric("Avg Support", f"{df_f['support'].mean():.3f}")

    # Segment legend + distribution
    if "segment" in df_f.columns:
        with st.expander("Customer segments"):
            st.markdown(
                """
                **Who these segments are**
                - **New Customers** – This group is brand new. They just made their first or second purchase, which is why their frequency and monetary values are still low.  
                - **High-value Customers** – These are the best customers. They shop often, spend a lot, and were just in the store around 2 weeks ago.  
                - **Lapsing Customers** – This is the most critical group. They were champions (they shopped often and spent a lot), but they haven’t been back in ~6 months.  
                - **Lost Customers** – This is the least valuable group. They shopped once or twice a very long time ago (almost a year!) and didn’t spend much.
                            """
            )
            seg_ct = (
                df_f.groupby("segment")["lift"]
                .agg(["size", "mean"])
                .reset_index()
                .rename(columns={"size": "rules", "mean": "avg_lift"})
            )

            # logical order
            present = seg_ct["segment"].astype(str).unique().tolist()
            ordered_categories = [s for s in SEGMENT_ORDER if s in present]
            seg_ct["segment"] = pd.Categorical(
                seg_ct["segment"],
                categories=ordered_categories,
                ordered=True,
            )
            seg_ct = seg_ct.sort_values("segment")

            fig_seg = px.bar(
                seg_ct,
                x="segment",
                y="rules",
                title="Rules per Segment",
                labels={"segment": "Segment", "rules": "Rules"},
                color_discrete_map=SEGMENT_COLOR_MAP,
                color="segment",
            )
            fig_seg.update_traces(
                hovertemplate=(
                    "<b>Segment:</b> %{x}<br>"
                    "<b>Rules:</b> %{y:,}"
                    "<extra></extra>"
                )
            )
            fig_seg.update_layout(
                xaxis_title="Segment",
                yaxis_title="Rules",
                margin=dict(l=0, r=0, t=60, b=0),
                legend_title_text="Customer segments",
            )
            st.plotly_chart(fig_seg, use_container_width=True)

    # Scatter: Support vs Confidence (clean hover only)
    st.subheader("Rule Distribution")
    _scatter_help()
    customdata = np.stack(
        [df_f["antecedent_clean"], df_f["consequent_clean"]], axis=-1)
    fig_sc = px.scatter(
        df_f, x="support", y="confidence", color="lift", size="lift",
        color_continuous_scale=SCATTER_SCALE,
        labels={"support": "Support",
                "confidence": "Confidence", "lift": "Lift"},
        title="How Often Rules Fire vs How Trustworthy They Are",
    )
    fig_sc.update_traces(
        hovertemplate=(
            "<b>Support:</b> %{x:.4f}<br>"
            "<b>Confidence:</b> %{y:.3f}<br>"
            "<b>Lift:</b> %{marker.color:.2f}<br><br>"
            "<b>Antecedent:</b> %{customdata[0]}<br>"
            "<b>Consequent:</b> %{customdata[1]}<extra></extra>"
        ),
        customdata=customdata,
        marker=dict(opacity=0.85, line=dict(width=0)),
    )
    fig_sc.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

    # Network (force layout; ID-free)
    st.subheader("Network View of Top Rules")
    _network_help()
    st.caption(
        "Tip: Start with 80–120 top nodes. Use “Search product” to reveal companion items.")

    df_net = df_f.sort_values("lift", ascending=False).head(3000)

    edges = []
    for _, r in df_net.iterrows():
        A = _split_items(r["antecedent"])
        C = _split_items(r["consequent"])
        if not A or not C:
            continue
        w = float(r["lift"])
        for a in A:
            for c in C:
                if a != c:
                    edges.append((a, c, w))
    if not edges:
        st.info("No edges to render with current filters.")
        return

    edges_df = (
        pd.DataFrame(edges, columns=["src", "dst", "lift"])
        .groupby(["src", "dst"], as_index=False)["lift"]
        .mean()
    )
    node_strength = (
        pd.concat(
            [
                edges_df[["src", "lift"]].rename(columns={"src": "item"}),
                edges_df[["dst", "lift"]].rename(columns={"dst": "item"}),
            ]
        )
        .groupby("item", as_index=False)["lift"]
        .sum()
        .rename(columns={"lift": "strength"})
    )

    top_n = st.slider("Top Nodes by Strength", 40, 250, 120, 10)
    keep_nodes = set(node_strength.sort_values(
        "strength", ascending=False).head(top_n)["item"])
    edges_df = edges_df[edges_df["src"].isin(
        keep_nodes) & edges_df["dst"].isin(keep_nodes)]
    node_strength = node_strength[node_strength["item"].isin(keep_nodes)]

    G = nx.Graph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["src"], r["dst"], weight=float(r["lift"]))

    pos = nx.spring_layout(
        G, k=0.6 / np.sqrt(max(len(G), 1)), iterations=200, seed=7)

    valid_nodes = [n for n in node_strength["item"] if n in G.nodes]
    ns = node_strength.set_index("item").loc[valid_nodes, "strength"]

    # edge trace
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", hoverinfo="none",
        line=dict(color=PALETTE["edge_light"], width=1),
    )

    # node trace
    size = 10 + 20 * (ns - ns.min()) / (ns.max() - ns.min() + 1e-9)
    node_x = [pos[n][0] for n in ns.index]
    node_y = [pos[n][1] for n in ns.index]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        textposition="top center",
        hovertext=[
            f"<b>Product:</b> {n}<br><b>Strength:</b> {ns[n]:.2f}"
            for n in ns.index
        ],
        hoverinfo="text",
        marker=dict(
            size=size,
            color=PALETTE["node_base"],
            line=dict(width=1, color=PALETTE["blue"]),
            opacity=0.95,
        ),
    )

    st.markdown(
        """
        <hr style="border:none;border-top:1px solid #E5E7EB;
                   margin:16px 0 26px 0;" />
        """,
        unsafe_allow_html=True,
    )

    # highlight
    hi_choice = st.selectbox("Search Product", ["(none)"] + list(ns.index))
    nbrs = None
    focus_nodes = list(ns.index)
    traces = [edge_trace, node_trace]
    if hi_choice != "(none)" and hi_choice in G:
        nbrs = set(G.neighbors(hi_choice)) | {hi_choice}
        colors = [
            PALETTE["node_focus"] if n == hi_choice
            else PALETTE["node_neighbor"] if n in nbrs
            else PALETTE["node_dim"]
            for n in ns.index
        ]
        opac = [1.0 if n in nbrs else 0.25 for n in ns.index]
        node_trace.marker.color = colors
        node_trace.marker.opacity = opac

        focus_nodes = list(nbrs)

        hi_x, hi_y = [], []
        for u, v in G.edges():
            if u in nbrs and v in nbrs:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                hi_x += [x0, x1, None]
                hi_y += [y0, y1, None]
        hi_edge = go.Scatter(x=hi_x, y=hi_y, mode="lines",
                             line=dict(color=PALETTE["node_focus"], width=2.5),
                             hoverinfo="none")
        traces = [edge_trace, hi_edge, node_trace]

    # auto labels when node is highlighted
    show_labels = hi_choice != "(none)"
    if show_labels:
        if nbrs:
            # label only the highlighted node and its neighbours
            node_trace.text = [n if n in nbrs else "" for n in ns.index]
        else:
            node_trace.text = list(ns.index)
        node_trace.mode = "markers+text"
    else:
        node_trace.text = None
        node_trace.mode = "markers"

    fig_net = go.Figure(data=traces)

    # zoom to highlighted network if selected
    xs = [pos[n][0] for n in focus_nodes if n in pos]
    ys = [pos[n][1] for n in focus_nodes if n in pos]
    if xs and ys:
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        x_pad = 0.2 * (x_span if x_span > 0 else 1.0)
        y_pad = 0.2 * (y_span if y_span > 0 else 1.0)
        fig_net.update_xaxes(range=[min(xs) - x_pad, max(xs) + x_pad])
        fig_net.update_yaxes(range=[min(ys) - y_pad, max(ys) + y_pad])

    fig_net.update_layout(
        title="Product Co-Purchase Network",
        showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=60, b=0), height=650,
    )
    st.plotly_chart(fig_net, use_container_width=True)

    with st.expander("Rule Table Preview"):
        cols = ["antecedent_clean", "consequent_clean",
                "support", "confidence", "lift"]
        if "industry" in df_f.columns:
            cols.append("industry")
        if "segment" in df_f.columns:
            cols.append("segment")

        rename_map = {
            "antecedent_clean": "Antecedent",
            "consequent_clean": "Consequent",
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
            "industry": "Industry",
            "segment": "Customer segment",
        }
        df_display = df_f[cols].rename(columns=rename_map)

        st.dataframe(df_display.head(200), use_container_width=True)

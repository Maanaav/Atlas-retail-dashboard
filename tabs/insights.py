import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

ABS_FILE = Path("data/ABS.csv")
RULES_FILE = Path("data/all_segment_rules_with_industry.csv")
LOGO_FILE = Path("assets/logo.png")

PALETTE = {
    "blue": "#264E86",
    "green": "#2CA58D",
    "orange": "#F4A300",
    "ink": "#0F172A",
}


# ---------- Helper explainers ----------

def insights_helpbar():
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1.popover("🧭 What this tab does", use_container_width=True):
        st.markdown(
            """
- Turns mined **co-purchase rules** + **ABS retail data** into actions.
- You can switch **audience focus** (store-ready vs analytics).
- Cards are opinionated summaries, not exploratory plots.
"""
        )
    with c2.popover("📂 How cards are filled", use_container_width=True):
        st.markdown(
            """
Each card follows a fixed template:

- **What we see** – one-sentence pattern.
- **Why it matters** – operational consequence.
- **Action** – imperative, ≤ 12 words.
- **Evidence** – support · confidence · lift (+ ABS stat if relevant).
- **Confidence** – High / Medium / Low based on lift and prevalence.
"""
        )
    with c3.popover("🛡️ Limits", use_container_width=True):
        st.markdown(
            """
- Rules are from **UK e-commerce**, ABS gives **Australian context** only.
- Avoid over-reacting to high lift on **very low support**.
- Use these as **decision aids**, not automated policy.
"""
        )


def insights_docs():
    with st.expander("How insights are generated (details)", expanded=False):
        st.markdown(
            """
### Data sources
- **Rules** – mined from UCI Online Retail II (UK) after cleaning:
  UK-only, positive quantity/price, no cancellations; basketised by invoice.
- **Metrics** – support (share of baskets), confidence (P(B|A)), lift (> 1 desirable).
- **Context** – ABS 8501.0 monthly retail turnover, aggregated and normalised to $M.

### Rule selection
- For *bread-and-butter* cards: we pick rules with **highest support**.
- For *lapsing customers*: we filter to the corresponding **customer segment** and
  then pick rules with **highest support** there.
- For technical bundle insight: we use **highest lift** within filters.

### Audience focus
- **Store-ready view** – minimal jargon, shelf and promo actions.
- **Analytics view** – the same patterns but with explicit metrics.

### Guardrails
- Extreme lift with tiny support is flagged as lower confidence.
- ABS is used to check directionality (e.g. December peaks in household goods).
"""
        )


# ---------- ABS helpers ----------

def _detect_abs_cols(csv_path: Path):
    hdr = pd.read_csv(csv_path, nrows=0)
    cols_norm = {c.strip().lower(): c for c in hdr.columns}

    def pick(*cands):
        for c in cands:
            if c in cols_norm:
                return cols_norm[c]
        return None

    region = pick("region", "state", "state/territory", "geography", "geog")
    date = pick("time_period", "time period", "period", "month", "date")
    value = pick("obs_value", "observation value", "value", "turnover", "amount")
    industry = pick("industry", "industry_code", "category", "group")
    return region, date, value, industry


@st.cache_data(show_spinner=False)
def load_abs(path: Path, keep_years: int = 5):
    if not path.exists():
        return pd.DataFrame(), {}

    region_col, date_col, value_col, industry_col = _detect_abs_cols(path)
    need = [c for c in [region_col, date_col, value_col, industry_col] if c]
    if not all([region_col, date_col, value_col]):
        return pd.DataFrame(), {
            "region": region_col,
            "date": date_col,
            "value": value_col,
            "industry": industry_col,
        }

    df = pd.read_csv(path, usecols=need, low_memory=False)
    ren = {region_col: "region", date_col: "date", value_col: "turnover"}
    if industry_col:
        ren[industry_col] = "industry"
    df = df.rename(columns=ren)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    df = df.dropna(subset=["date", "turnover"])

    cut = df["date"].max() - pd.DateOffset(years=keep_years)
    df = df[df["date"] >= cut]

    keys = ["date"]
    if "region" in df.columns:
        keys.append("region")
    if "industry" in df.columns:
        keys.append("industry")
    df = df.groupby(keys, as_index=False)["turnover"].sum()
    return df, {
        "region": region_col,
        "date": date_col,
        "value": value_col,
        "industry": industry_col,
    }


def _latest_yoy(abs_df: pd.DataFrame) -> float:
    if abs_df.empty:
        return 0.0
    m = abs_df.assign(M=abs_df["date"].dt.to_period("M")).groupby("M", as_index=False)["turnover"].sum()
    m["date"] = m["M"].dt.to_timestamp()
    if len(m) < 13:
        return 0.0
    latest = m.iloc[-1]["turnover"]
    last_year = m.iloc[-13]["turnover"]
    return float((latest - last_year) / last_year * 100.0) if last_year else 0.0


def _top_industry(abs_df: pd.DataFrame) -> str:
    if "industry" not in abs_df.columns or abs_df.empty:
        return "Total"
    last12 = abs_df[abs_df["date"] >= abs_df["date"].max() - pd.DateOffset(months=12)]
    g = last12.groupby("industry", as_index=False)["turnover"].sum().sort_values("turnover", ascending=False)
    return str(g.iloc[0]["industry"]) if not g.empty else "Total"


def _december_uplift(abs_df: pd.DataFrame) -> float:
    if abs_df.empty:
        return 0.0
    tmp = abs_df.copy()
    tmp["month"] = tmp["date"].dt.month
    last24 = tmp[tmp["date"] >= tmp["date"].max() - pd.DateOffset(months=24)]
    monthly = last24.groupby("month", as_index=False)["turnover"].mean()
    if monthly.empty or "turnover" not in monthly:
        return 0.0
    mean_all = monthly["turnover"].mean()
    dec = monthly.loc[monthly["month"] == 12, "turnover"]
    return (float(dec.iloc[0]) / float(mean_all) - 1) * 100.0 if not dec.empty and mean_all > 0 else 0.0


# ---------- Rules helpers ----------

@st.cache_data(show_spinner=False)
def load_rules(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    keep = [
        c
        for c in [
            "antecedent",
            "consequent",
            "support",
            "confidence",
            "lift",
            "industry",
            "segment",
        ]
        if c in df.columns
    ]
    if not keep:
        return pd.DataFrame()

    df = df[keep].dropna()

    for c in ("support", "confidence", "lift"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("antecedent", "consequent"):
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(r"[\{\}\[\]\(\)\"']", "", regex=True)
                # drop any "123 | " product IDs
                .str.replace(r"\s*\d+\s*\|\s*", "", regex=True)
                .str.strip()
            )

    df = df.dropna(subset=["support", "confidence", "lift"])
    return df


def _rules_summary(df_rules: pd.DataFrame):
    if df_rules.empty:
        return 0, 0.0
    return len(df_rules), float(df_rules["lift"].mean())


def pick_rules(df: pd.DataFrame, segment_val, industry_val):
    """Return 3 rules: top_lift, top_support, top_support_lapsing."""
    work = df.copy()

    # filters
    if industry_val is not None and "industry" in work.columns:
        work = work[work["industry"] == industry_val]

    if segment_val is not None and "segment" in work.columns:
        work = work[work["segment"] == segment_val]

    if work.empty:
        return None, None, None

    # 1) strongest bundle (lift)
    r_lift = work.sort_values("lift", ascending=False).head(1)

    # 2) bread-and-butter (support)
    r_sup = work.sort_values("support", ascending=False).head(1)

    # 3) lapsing customers (segment that represents "slipping")
    r_slip = None
    if "segment" in df.columns:
        # try typical code 2 or 3 for lapsing; fall back to max segment id
        seg_vals = sorted(df["segment"].dropna().unique())
        if len(seg_vals):
            # heuristic: last value = riskiest / lapsing
            lapsing_code = seg_vals[-1]
            slip = work[work["segment"] == lapsing_code]
            if not slip.empty:
                r_slip = slip.sort_values("support", ascending=False).head(1)

    return r_lift, r_sup, r_slip


# ---------- UI helpers ----------

def _hero(title, tagline):
    st.markdown(
        f"""
<div style="
    background: linear-gradient(90deg, #1e3a8a, #0f766e);
    border-radius:20px; padding:18px 22px; margin:10px 0 18px 0;
    color:white; display:flex; align-items:center; gap:14px;">
    <div style="font-size:18px; font-weight:600;">{title}</div>
    <div style="opacity:.85;">{tagline}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _insight_card(title, what, why, action, evidence, conf, icon="✅"):
    st.markdown(
        f"""
<div style="
  border:1px solid #e5e7eb;
  border-radius:14px;
  padding:14px 16px;
  margin-top:10px;
  background:#fcfdff;">
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="font-size:18px">{icon}</div>
    <div style="font-weight:700; font-size:15.5px;">{title}</div>
  </div>
  <div style="margin-top:8px; color:#0f172a;">{what}</div>
  <div style="color:#334155; margin-top:4px;"><b>Why:</b> {why}</div>
  <div style="margin-top:6px;"><b>Action:</b> {action}</div>
  <div style="color:#64748b; margin-top:4px; font-size:13px;"><i>{evidence}</i></div>
  <div style="margin-top:6px; font-size:12px; color:#475569;">
    Confidence: <b>{conf}</b>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------- Main entry ----------

def show():
    st.title("Insights")
    insights_helpbar()

    abs_df, _det = load_abs(ABS_FILE)
    rules_df = load_rules(RULES_FILE)

    n_rules, avg_lift = _rules_summary(rules_df)
    yoy = _latest_yoy(abs_df)
    top_cat = _top_industry(abs_df)
    dec_uplift = _december_uplift(abs_df)

    _hero("Insights", "Mined rules → clear actions, grounded in ABS retail trends.")

    # header row: logo + title + audience toggle
    col_logo, col_hdr, col_mode = st.columns([1, 3, 2], vertical_alignment="center")
    with col_logo:
        if LOGO_FILE.exists():
            st.image(str(LOGO_FILE), width=56)
    with col_hdr:
        st.subheader("Insight summary ↪")
        st.caption("Filters tailor the rules; audience focus controls how technical the cards are.")
    with col_mode:
        audience_mode = st.radio(
            "Audience focus",
            ["Store-ready view", "Analytics view"],
            horizontal=True,
            label_visibility="visible",
        )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rules shown (all)", f"{n_rules:,}")
    c2.metric("Average lift", f"{avg_lift:.2f}" if n_rules else "—")
    c3.metric("Top category (ABS)", top_cat)
    c4.metric("YoY turnover (ABS)", f"{yoy:+.1f}%" if abs_df.size else "—")

    # focus filters (segment + industry)
    st.markdown("### Focus filters")

    f1, f2 = st.columns(2)

    # segment filter
    segment_val = None
    segment_label = "All customers"
    if "segment" in rules_df.columns:
        seg_unique = sorted(rules_df["segment"].dropna().unique())
        seg_labels = ["All customers"]
        seg_map = {"All customers": None}
        # heuristic, map ordered codes to human labels
        human_names = [
            "New shoppers",
            "Repeat customers",
            "High-value regulars",
            "At-risk / lapsing customers",
        ]
        for i, val in enumerate(seg_unique):
            label = human_names[i] if i < len(human_names) else f"Segment {val}"
            seg_labels.append(label)
            seg_map[label] = val

        with f1:
            chosen_label = st.selectbox("Focus segment", seg_labels, index=0)
        segment_val = seg_map[chosen_label]
        segment_label = chosen_label
    else:
        with f1:
            st.selectbox("Focus segment", ["All customers"], index=0, disabled=True)

    # industry filter
    industry_val = None
    industry_label = "All industries"
    if "industry" in rules_df.columns and not rules_df.empty:
        inds = sorted(rules_df["industry"].dropna().astype(str).unique())
        inds_opts = ["All industries"] + inds
        with f2:
            chosen_ind = st.selectbox("Focus industry", inds_opts, index=0)
        if chosen_ind != "All industries":
            industry_val = chosen_ind
            industry_label = chosen_ind
    else:
        with f2:
            st.selectbox("Focus industry", ["All industries"], index=0, disabled=True)

    # filtered rule picks
    r_lift, r_sup, r_slip = pick_rules(rules_df, segment_val, industry_val)

    # quick ABS sparkline
    if not abs_df.empty:
        m = abs_df.assign(M=abs_df["date"].dt.to_period("M")).groupby("M", as_index=False)["turnover"].sum()
        m["date"] = m["M"].dt.to_timestamp()
        fig = px.line(
            m.tail(24),
            x="date",
            y="turnover",
            title="National retail turnover — recent trend",
            markers=True,
            color_discrete_sequence=[PALETTE["blue"]],
        )
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=10), height=240)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Focused insights")

    # ---------- Store-ready view (manager style) ----------
    if audience_mode == "Store-ready view":
        # bread-and-butter combo (top support)
        if r_sup is not None:
            rs = r_sup.iloc[0]
            a = str(rs["antecedent"])
            c = str(rs["consequent"])
            _insight_card(
                title="Protect your bread-and-butter combo",
                what=f"Customers most often buy **{a}** together with **{c}** "
                     f"in this focus ({segment_label}, {industry_label}).",
                why="This pair is a stable driver of everyday revenue.",
                action="Place them together and keep both fully stocked.",
                evidence=f"Support={rs['support']:.3f} (share of baskets).",
                conf="High" if rs["support"] > 0.01 else "Medium",
                icon="🍞",
            )

        # lapsing customers (top support within lapsing segment)
        if r_slip is not None:
            rsl = r_slip.iloc[0]
            a2 = str(rsl["antecedent"])
            c2 = str(rsl["consequent"])
            _insight_card(
                title="Win back slipping customers",
                what=f"At-risk customers most often pick **{a2}** with **{c2}**.",
                why="Good candidate combo for win-back offers.",
                action="Use this combo in reactivation emails or SMS.",
                evidence=f"Support={rsl['support']:.3f} within lapsing segment.",
                conf="Medium",
                icon="🔄",
            )

        # seasonal push using ABS
        _insight_card(
            title="Prepare for the December lift",
            what=f"Household-related spend rises about {dec_uplift:+.1f}% in December vs average.",
            why="Shoppers move early on gifts and home items.",
            action="Order extra stock by mid-November in key categories.",
            evidence="ABS 8501.0 monthly turnover; simple 24-month average comparison.",
            conf="Medium-High",
            icon="🎄",
        )

    # ---------- Analytics view (analyst style) ----------
    else:
        # high-lift bundle
        if r_lift is not None:
            rl = r_lift.iloc[0]
            _insight_card(
                title=f"Strongest bundle in focus: {rl['antecedent']} → {rl['consequent']}",
                what=f"Lift={rl['lift']:.2f}, Support={rl['support']:.3f}, "
                     f"Confidence={rl['confidence']:.2f}.",
                why="Lift > 1 implies the consequent is meaningfully enriched "
                    "when the antecedent is present.",
                action="Run an attach-rate experiment around this pair.",
                evidence=f"Top-lift rule under segment={segment_label}, industry={industry_label}.",
                conf="High" if rl["lift"] >= 1.5 and rl["support"] > 0.003 else "Medium",
                icon="📊",
            )

        # bread-and-butter (top support)
        if r_sup is not None:
            rs = r_sup.iloc[0]
            _insight_card(
                title="Bread-and-butter driver (top support)",
                what=f"Highest-support rule in focus: {rs['antecedent']} → {rs['consequent']} "
                     f"with Support={rs['support']:.3f}.",
                why="High support rules anchor demand and improve forecast stability.",
                action="Model price and promo elasticity for this pair.",
                evidence=f"Support={rs['support']:.3f}, Lift={rs['lift']:.2f}.",
                conf="High" if rs["support"] > 0.01 else "Medium",
                icon="📈",
            )

        # lapsing segment rule (if available)
        if r_slip is not None:
            rsl = r_slip.iloc[0]
            _insight_card(
                title="Lapsing-customer signal",
                what=f"In the lapsing segment, {rsl['antecedent']} → {rsl['consequent']} "
                     f"is the top-support rule.",
                why="This pattern can inform personalised reactivation strategies.",
                action="Build uplift models using this pair as a candidate feature.",
                evidence=f"Lapsing-segment Support={rsl['support']:.3f}, Lift={rsl['lift']:.2f}.",
                conf="Medium",
                icon="🔍",
            )

        _insight_card(
            title="Seasonality anchor from ABS",
            what=f"December turnover is roughly {dec_uplift:+.1f}% above the 24-month mean.",
            why="Useful baseline when interpreting year-end spikes in rule activity.",
            action="Align experiment windows with known seasonal peaks.",
            evidence="ABS 8501.0; turnover aggregated to national level, 24-month window.",
            conf="Medium-High",
            icon="📐",
        )

    st.caption(
        "This tab uses filtered rule picks (segment + industry) to surface a bread-and-butter "
        "combo, a lapsing-customer combo, and—optionally—a strongest-lift bundle, then "
        "frames them either for store-ready decisions or analytics deep-dives."
    )

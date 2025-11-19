import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import re

ABS_FILE = Path("data/ABS.csv")
RULES_FILE = Path("data/all_segment_rules_with_industry.csv")
LOGO_FILE = Path("assets/logo.png")

PALETTE = {
    "blue": "#264E86",
    "green": "#2CA58D",
    "orange": "#F4A300",
    "ink": "#0F172A",
}


# ---------- small helpers / layout ----------

def insights_helpbar():
    """Top helper strip – three wide popover buttons."""
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.popover("🧭 What this tab does", use_container_width=True):
            st.markdown(
                """
- Turns mined UK basket rules into **store-ready recommendations**.
- Adds **ABS turnover** so patterns aren’t read in isolation.
- Lets you switch between **Store-ready** and **Analytics** views.
"""
            )

    with col2:
        with st.popover("📁 How cards are filled", use_container_width=True):
            st.markdown(
                """
Each card follows the same structure:

- **What we see** – one-line summary of the pattern.  
- **Why it matters** – commercial / operational impact.  
- **Action** – short imperative step.  
- **Evidence** – support · confidence · lift (+ ABS stat if used).  
- **Confidence** – High / Medium, based on lift and stability.
"""
            )

    with col3:
        with st.popover("🛡️ Limits & guardrails", use_container_width=True):
            st.markdown(
                """
- Rules are from **UCI Online Retail II (UK)**, not local tills.  
- ABS is used only as **macro context**, not to prove causality.  
- Avoid acting on **tiny-support spikes** even with high lift.  
- Use this tab to **inform** decisions, not to auto-allocate budgets.
"""
            )


def _hero(title, tagline, mode_label):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #1e3a8a, #0f766e);
            border-radius:20px; padding:16px 20px; margin:8px 0 16px 0;
            color:white; display:flex; align-items:center; justify-content:space-between;">
            <div>
              <div style="font-size:18px; font-weight:600; margin-bottom:4px;">{title}</div>
              <div style="opacity:.9; font-size:13px;">{tagline}</div>
            </div>
            <div style="
                padding:6px 12px; border-radius:999px;
                background:rgba(15,23,42,0.3); font-size:12px;">
                Current view: <b>{mode_label}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _insight_card(title, what, why, action, evidence, confidence, icon="🧩"):
    st.markdown(
        f"""
        <div style="
          border:1px solid #e5e7eb; border-radius:16px;
          padding:16px 18px; margin-top:12px; background:#ffffff;">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="font-size:18px;">{icon}</div>
            <div style="font-weight:700; font-size:16px; color:#0f172a;">{title}</div>
          </div>
          <div style="margin-top:10px; color:#111827; font-size:14px;">
            <b>What we see:</b> {what}
          </div>
          <div style="margin-top:4px; color:#374151; font-size:14px;">
            <b>Why it matters:</b> {why}
          </div>
          <div style="margin-top:6px; color:#111827; font-size:14px;">
            <b>Action:</b> {action}
          </div>
          <div style="margin-top:6px; color:#6b7280; font-size:13px;">
            <i>{evidence}</i>
          </div>
          <div style="margin-top:6px; color:#4b5563; font-size:12px;">
            Confidence: <b>{confidence}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- data cleaners ----------

def clean_product_name(x: str) -> str:
    if not isinstance(x, str):
        return x

    # strip IDs like "85123A | " at the front
    x = re.sub(r"^\s*[A-Za-z0-9]+\s*\|\s*", "", x)

    # strip leading numeric codes like "22423 "
    x = re.sub(r"^\s*\d+\s+", "", x)

    # kill stray markdown asterisks from original strings
    x = x.replace("*", "")

    # collapse whitespace
    x = re.sub(r"\s{2,}", " ", x).strip()
    return x


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
        return pd.DataFrame(), {"region": region_col, "date": date_col,
                                "value": value_col, "industry": industry_col}

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
    return df, {"region": region_col, "date": date_col,
                "value": value_col, "industry": industry_col}


def _detect_segment_col(df: pd.DataFrame):
    for c in ["segment", "customer_segment", "segment_label"]:
        if c in df.columns:
            return c
    return None


@st.cache_data(show_spinner=False)
def load_rules(path: Path):
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    keep = [c for c in ["antecedent", "consequent", "support",
                        "confidence", "lift", "industry", "segment"]
            if c in df.columns]
    if not keep:
        return pd.DataFrame()

    df = df[keep].dropna()
    for c in ["support", "confidence", "lift"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["antecedent", "consequent"]:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                     .str.replace(r"[\{\}\[\]\(\)\"']", "", regex=True)
                     .apply(clean_product_name))

    df = df.dropna(subset=["support", "confidence", "lift"])
    return df


# ---------- ABS helpers ----------

def _latest_yoy(abs_df: pd.DataFrame) -> float:
    if abs_df.empty:
        return 0.0
    m = abs_df.assign(M=abs_df["date"].dt.to_period("M")) \
              .groupby("M", as_index=False)["turnover"].sum()
    m["date"] = m["M"].dt.to_timestamp()
    if len(m) < 13:
        return 0.0
    latest = m.iloc[-1]["turnover"]
    prev = m.iloc[-13]["turnover"]
    if prev == 0:
        return 0.0
    return float((latest - prev) / prev * 100.0)


def _top_industry(abs_df: pd.DataFrame) -> str:
    if abs_df.empty or "industry" not in abs_df.columns:
        return "Total"
    last12 = abs_df[abs_df["date"] >= abs_df["date"].max() - pd.DateOffset(months=12)]
    t = last12.groupby("industry", as_index=False)["turnover"].sum() \
              .sort_values("turnover", ascending=False)
    return str(t.iloc[0]["industry"]) if not t.empty else "Total"


def _december_uplift(abs_df: pd.DataFrame) -> float:
    if abs_df.empty:
        return 0.0
    tmp = abs_df.copy()
    tmp["month"] = tmp["date"].dt.month
    last24 = tmp[tmp["date"] >= tmp["date"].max() - pd.DateOffset(months=24)]
    monthly = last24.groupby("month", as_index=False)["turnover"].mean()
    if monthly.empty:
        return 0.0
    mean_all = float(monthly["turnover"].mean())
    dec = monthly.loc[monthly["month"] == 12, "turnover"]
    if dec.empty or mean_all == 0:
        return 0.0
    return float(dec.iloc[0] / mean_all - 1) * 100.0


# ---------- rule selection helpers ----------

def _pick_top_rule(df: pd.DataFrame):
    if df.empty:
        return None
    return df.sort_values(
        ["support", "confidence", "lift"],
        ascending=[False, False, False]
    ).iloc[0]


def _pick_lapsing_rule(df: pd.DataFrame, seg_col: str | None):
    if seg_col is None or seg_col not in df.columns or df.empty:
        return None, None
    seg_vals = sorted(df[seg_col].dropna().unique())
    if not seg_vals:
        return None, None
    lapsing_value = seg_vals[-1]
    sub = df[df[seg_col] == lapsing_value]
    if sub.empty:
        return None, lapsing_value
    return _pick_top_rule(sub), lapsing_value


# ---------- main tab ----------

def show():
    st.title("Insights")
    insights_helpbar()

    abs_df, _ = load_abs(ABS_FILE)
    rules_df = load_rules(RULES_FILE)

    if rules_df.empty:
        st.error(f"No rules available at {RULES_FILE}.")
        return

    seg_col = _detect_segment_col(rules_df)

    # ----- focus filters -----
    left_filters, right_mode = st.columns([2, 1])

    with left_filters:
        # segment display mapping
        seg_label_map = {}
        seg_focus_label = "All customers"
        seg_focus_value = None
        if seg_col:
            seg_vals = sorted(rules_df[seg_col].dropna().unique())
            if pd.api.types.is_numeric_dtype(rules_df[seg_col]):
                default_labels = [
                    "New shoppers",
                    "Repeat shoppers",
                    "High-value shoppers",
                    "Lapsing customers",
                ]
                for i, v in enumerate(seg_vals):
                    seg_label_map[v] = default_labels[i] if i < len(default_labels) else f"Segment {v}"
            else:
                seg_label_map = {v: str(v) for v in seg_vals}

            options = ["All customers"] + [seg_label_map[v] for v in seg_vals]
            seg_focus_label = st.selectbox("Focus segment", options, index=0)
            if seg_focus_label != "All customers":
                # inverse lookup
                for raw, lab in seg_label_map.items():
                    if lab == seg_focus_label:
                        seg_focus_value = raw
                        break

        ind_focus_label = "All industries"
        ind_focus_value = None
        if "industry" in rules_df.columns:
            inds = sorted(rules_df["industry"].astype(str).unique())
            ind_focus_label = st.selectbox(
                "Focus industry",
                ["All industries"] + inds,
                index=0,
            )
            if ind_focus_label != "All industries":
                ind_focus_value = ind_focus_label

    with right_mode:
        mode = st.radio(
            "View mode",
            ["Store-ready view", "Analytics view"],
            horizontal=True,
            index=0,
        )

    # apply filters
    focus = rules_df.copy()
    if seg_focus_value is not None and seg_col:
        focus = focus[focus[seg_col] == seg_focus_value]
    if ind_focus_value is not None and "industry" in focus.columns:
        focus = focus[focus["industry"].astype(str) == ind_focus_value]

    if focus.empty:
        st.warning("No rules for this combination of segment and industry.")
        return

    # KPIs
    n_rules = int(len(focus))
    avg_lift = float(focus["lift"].mean())
    yoy = _latest_yoy(abs_df)
    top_cat = _top_industry(abs_df)

    _hero(
        "Insights",
        "Use mined rules plus ABS context to decide what to pair and where to focus.",
        mode_label=mode,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rules in focus", f"{n_rules:,}")
    k2.metric("Average lift", f"{avg_lift:.2f}")
    k3.metric("Top ABS category", top_cat)
    k4.metric("YoY turnover (ABS)", f"{yoy:+.1f}%" if abs_df.size else "—")

    # ABS sparkline
    if not abs_df.empty:
        m = abs_df.assign(M=abs_df["date"].dt.to_period("M")) \
                  .groupby("M", as_index=False)["turnover"].sum()
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

    dec_uplift = _december_uplift(abs_df)

    # pick rules
    bread_rule = _pick_top_rule(focus)
    lapsing_rule, lapsing_value = _pick_lapsing_rule(focus, seg_col)

    # text labels for segment / industry context
    seg_context = seg_focus_label
    ind_context = ind_focus_label

    # -------- store vs analytics text --------

    if mode == "Store-ready view":
        # bread-and-butter combo
        if bread_rule is not None:
            a = bread_rule["antecedent"]
            c = bread_rule["consequent"]
            sup = float(bread_rule["support"])
            _insight_card(
                title="Protect your bread-and-butter combo",
                what=(
                    f"Customers most often buy <b>{a}</b> together with <b>{c}</b> "
                    f"in this focus ({seg_context}, {ind_context})."
                ),
                why="This pair is a stable driver of everyday revenue.",
                action="Place them together and keep both fully stocked.",
                evidence=f"Support≈{sup:.3f} (share of baskets).",
                confidence="High" if sup >= 0.02 else "Medium",
                icon="🧱",
            )

        # lapsing customers
        if lapsing_rule is not None:
            a = lapsing_rule["antecedent"]
            c = lapsing_rule["consequent"]
            sup = float(lapsing_rule["support"])
            seg_label = (
                seg_label_map.get(lapsing_value, f"Segment {lapsing_value}")
                if seg_col and lapsing_value is not None
                else "Lapsing customers"
            )
            _insight_card(
                title="Win back slipping customers",
                what=(
                    f"{seg_label} who still shop most often buy <b>{a}</b> "
                    f"with <b>{c}</b>."
                ),
                why="Highlighting this pair may recover at-risk shoppers.",
                action="Feature this combo in outbound offers or at aisle ends.",
                evidence=f"Support≈{sup:.3f} within lapsing segment.",
                confidence="Medium",
                icon="🩹",
            )

    else:  # Analytics view
        if bread_rule is not None:
            a = bread_rule["antecedent"]
            c = bread_rule["consequent"]
            sup = float(bread_rule["support"])
            conf = float(bread_rule["confidence"])
            lift = float(bread_rule["lift"])
            _insight_card(
                title=f"Bread-and-butter combo: {a} → {c}",
                what=(
                    f"Top rule in focus: support={sup:.3f}, "
                    f"confidence={conf:.2f}, lift={lift:.2f}."
                ),
                why="High lift at non-trivial support marks a dependable attach pattern.",
                action="Use this pair as a benchmark basket for promo tests.",
                evidence=(
                    f"Rule from filtered UK baskets; focus={seg_context}, {ind_context}. "
                    f"ABS YoY turnover={yoy:+.1f}%, December uplift≈{dec_uplift:+.1f}%."
                ),
                confidence="High" if (lift >= 2.0 and sup >= 0.01) else "Medium",
                icon="📊",
            )

        if lapsing_rule is not None:
            a = lapsing_rule["antecedent"]
            c = lapsing_rule["consequent"]
            sup = float(lapsing_rule["support"])
            conf = float(lapsing_rule["confidence"])
            lift = float(lapsing_rule["lift"])
            seg_label = (
                seg_label_map.get(lapsing_value, f"Segment {lapsing_value}")
                if seg_col and lapsing_value is not None
                else "Lapsing customers"
            )
            _insight_card(
                title=f"Lapsing segment combo: {a} → {c}",
                what=(
                    f"Within {seg_label}, the strongest rule has "
                    f"support={sup:.3f}, confidence={conf:.2f}, lift={lift:.2f}."
                ),
                why="This shows what remaining spend from at-risk shoppers looks like.",
                action="Use as seed for targeted retention / CRM experiments.",
                evidence=f"Segment column={seg_col or 'n/a'}; lapsing code={lapsing_value}.",
                confidence="Medium",
                icon="🧪",
            )

    st.caption(
        "This tab converts UK association rules into store-ready moves and analytical summaries, "
        "with ABS providing a macro-level reality check."
    )

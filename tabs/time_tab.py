import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---- Helper UI utilities for Time Dynamics ----
PALETTE = {"blue": "#264E86", "green": "#2CA58D", "orange": "#F4A300"}

def time_dynamics_helpbar():
    """Compact helper bubbles for quick guidance."""
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1.popover("📝 What is this?", use_container_width=True):
        st.markdown(
            "- **Goal:** show how retail activity changes over time.\n"
            "- **Data:** ABS 8501.0 monthly turnover (last 3–5 years).\n"
            "- **Why it matters:** timing → stocking, staffing, and promo windows."
        )
    with c2.popover("👁️‍🗨️ Reading the charts", use_container_width=True):
        st.markdown(
            "• **Line:** monthly turnover.\n"
            "• **3-month MA:** smoother trend overlay.\n"
            "• **YoY marker:** compares same month last year to remove seasonality."
        )
    with c3.popover("🌀 Decomposition", use_container_width=True):
        st.markdown(
            "- **Trend** (long-run level)\n"
            "- **Seasonality** (repeating monthly pattern)\n"
            "- **Residual** (one-offs / noise)\n"
            "Use it to separate real growth from Christmas spikes."
        )
    with c4.popover("⚠️     Caveats", use_container_width=True):
        st.markdown(
            "- ABS is **macro**; it won’t match basket counts exactly.\n"
            "- Check **UNIT_MULT** scaling; this app shows **$ Millions** after normalization.\n"
            "- Missing months are forward-filled for plotting only."
        )

def time_dynamics_docs():
    """Long-form helper—put at the bottom of the tab."""
    with st.expander("How we compute these time views (click to expand)", expanded=False):
        st.markdown(
            f"""
**Pipeline in brief**

1) **Load & normalize** ABS 8501.0 CSV (header-agnostic detection).  
2) Convert value column to numeric, apply **10^UNIT_MULT**, standardize to **$M**.  
3) **Aggregate** duplicates by **region × date (× industry)**.  
4) Optional overlays:  
   - **3-month moving average** (per region)  
   - **YoY %** = (value − value_12m_ago)/value_12m_ago  
5) If available, run **STL decomposition** (statsmodels) to separate **trend/seasonality/residual**.

**How to use**

- Use the **Region** selector to compare areas or focus on Australia.  
- Toggle **3-month MA** to view underlying trend.  
- Use **YoY** to judge whether a spike is seasonal or structural.  

**Interpretation tips**

- A rising **trend** + positive **YoY** = sustained growth.  
- Seasonal **peaks** (Nov–Dec) suggest earlier ordering & roster ramp-up.  
- Large **residuals** hint at events (promo, supply shock); annotate for future.
"""
        )

# optional: STL (graceful fallback if not installed)
try:
    from statsmodels.tsa.seasonal import STL
except Exception:
    STL = None

DATA_DIR = Path("data")
MONTHLY_FILE = DATA_DIR / "retail_monthly.parquet"
ABS_FILE = DATA_DIR / "ABS.csv"
RULES_FILE = DATA_DIR / "all_segment_rules_with_industry.csv"

def _detect_cols(csv_path: Path):
    hdr = pd.read_csv(csv_path, nrows=0)
    cols_norm = {c.strip().lower(): c for c in hdr.columns}
    def get(*cands):
        for c in cands:
            if c in cols_norm:
                return cols_norm[c]
        return None
    region   = get("region", "state", "state/territory", "geography", "geog")
    date     = get("time_period", "time period", "period", "month", "date")
    value    = get("obs_value", "observation value", "value", "turnover", "amount")
    industry = get("industry", "industry_code", "category", "group")
    return region, date, value, industry

@st.cache_data(show_spinner=False)
def _load_abs(csv_path: Path, keep_years: int = 7):
    region_col, date_col, value_col, industry_col = _detect_cols(csv_path)
    need = [c for c in [region_col, date_col, value_col, industry_col, "UNIT_MULT"] if c]
    if not all([region_col, date_col, value_col]):
        return pd.DataFrame(), {"region":region_col, "date":date_col, "value":value_col, "industry":industry_col}

    df = pd.read_csv(csv_path, usecols=need, low_memory=False)
    ren = {region_col:"region", date_col:"date", value_col:"turnover"}
    if industry_col: ren[industry_col] = "industry"
    df = df.rename(columns=ren)
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    if "UNIT_MULT" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            df["UNIT_MULT"] = pd.to_numeric(df["UNIT_MULT"], errors="coerce").fillna(0)
            df["turnover"] = df["turnover"] * (10 ** df["UNIT_MULT"])
        df = df.drop(columns=["UNIT_MULT"])
    if "region" not in df.columns:
        df["region"] = "Australia"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "turnover"])
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    df["date"] = df["month"]
    df = df.drop(columns=["month"])
    cut = df["date"].max() - pd.DateOffset(years=keep_years)
    df = df[df["date"] >= cut]
    keys = ["region", "date"] + (["industry"] if "industry" in df.columns else [])
    df = df.groupby(keys, as_index=False)["turnover"].sum()
    if df["turnover"].max() > 1e6:
        df["turnover"] = df["turnover"] / 1e6
    return df, {"region":region_col, "date":date_col, "value":value_col, "industry":industry_col}

@st.cache_data(show_spinner=False)
def _load_monthly(path: Path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")

@st.cache_data(show_spinner=False)
def _load_rules_industry(path: Path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    if "industry" not in df.columns:
        return pd.DataFrame()
    s = (df.groupby("industry", as_index=False)
            .size()
            .rename(columns={"size":"rule_count"}))
    s["rule_share"] = s["rule_count"] / s["rule_count"].sum()
    return s.sort_values("rule_share", ascending=False)

def _year_filter(df, date_col="date"):
    y_min = int(df[date_col].dt.year.min())
    y_max = int(df[date_col].dt.year.max())
    y0, y1 = st.slider("Year range", y_min, y_max, (max(y_min, y_max-3), y_max))
    return df[(df[date_col].dt.year >= y0) & (df[date_col].dt.year <= y1)], (y0, y1)

def _stl_section(df_reg: pd.DataFrame, region_label: str):
    if STL is None:
        st.info("Install `statsmodels` to enable STL decomposition (pip install statsmodels).")
        return
    s = (df_reg.set_index("date")["turnover"]
            .asfreq("MS")
            .interpolate("linear"))
    if len(s) < 24:
        st.info("Not enough points for decomposition (need ≥ 24 months).")
        return
    res = STL(s, period=12, robust=True).fit()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Observed","Trend","Seasonal","Residual"),
                        vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Observed"), 1, 1)
    fig.add_trace(go.Scatter(x=s.index, y=res.trend, name="Trend"), 2, 1)
    fig.add_trace(go.Scatter(x=s.index, y=res.seasonal, name="Seasonal"), 3, 1)
    fig.add_trace(go.Scatter(x=s.index, y=res.resid, name="Residual"), 4, 1)
    fig.update_layout(height=900, title=f"STL Decomposition — {region_label}", showlegend=False, margin=dict(l=0,r=0,t=60,b=0))
    st.plotly_chart(fig, use_container_width=True)

def show():
    st.title("Time Dynamics")
    time_dynamics_helpbar()

    m = _load_monthly(MONTHLY_FILE)
    if not m.empty:
        st.caption("Using Team-A monthly rollup.")
        m, year_span = _year_filter(m, "date")
        c1, c2 = st.columns([3, 2])
        with c1:
            fig1 = px.line(
                m, x="date", y="txn_count",
                labels={"date":"Month", "txn_count":"Transactions"},
                title=f"Monthly Transaction Frequency ({year_span[0]}–{year_span[1]})"
            )
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            mm = m.copy()
            mm["weekday"] = mm["date"].dt.weekday
            mm["bucket"] = np.where(mm["weekday"] < 5, "Weekday", "Weekend")
            wk = mm.groupby("bucket", as_index=False)["total_sales"].sum()
            fig2 = px.bar(
                wk, x="bucket", y="total_sales",
                labels={"bucket":"", "total_sales":"Sales"},
                title="Weekday vs Weekend Sales"
            )
            st.plotly_chart(fig2, use_container_width=True)
        with st.expander("Data preview"):
            st.dataframe(m.tail(12), use_container_width=True)
        st.subheader("Decomposition (Optional)")
        if st.checkbox("Decompose total sales (requires statsmodels)", value=False):
            if STL is None:
                st.info("Install `statsmodels` to enable decomposition.")
            else:
                s = m.set_index("date")["total_sales"].asfreq("MS").interpolate("linear")
                if len(s) >= 24:
                    res = STL(s, period=12, robust=True).fit()
                    figd = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                        subplot_titles=("Observed","Trend","Seasonal","Residual"),
                                        vertical_spacing=0.07)
                    figd.add_trace(go.Scatter(x=s.index, y=s.values), 1, 1)
                    figd.add_trace(go.Scatter(x=s.index, y=res.trend), 2, 1)
                    figd.add_trace(go.Scatter(x=s.index, y=res.seasonal), 3, 1)
                    figd.add_trace(go.Scatter(x=s.index, y=res.resid), 4, 1)
                    figd.update_layout(height=900, title="STL Decomposition — Total Sales", showlegend=False, margin=dict(l=0,r=0,t=60,b=0))
                    st.plotly_chart(figd, use_container_width=True)
                else:
                    st.info("Need ≥ 24 monthly points for STL.")
        time_dynamics_docs()
        return

    if not ABS_FILE.exists():
        st.error(f"ABS file not found: {ABS_FILE}")
        st.caption("Provide either data/retail_monthly.parquet or data/ABS.csv")
        time_dynamics_docs()
        return

    df_abs, detected = _load_abs(ABS_FILE)
    if df_abs.empty:
        st.error("ABS file parsed but produced no usable rows.")
        with st.expander("Detected columns"):
            st.write(detected)
        time_dynamics_docs()
        return

    df_abs, year_span = _year_filter(df_abs, "date")
    regs = df_abs["region"].astype(str).unique().tolist()
    if {"New South Wales", "Victoria"}.issubset(regs):
        sel = ["New South Wales", "Victoria"]
        colors = {"New South Wales":"#264E86", "Victoria":"#2CA58D"}
        title_suffix = "NSW vs VIC"
    elif "Australia" in regs:
        sel = ["Australia"]
        colors = {"Australia":"#264E86"}
        title_suffix = "Australia"
    else:
        sel = (df_abs.groupby("region")["turnover"].sum()
                    .sort_values(ascending=False).head(2).index.tolist())
        colors = {sel[i]: ("#264E86" if i==0 else "#2CA58D") for i in range(len(sel))}
        title_suffix = " — ".join(sel)

    d = df_abs[df_abs["region"].isin(sel)].copy()
    monthly = (d.groupby(["region", "date"], as_index=False)
                ["turnover"].sum().sort_values(["region","date"]))
    monthly["roll3"] = monthly.groupby("region")["turnover"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    st.caption(f"Using ABS Retail Trade fallback. Range: {year_span[0]}–{year_span[1]}")

    fig = px.line(
        monthly, x="date", y="turnover", color="region",
        color_discrete_map=colors,
        labels={"turnover":"Turnover ($M)", "date":"Month", "region":""},
        title=f"Monthly Turnover — {title_suffix}"
    )
    fig_avg = px.line(monthly, x="date", y="roll3", color="region",
                      color_discrete_map=colors)
    for tr in fig_avg.data:
        tr.name = f"{tr.name} (3-mo avg)"
        tr.line.width = 2.5
        tr.line.dash = "dash"
        fig.add_trace(tr)
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Decomposition (Trend / Seasonality / Residual)")
    target_reg = st.selectbox("Select region for decomposition", sel)
    if STL is None:
        st.info("Install `statsmodels` to enable STL decomposition (pip install statsmodels).")
    else:
        df_reg = monthly[monthly["region"] == target_reg][["date","turnover"]].copy()
        _stl_section(df_reg, target_reg)

    st.subheader("Category Comparison — ABS vs UK Rules")
    rules_ind = _load_rules_industry(RULES_FILE)
    if "industry" in df_abs.columns and not rules_ind.empty:
        last12_cut = df_abs["date"].max() - pd.DateOffset(months=12)
        abs12 = (df_abs[df_abs["date"] >= last12_cut]
                    .dropna(subset=["industry"])
                    .groupby("industry", as_index=False)["turnover"].sum())
        abs12["abs_share"] = abs12["turnover"] / abs12["turnover"].sum()

        cmp = pd.merge(
            abs12[["industry","abs_share"]],
            rules_ind[["industry","rule_share"]],
            on="industry", how="inner"
        ).sort_values("abs_share", ascending=False)

        if cmp.empty:
            st.info("No overlapping industry labels between ABS and rules.")
        else:
            cmp_melt = cmp.melt(id_vars="industry", value_vars=["abs_share","rule_share"],
                                var_name="source", value_name="share")
            figc = px.bar(
                cmp_melt, x="industry", y="share", color="source", barmode="group",
                labels={"industry":"Industry", "share":"Share"},
                title="ABS Turnover Share (last 12m) vs UK Rule Share"
            )
            figc.update_layout(xaxis_tickangle=-30, margin=dict(l=0,r=0,t=60,b=0))
            st.plotly_chart(figc, use_container_width=True)
            with st.expander("Comparison table"):
                st.dataframe(cmp, use_container_width=True)
    else:
        st.info("Industry-level comparison unavailable (missing `industry` in ABS or rules).")

    with st.expander("ABS data preview"):
        st.dataframe(monthly.tail(24), use_container_width=True)

    time_dynamics_docs()

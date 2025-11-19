import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---- Helper UI utilities for Time Dynamics ----
PALETTE = {"blue": "#264E86", "green": "#2CA58D", "orange": "#F4A300"}


def time_dynamics_helpbar():
    """Compact helper bubbles for quick guidance."""
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1.popover("📝 What this page shows", use_container_width=True):
        st.markdown(
            """
            - Uses **ABS 8501.0** to track how retail turnover moves over time.  
            - Lets you **slide the year range** and see how the picture changes.  
            - Focus is on **timing**: when things pick up, flatten out, or spike.
            """
        )
    with c2.popover("📊 How to read these chart", use_container_width=True):
        st.markdown(
            """
            - **Line:** monthly retail turnover for the selected region.
            - **3-month moving average:** smoother trend overlay on top of raw monthly values.
            - **Decomposition:**  breaks the series into **trend**, **seasonality**, and **residual noise**.
            - **Category comparison:** compares each industry's share of **ABS turnover** (last 12 months) with its share of all **mined UK rules**.
            """
        )
    with c3.popover("🌀 What the decomposition means", use_container_width=True):
        st.markdown(
            """
            - **Trend** – the long-run level, ignoring regular ups and downs.  
            - **Seasonality** – the repeating monthly pattern (e.g. December bump).  
            - **Residual** – one-off effects and noise (promos, supply shocks, data quirks).  

            Use it to separate **real growth** from the usual seasonal shape.
            """
        )
    with c4.popover("❓ Things to watch out for", use_container_width=True):
        st.markdown(
            """
            - ABS is **macro** – trends here won’t match any single store 1:1.  
            - Values are shown in **$ millions**.  
            - For decomposition, missing months are **interpolated** only for the STL model;  
            the main line chart always shows the raw reported months.  
            - The category comparison mixes **Australian ABS turnover** with **UK online rules** –  
            treat it as a **relative mix check**, not a precise calibration.
            """
        )


def time_dynamics_docs():
    """Long-form helper—put at the bottom of the tab."""
    with st.expander("Methodology & Data Notes", expanded=False):
        st.markdown(
            """
**Pipeline in brief (ABS Retail Trade)**

1. Load ABS 8501.0 CSV (header-agnostic detection of region, date, value, industry).  
2. Filter to **Measure = Current prices** and, where available, prefer  
   **Adjustment type = Seasonally adjusted**.  
3. Convert the value column to numeric, apply **10^UNIT_MULT**, then standardise to **$M**.  
4. Aggregate duplicates by **region × date (× industry)**.  
5. For region-level trend lines, use the **Total industry** series when present.  
6. Overlays:  
   - **3-month moving average** (per region)  
   - **STL decomposition** (trend / seasonality / residual).
   - **Industry mix comparison**: last 12 months of ABS turnover by industry vs the share of UK rules tagged to each industry.

**How to use**

- Use the main line chart to see how **national turnover** evolves over time.  
- Look at the **3-month moving average** to understand underlying trend beneath noisy monthly spikes.  
- Use the **decomposition** to separate long-run growth from seasonal patterns.
- Use the **industry comparison** to see which categories your UK rules over or under represent relative to their share of Australian retail turnover.



**Interpretation tips**

- A rising **trend** over a few years suggests sustained growth, not just seasonal peaks.  
- Seasonal **peaks** suggest earlier ordering & roster ramp-up.  
- Large **residuals** in the decomposition hint at events (promo, supply shock); annotate these for future planning.
"""
        )


# optional: STL (graceful fallback if not installed)
try:
    from statsmodels.tsa.seasonal import STL
except Exception:
    STL = None

DATA_DIR = Path("data")
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

    region = get("region", "state", "state/territory", "geography", "geog")
    date = get("time_period", "time period", "period", "month", "date")
    value = get("obs_value", "observation value",
                "value", "turnover", "amount")
    industry = get("industry", "industry_code", "category", "group")
    return region, date, value, industry


@st.cache_data(show_spinner=False)
def _load_abs(csv_path: Path, keep_years: int = 7):
    """
    Load ABS retail CSV and return a clean, level series in $M:
    - Filters to Measure = Current prices
    - Prefers Adjustment type = Seasonally adjusted (if available)
    - Applies UNIT_MULT and converts to millions of dollars
    - Aggregates duplicates by region × date (× industry)
    """
    # Detect main columns via helper
    region_col, date_col, value_col, industry_col = _detect_cols(csv_path)

    # Second pass over header to find Measure / Adjustment Type
    hdr = pd.read_csv(csv_path, nrows=0)
    cols_norm = {c.strip().lower(): c for c in hdr.columns}

    def pick(*names):
        for name in names:
            if name in cols_norm:
                return cols_norm[name]
        return None

    measure_col = pick("measure", "measures")
    adj_col = pick("adjustment type", "adjustment_type", "adjustment")

    need = [
        c
        for c in [
            region_col,
            date_col,
            value_col,
            industry_col,
            "UNIT_MULT",
            measure_col,
            adj_col,
        ]
        if c
    ]

    if not all([region_col, date_col, value_col]):
        return pd.DataFrame(), {
            "region": region_col,
            "date": date_col,
            "value": value_col,
            "industry": industry_col,
        }

    df = pd.read_csv(csv_path, usecols=need, low_memory=False)

    # Filter to a single, meaningful level series
    if measure_col and measure_col in df.columns:
        df[measure_col] = df[measure_col].astype(str).str.strip().str.lower()
        df = df[df[measure_col] == "current prices"]

    if adj_col and adj_col in df.columns:
        df[adj_col] = df[adj_col].astype(str)
        # default to seasonally adjusted if present
        mask_sa = df[adj_col].str.contains("seasonally", case=False, na=False)
        if mask_sa.any():
            df = df[mask_sa]

    # Rename to internal names
    ren = {region_col: "region", date_col: "date", value_col: "turnover"}
    if industry_col:
        ren[industry_col] = "industry"
    if measure_col:
        ren[measure_col] = "measure"
    if adj_col:
        ren[adj_col] = "adj_type"
    df = df.rename(columns=ren)

    # Numeric + unit scaling
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    if "UNIT_MULT" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            df["UNIT_MULT"] = pd.to_numeric(
                df["UNIT_MULT"], errors="coerce"
            ).fillna(0)
            df["turnover"] = df["turnover"] * (10 ** df["UNIT_MULT"])
        df = df.drop(columns=["UNIT_MULT"])

    # If region missing for some reason, default
    if "region" not in df.columns:
        df["region"] = "Australia"

    # Dates → month start
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "turnover"])

    df["month"] = df["date"].dt.to_period("M").dt.start_time
    df["date"] = df["month"]
    df = df.drop(columns=["month"])

    # Keep latest N years for readability
    cut = df["date"].max() - pd.DateOffset(years=keep_years)
    df = df[df["date"] >= cut]

    # Aggregate duplicates
    keys = ["region", "date"] + \
        (["industry"] if "industry" in df.columns else [])
    df = df.groupby(keys, as_index=False)["turnover"].sum()

    # Convert to $M
    if df["turnover"].max() > 1e6:
        df["turnover"] = df["turnover"] / 1e6

    # Drop helper columns if present
    for col in ["measure", "adj_type"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df, {
        "region": region_col,
        "date": date_col,
        "value": value_col,
        "industry": industry_col,
    }


@st.cache_data(show_spinner=False)
def _load_rules_industry(path: Path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    if "industry" not in df.columns:
        return pd.DataFrame()
    s = (
        df.groupby("industry", as_index=False)
        .size()
        .rename(columns={"size": "rule_count"})
    )
    s["rule_share"] = s["rule_count"] / s["rule_count"].sum()
    return s.sort_values("rule_share", ascending=False)


def _year_filter(df, date_col="date"):
    y_min = int(df[date_col].dt.year.min())
    y_max = int(df[date_col].dt.year.max())
    y0, y1 = st.slider(
        "Year range",
        y_min,
        y_max,
        (max(y_min, y_max - 3), y_max),
    )
    return (
        df[(df[date_col].dt.year >= y0) & (df[date_col].dt.year <= y1)],
        (y0, y1),
    )


def _stl_section(df_reg: pd.DataFrame, region_label: str):
    if STL is None:
        st.info(
            "Install `statsmodels` to enable STL decomposition (pip install statsmodels)."
        )
        return
    s = (
        df_reg.set_index("date")["turnover"]
        .asfreq("MS")
        .interpolate("linear")
    )
    if len(s) < 12:
        st.info("Not enough points for decomposition (need ≥ 12 months).")
        return
    res = STL(s, period=12, robust=True).fit()
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.07,
    )
    fig.update_yaxes(title_text="Turnover ($M)", row=1, col=1)
    fig.update_yaxes(title_text="Turnover ($M)", row=2, col=1)
    fig.update_yaxes(title_text="Turnover ($M)", row=3, col=1)
    fig.update_yaxes(title_text="Turnover ($M)", row=4, col=1)

    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=""), 1, 1)
    fig.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name=""), 2, 1)
    fig.add_trace(go.Scatter(x=res.seasonal.index,
                  y=res.seasonal, name=""), 3, 1)
    fig.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name=""), 4, 1)
    fig.update_layout(
        height=900,
        title=f"STL Decomposition — {region_label}",
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def show():
    st.title("Time Dynamics")
    time_dynamics_helpbar()

    # ------------------------------------------------------------------
    # ABS Retail Trade (ABS.csv) – only path now
    # ------------------------------------------------------------------
    if not ABS_FILE.exists():
        st.error(f"ABS file not found: {ABS_FILE}")
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

    # For region-level trends, prefer the 'Total' industry series if present
    trend_df = df_abs.copy()
    if "industry" in trend_df.columns:
        mask_total = trend_df["industry"].astype(str).str.lower() == "total"
        if mask_total.any():
            trend_df = trend_df[mask_total]

    regs = trend_df["region"].astype(str).unique().tolist()
    if {"New South Wales", "Victoria"}.issubset(regs):
        sel = ["New South Wales", "Victoria"]
        colors = {"New South Wales": "#264E86", "Victoria": "#2CA58D"}
        title_suffix = "NSW vs VIC"
    elif "Australia" in regs:
        sel = ["Australia"]
        colors = {"Australia": "#264E86"}
        title_suffix = "Australia"
    else:
        sel = (
            trend_df.groupby("region")["turnover"]
            .sum()
            .sort_values(ascending=False)
            .head(2)
            .index
            .tolist()
        )
        colors = {
            sel[i]: ("#264E86" if i == 0 else "#2CA58D") for i in range(len(sel))
        }
        title_suffix = " — ".join(sel)

    d = trend_df[trend_df["region"].isin(sel)].copy()
    monthly = (
        d.groupby(["region", "date"], as_index=False)["turnover"]
        .sum()
        .sort_values(["region", "date"])
    )
    monthly["roll3"] = monthly.groupby("region")["turnover"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    st.caption(
        f"Using ABS Retail Trade. Range: {year_span[0]}–{year_span[1]}"
    )

    fig = px.line(
        monthly,
        x="date",
        y="turnover",
        color="region",
        color_discrete_map=colors,
        labels={"turnover": "Turnover ($M)", "date": "Month", "region": ""},
        title=f"Monthly Retail Turnover — {title_suffix}",
    )
    fig_avg = px.line(
        monthly,
        x="date",
        y="roll3",
        color="region",
        color_discrete_map=colors,
    )
    for tr in fig_avg.data:
        tr.name = f"{tr.name} (3-mo avg)"
        tr.line.width = 2.5
        tr.line.dash = "dash"
        fig.add_trace(tr)

    fig.update_traces(
        hovertemplate=(
            "<b>Region:</b> %{fullData.name}<br>"
            "<b>Month:</b> %{x|%b %d, %Y}<br>"
            "<b>Turnover ($M):</b> %{y:,.1f}"
            "<extra></extra>"
        )
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    # STL decomposition on the primary region
    st.subheader("Trend & Seasonality Breakdown")
    target_reg = sel[0]
    if STL is None:
        st.info(
            "Install `statsmodels` to enable STL decomposition (pip install statsmodels)."
        )
    else:
        df_reg = monthly[monthly["region"] ==
                         target_reg][["date", "turnover"]].copy()
        _stl_section(df_reg, target_reg)

    # ------------------------------------------------------------------
    # Category Comparison — ABS vs UK Rules
    # ------------------------------------------------------------------
    st.subheader("Category Comparison — ABS vs UK Rules")
    rules_ind = _load_rules_industry(RULES_FILE)
    if "industry" in df_abs.columns and not rules_ind.empty:
        # True last 12 months: max-11 … max (inclusive)
        last12_cut = df_abs["date"].max() - pd.DateOffset(months=11)
        abs12 = df_abs[df_abs["date"] >= last12_cut].copy()

        # Exclude the rolled-up 'Total' so we compare component industries
        abs12 = abs12[abs12["industry"].astype(str).str.lower() != "total"]

        abs12 = (
            abs12.dropna(subset=["industry"])
            .groupby("industry", as_index=False)["turnover"]
            .sum()
        )
        abs12["abs_share"] = abs12["turnover"] / abs12["turnover"].sum()

        cmp = (
            pd.merge(
                abs12[["industry", "abs_share"]],
                rules_ind[["industry", "rule_share"]],
                on="industry",
                how="inner",
            )
            .sort_values("abs_share", ascending=False)
        )

        if cmp.empty:
            st.info("No overlapping industry labels between ABS and rules.")
        else:
            cmp_melt = cmp.melt(
                id_vars="industry",
                value_vars=["abs_share", "rule_share"],
                var_name="source",
                value_name="share",
            )
            cmp_melt["source"] = cmp_melt["source"].map({
                "abs_share": "ABS turnover share (last 12m)",
                "rule_share": "UK rules share"
            })
            figc = px.bar(
                cmp_melt,
                x="industry",
                y="share",
                color="source",
                barmode="group",
                labels={"industry": "Industry", "share": "Share of total"},
                title="ABS Turnover Share vs UK Rules Mix",
            )
            figc.update_yaxes(tickformat=".0%")
            figc.update_traces(
                hovertemplate=(
                    "<b>Industry:</b> %{x}<br>"
                    "<b>Share:</b> %{y:.1%}<br>"
                    "<b>Source:</b> %{fullData.name}"
                    "<extra></extra>"
                )
            )
            figc.update_layout(
                xaxis_tickangle=-30,
                margin=dict(l=0, r=0, t=60, b=0),
            )
            st.plotly_chart(figc, use_container_width=True)

            with st.expander("Comparison Data"):
                cmp_display = cmp.copy()
                cmp_display["industry"] = (
                    cmp_display["industry"].astype(str).str.title()
                )
                cmp_display["abs_share"] = (
                    cmp_display["abs_share"] * 100
                ).map("{:.1f}%".format)
                cmp_display["rule_share"] = (
                    cmp_display["rule_share"] * 100
                ).map("{:.1f}%".format)

                cmp_display = cmp_display.rename(
                    columns={
                        "industry": "Industry",
                        "abs_share": "ABS Share (%)",
                        "rule_share": "Rule Share (%)",
                    }
                )

                cmp_display = cmp_display.reset_index(drop=True)

                st.dataframe(cmp_display, use_container_width=True)
    else:
        st.info(
            "Industry-level comparison unavailable (missing `industry` in ABS or rules)."
        )

    # ------------------------------------------------------------------
    # Source data preview
    # ------------------------------------------------------------------
    with st.expander("Source Data (ABS Retail Trade)"):
        # Take the tail (most recent months), create a copy to avoid warnings
        m_display = monthly.tail(24).copy()

        # Format Date to "Jan 2024" style
        m_display["date"] = m_display["date"].dt.strftime("%b %Y")

        # Format Region to Title Case
        if "region" in m_display.columns:
            m_display["region"] = m_display["region"].astype(str).str.title()

        # Rename columns to caps
        m_display = m_display.rename(
            columns={
                "region": "Region",
                "date": "Month",
                "turnover": "Turnover ($M)",
                "roll3": "3-Month Avg ($M)",
            }
        )

        m_display = m_display.reset_index(drop=True)

        st.dataframe(m_display, use_container_width=True)

    time_dynamics_docs()

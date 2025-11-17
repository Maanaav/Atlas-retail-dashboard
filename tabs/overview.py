import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

ABS_FILE = Path("data/ABS.csv")

# ------------------- Helper UI utilities for Overview ----------------------


def overview_helpbar():
    """Top helper row — concise guidance for first-time viewers."""
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1.popover("🗂️ About this data", use_container_width=True):
        st.markdown(
            """
- **Source** – Australian Bureau of Statistics, *Retail Trade, Australia (ABS 8501.0)*.  
- **Measure** – Monthly **retail turnover ($ millions)** by state and industry.  
- **Update frequency** – Published monthly, usually with a 4–5 week delay.  
- **Coverage** – Includes both in-store and online retail (excludes pure services).
"""
        )
    with c2.popover("📊 How to read these chart", use_container_width=True):
        st.markdown(
            """
- **Each line** shows one region’s monthly turnover over time.  
- **Peaks** (usually Nov–Dec) indicate seasonal demand; **dips** are off-season.  
- **YoY %** in the KPIs compares the latest month with the same month last year.  
- The optional **3-month moving average** smooths noise so trends are easier to see.
"""
        )
    with c3.popover("❓Things to watch out for", use_container_width=True):
        st.markdown(
            """
- **Turnover ≠ number of sales** – it’s the dollar value of sales, not the count.  
- **ABS data ≠ UCI baskets** – ABS is macro context; your basket rules are micro-level.  
- **Industry shares** follow ABS classifications, which may differ from your internal categories.
"""
        )


def overview_docs():
    """Optional long-form explainer at the bottom of the tab."""
    with st.expander("Methodology & Data Notes", expanded=False):
        st.markdown(
            """
### Data preparation

1. **Column detection:** header-agnostic parsing finds *region*, *date*, *turnover*, *industry*.  
2. **Normalization:** converts turnover to numeric, applies `UNIT_MULT` (×10ⁿ) if present, and rescales to **$ millions**.  
3. **Filtering:** keeps the most recent 3–5 years for readability.  
4. **Aggregation:** sums duplicates by `region × date (× industry)`.

### Visual design

- **Line chart:** monthly trend by region with color palette  
  (`#264E86` NSW, `#2CA58D` VIC, `#F4A300` AUS, gray others).  
- **KPIs:** dynamic metrics showing total turnover and YoY growth.  
- **Bar chart:** appears only when one region selected; shows top industries (last 12 months).

### Interpretation guidance

- **Rising turnover** = higher retail spending, not necessarily higher unit sales.  
- **Stable pattern with annual spikes** = normal retail seasonality.  
- **Divergent state trends** may indicate local policy, tourism, or population shifts.  

**Tip:** Use this page to anchor *micro-level* basket insights in *macro-level* economic reality.
"""
        )


# ---------- Core Logic for Overview Tab ----------
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


@st.cache_data(show_spinner=False, ttl=600)
def load_abs(csv_path: Path, keep_years: int = 5):
    region_col, date_col, value_col, industry_col = _detect_cols(csv_path)
    need = [c for c in [region_col, date_col,
                        value_col, industry_col, "UNIT_MULT"] if c]
    if not all([region_col, date_col, value_col]):
        return pd.DataFrame(), {"region": region_col, "date": date_col, "value": value_col, "industry": industry_col}
    df = pd.read_csv(csv_path, usecols=need, low_memory=False)
    ren = {region_col: "region", date_col: "date", value_col: "turnover"}
    if industry_col:
        ren[industry_col] = "industry"
    df = df.rename(columns=ren)
    # Clean strings and unit multipliers
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    if "UNIT_MULT" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            df["UNIT_MULT"] = pd.to_numeric(
                df["UNIT_MULT"], errors="coerce").fillna(0)
            df["turnover"] = df["turnover"] * (10 ** df["UNIT_MULT"])
        df = df.drop(columns=["UNIT_MULT"])
    # Handle date parsing, force to first of month if only year-month
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Drop completely empty or invalid
    df = df.dropna(subset=["date", "region", "turnover"])
    # Aggregate to monthly
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    df["date"] = df["month"]
    df = df.drop(columns=["month"])
    # Keep latest N years
    cut = df["date"].max() - pd.DateOffset(years=keep_years)
    df = df[df["date"] >= cut]
    # Aggregate all duplicates
    keys = ["region", "date"] + \
        (["industry"] if "industry" in df.columns else [])
    df = df.groupby(keys, as_index=False)["turnover"].sum()
    if df["turnover"].max() > 1e6:
        df["turnover"] = df["turnover"] / 1e6

    df["region"] = df["region"].astype(str).str.title()
    if "industry" in df.columns:
        df["industry"] = df["industry"].astype(str).str.title()
    return df, {"region": region_col, "date": date_col, "value": value_col, "industry": industry_col}


def show(DATA_DIR: Path = Path("data")):
    st.title("Overview: Australian Retail Trends (ABS 8501.0)")
    overview_helpbar()  # <<---------------- helper UI row!

    path = DATA_DIR / "ABS.csv"
    if not path.exists():
        st.error(f"File not found: {path}")
        return
    df, detected = load_abs(path)
    if df.empty:
        st.error(
            f"No usable rows. Detected columns → {detected}. Check header names in ABS.csv.")
        return

    geos = sorted(df["region"].astype(str).unique().tolist())
    default_geo = [g for g in geos if g.startswith(
        "New South Wales") or g.startswith("Victoria")]
    if not default_geo:
        default_geo = geos[:2]

    sel_geo = geos
    df_geo = df.copy()

    start = df_geo["date"].min()
    end = df_geo["date"].max()

    # Check for duplicate dates per region
    if df_geo.groupby(["region", "date"]).size().max() > 1:
        st.warning(
            "Note: Some regions contained multiple records per month. "
            "These have been aggregated so each region has a single monthly total."
        )

    st.markdown(
        f"""
        <div style="
            margin-top:4px;
            margin-bottom:16px;
            padding:6px 12px;
            border-radius:999px;
            background:#EEF2FF;
            display:inline-flex;
            align-items:center;
            gap:10px;
            font-size:0.85rem;
            color:#111827;">
          <span>📅 <b>{start:%b %Y}</b> – <b>{end:%b %Y}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPIs for latest month
    latest = df_geo["date"].max()
    now = df_geo[df_geo["date"] == latest].groupby(
        "region", as_index=False)["turnover"].sum()
    prev = df_geo[df_geo["date"] == (
        latest - pd.DateOffset(years=1))].groupby("region", as_index=False)["turnover"].sum()
    total_now = float(now["turnover"].sum())
    total_prev = float(prev["turnover"].sum()) if not prev.empty else 0.0
    yoy = ((total_now - total_prev) / total_prev *
           100.0) if total_prev > 0 else 0.0

    c1, c2 = st.columns(2) if len(sel_geo) == 1 else st.columns(3)
    c1.metric("Total Turnover (latest)",
              f"${total_now:,.0f}M", f"{yoy:+.1f}% YoY")
    c2.metric(f"{sel_geo[0]} (latest)",
              f"${float(now[now.region == sel_geo[0]]['turnover'].sum()):,.0f}M" if sel_geo else "—")
    if len(sel_geo) > 1:
        c3 = c2 if len(sel_geo) == 2 else st.columns(3)[2]
        c3.metric(f"{sel_geo[1]} (latest)",
                  f"${float(now[now.region == sel_geo[1]]['turnover'].sum()):,.0f}M")

    st.download_button(
        "Download filtered (CSV)",
        df_geo.to_csv(index=False).encode("utf-8"),
        file_name="abs_filtered.csv",
        mime="text/csv"
    )

    st.subheader("Time Trends")
    smooth = st.toggle("Show 3-month moving average", value=True,
                       help="Adds a 3-month moving average line per region.")
    plot_df = df_geo.copy()
    if smooth:
        plot_df["ma3"] = (
            plot_df.sort_values(["region", "date"])
            .groupby("region")["turnover"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )

    fig = px.line(
        plot_df, x="date", y="turnover", color="region",
        color_discrete_sequence=["#264E86", "#2CA58D", "#F4A300", "#6c757d"],
        labels={"turnover": "Turnover ($M)",
                "date": "Month", "region": "Region"},
        title="Monthly Retail Turnover"
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Region:</b> %{fullData.name}<br>"
            "<b>Month:</b> %{x|%b %d, %Y}<br>"
            "<b>Turnover ($M):</b> %{y:,.1f}"
        )
    )

    if smooth:
        for r, g in plot_df.groupby("region"):
            fig.add_scatter(x=g["date"], y=g["ma3"], mode="lines", name=f"{r} · 3M MA",
                            line=dict(width=2, dash="dash"),
                            showlegend=True)
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # If industry present and only one region selected, quick bar
    if "industry" in df_geo.columns and len(sel_geo) == 1:
        st.subheader(
            f"Turnover by industry — {sel_geo[0]} (last 12 months)")
        last12 = df_geo[df_geo["date"] >=
                        df_geo["date"].max() - pd.DateOffset(months=12)]
        ind = last12.groupby("industry", as_index=False)["turnover"].sum()
        ind["industry"] = ind["industry"].astype(str).str.title()
        ind = ind[ind["industry"].str.lower() != "total"]
        ind_display = ind.rename(
            columns={
                "industry": "Industry",
                "turnover": "Turnover ($M)",
            }
        )
        fig_ind = px.bar(
            ind_display,
            x="Industry",
            y="Turnover ($M)",
        )
        fig_ind.update_traces(
            hovertemplate=(
                "<b>Industry:</b> %{x}<br>"
                "<b>Turnover ($M):</b> %{y:,.1f}"
            )
        )
        fig_ind.update_layout(
            xaxis_tickangle=-35,   # rotate labels
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_ind, use_container_width=True)

    with st.expander("Data Preview"):
        preview = df_geo.sort_values(["region", "date"]).head(24).copy()

        # nicer month label
        preview["date"] = preview["date"].dt.strftime("%b %Y")

        # capitalised / descriptive column names
        preview = preview.rename(
            columns={
                "region": "Region",
                "date": "Month",
                "industry": "Industry",
                "turnover": "Turnover ($M)",
            }
        )

        st.dataframe(preview, use_container_width=True)

    # Optional: add detailed docs at the bottom
    overview_docs()

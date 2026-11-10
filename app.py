import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

DATA = "data/"
st.set_page_config(page_title="Ahmedabad Urban Heat", layout="wide")


@st.cache_data
def load_data():
    pts = pd.read_csv(DATA + "dash_points.csv")
    loc = pd.read_csv(DATA + "dash_locations.csv")
    res = pd.read_csv(DATA + "model_results.csv")
    return pts, loc, res


@st.cache_resource
def load_model():
    # joblib uses pickle: only load the model file you exported yourself from Colab
    return joblib.load(DATA + "rf_lst_2024.joblib")


pts, loc, res = load_data()
model = load_model()

LEVELS = ["Low", "Moderate", "High"]
LEVEL_COLORS = {"Low": "#fee8c8", "Moderate": "#fdbb84", "High": "#e34a33"}   # ordered: light → dark


def point_map(data, color, scale=None, rng=None, hover=None):
    fig = px.scatter_map(
        data, lat="latitude", lon="longitude", color=color,
        color_continuous_scale=scale, range_color=rng,
        color_discrete_map=LEVEL_COLORS, category_orders={color: LEVELS},
        hover_data=hover, center=dict(lat=23.02, lon=72.57), zoom=10.3, height=620,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


# ---------- Header and year selector ----------
st.title("Urban Heat in Ahmedabad: A Satellite-Based Analysis")
st.caption("Landsat 8 land surface temperature (LST), March–May median composites · "
           "100 m sample grid · 921.75 km² study area · ESA WorldCover 2021 land cover")

year = st.sidebar.selectbox("Year", sorted(pts["year"].unique()), index=2)
d = pts[pts["year"] == year]
lc_means = d.groupby("landcover_name")["LST"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean LST", f"{d['LST'].mean():.1f} °C")
c2.metric("Max LST", f"{d['LST'].max():.1f} °C")
c3.metric("Built-up − Cropland", f"{lc_means['Built-up'] - lc_means['Cropland']:+.2f} °C")
c4.metric("Built-up − Tree cover", f"{lc_means['Built-up'] - lc_means['Tree cover']:+.2f} °C")

tab_map, tab_rel, tab_risk, tab_model = st.tabs(["Maps", "Relationships", "Heat risk", "Model"])

# ---------- Maps ----------
LAYERS = {
    "LST (°C)":                        ("LST", "OrRd"),
    "LST relative to city mean (°C)":  ("LST_rel", "RdBu_r"),
    "NDVI (vegetation)":               ("NDVI", "Greens"),
    "NDBI (surface dryness)":          ("NDBI", "Purples"),
    "NDWI":                            ("NDWI", "Blues"),
}
with tab_map:
    layer = st.radio("Layer", list(LAYERS), horizontal=True)
    col, scale = LAYERS[layer]
    lo, hi = d[col].quantile([0.01, 0.99])
    if col == "LST_rel":                       # diverging scale: keep 0 in the middle
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    st.plotly_chart(point_map(d, col, scale, (lo, hi), ["landcover_name", "LST", "NDVI", "NDBI"]))

# ---------- Relationships ----------
with tab_rel:
    left, right = st.columns(2)
    with left:
        idx = st.selectbox("Index", ["NDVI", "NDBI", "NDWI"])
        fig = px.density_heatmap(d, x=idx, y="LST", nbinsx=60, nbinsy=60,
                                 color_continuous_scale="Blues", height=450)
        st.plotly_chart(fig)
        st.write(f"Pearson r = {d[idx].corr(d['LST']):.2f} · "
                 f"Spearman ρ = {d[idx].corr(d['LST'], method='spearman'):.2f}")
    with right:
        st.markdown("**Average LST by land cover**")
        bars = lc_means.sort_values().reset_index()
        fig = px.bar(bars, x="LST", y="landcover_name", orientation="h", height=450,
                     labels={"LST": "Mean LST (°C)", "landcover_name": ""})
        fig.update_traces(marker_color="#2a78d6")
        st.plotly_chart(fig)

# ---------- Heat risk ----------
with tab_risk:
    view = st.radio("Show", ["Heat risk within built-up areas", "Heat hazard, all land"], horizontal=True)
    col = "risk" if view.startswith("Heat risk") else "hazard"
    st.plotly_chart(point_map(loc.dropna(subset=[col]), col, hover=["landcover_name", "LST_rel_mean"]))
    st.caption("Levels from the 2015–2024 mean LST relative to the city mean: Low < 50th percentile, "
               "Moderate 50–80th, High ≥ 80th. Built-up risk is ranked among built-up areas only. "
               "Daytime surface temperature; does not include population or vulnerability.")

# ---------- Model ----------
with tab_model:
    st.subheader("Model comparison")
    st.caption("The spatial 5-fold split (3 km blocks) is the honest score: it tests on neighbourhoods the model never saw.")
    st.dataframe(res, hide_index=True, width="stretch")

    st.subheader("What-if: predicted 2024 LST")
    a, b, c = st.columns(3)
    ndvi = a.slider("NDVI", -0.1, 0.8, 0.25, 0.01)
    ndbi = b.slider("NDBI", -0.4, 0.2, 0.0, 0.01)
    lc = c.selectbox("Land cover", sorted(pts["landcover_name"].unique()))
    pred = model.predict(pd.DataFrame({"NDVI": [ndvi], "NDBI": [ndbi], "landcover_name": [lc]}))[0]
    st.metric("Predicted LST", f"{pred:.1f} °C")
    st.caption("NDVI and NDBI are correlated (r ≈ −0.73). Unrealistic combinations, "
               "such as very high NDVI with very high NDBI, give unreliable predictions.")

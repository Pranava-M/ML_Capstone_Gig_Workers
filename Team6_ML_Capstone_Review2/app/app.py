"""Gig Analytics — Review 2 dashboard (Team 6, 23CSE301 Machine Learning).

One page per dataset / ML track:
    Fares     -> Uber Fares dataset            (regression,     notebooks/regression.ipynb)
    Bookings  -> Ola & Uber Bookings dataset   (classification, notebooks/classification.ipynb)
    Segments  -> NCR Ride Bookings dataset     (clustering,     notebooks/clustering.ipynb)

Run from the project root:   streamlit run app/app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data", ROOT / "models"
RANDOM_STATE = 42  # identical to the notebooks, so the test split is the same

INK, MUTED, ACCENT, GREEN, SAND, TAN = "#14110f", "#7a7168", "#c4573a", "#4f7a63", "#cdbfb2", "#a8735a"
COLORWAY = [ACCENT, "#2f3a45", GREEN, "#d9a441", TAN, "#6d5a8a", "#8aa0b4", "#b04a6a", "#5c8d89", "#9c9184"]

st.set_page_config(page_title="Gig Analytics · Team 6", page_icon="🚕", layout="wide", initial_sidebar_state="collapsed")

PAGES = ["Overview", "Fares", "Bookings", "Segments", "About"]
if "page" not in st.session_state:
    st.session_state.page = "Overview"
PAGE = st.session_state.page


def go_to(p):
    st.session_state.page = p


# =============================================================== STYLE
CSS = f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,500;1,9..144,600&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap");
:root {{ --ink:{INK}; --muted:{MUTED}; --accent:{ACCENT}; --green:{GREEN}; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: radial-gradient(900px 700px at 0% 8%, rgba(232,180,160,.55) 0%, rgba(232,180,160,0) 60%),
              radial-gradient(900px 700px at 100% 0%, rgba(214,222,212,.6) 0%, rgba(214,222,212,0) 60%),
              linear-gradient(180deg, #efe7dc 0%, #f1ebe2 100%) !important;
  color: var(--ink); font-family: 'Inter', system-ui, sans-serif; }}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer, [data-testid="stSidebar"] {{ display:none !important; }}
.block-container {{ max-width: 1240px; padding-top: 1.1rem; padding-bottom: 4rem; position: relative; z-index: 2; }}
.bgword {{ position: fixed; left: 50%; top: 54%; transform: translate(-50%,-50%); z-index: 0; pointer-events: none; font-family: 'Fraunces', serif; font-weight: 600; font-size: 25vw; line-height: 1; color: rgba(20,17,15,.04); white-space: nowrap; }}
.brand {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.2rem; color:var(--ink); white-space:nowrap; }}
.brand b {{ color: var(--accent); padding:0 .15rem; }}
.eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.2em; text-transform:uppercase; color:var(--muted); }}
.chip {{ display:inline-flex; align-items:center; gap:.6rem; font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.2em; text-transform:uppercase; color:var(--green); border:1px solid rgba(20,17,15,.14); border-radius:999px; padding:.42rem 1.05rem; background:rgba(255,255,255,.25); }}
.chip:before {{ content:''; width:7px; height:7px; border-radius:50%; background:var(--green); }}
h1.hero {{ font-family:'Fraunces',serif; font-weight:500; font-size:clamp(2.6rem,5.6vw,5rem); line-height:1.03; letter-spacing:-.02em; margin:1.1rem 0 1.3rem; color:#211b17; }}
h1.hero em, h2.sec em {{ color: var(--accent); font-style: italic; font-weight:600; }}
h2.sec {{ font-family:'Fraunces',serif; font-weight:500; font-size:clamp(2rem,4vw,3.1rem); line-height:1.05; letter-spacing:-.02em; margin:.5rem 0 .4rem; }}
h3.sub {{ font-family:'Fraunces',serif; font-weight:500; font-size:1.55rem; margin:2rem 0 .2rem; }}
p.lead {{ font-size:1.12rem; color:#5d554e; max-width:640px; line-height:1.7; }}
p.lead b {{ color:var(--ink); font-weight:600; }}
.card {{ background:rgba(255,255,255,.45); border:1px solid rgba(20,17,15,.09); border-radius:22px; padding:1.2rem 1.4rem; backdrop-filter: blur(6px); height:100%; }}
.kpi {{ font-family:'Fraunces',serif; font-weight:500; font-size:2.2rem; line-height:1.05; }}
.kpi em {{ color:var(--accent); font-style:normal; }}
.kpilabel {{ font-family:'JetBrains Mono',monospace; font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin-top:.45rem; }}
.track {{ font-family:'Fraunces',serif; font-size:1.45rem; font-weight:500; margin:.35rem 0 .5rem; }}
.track-meta {{ font-size:.92rem; color:#5d554e; line-height:1.6; }}
.track-meta b {{ color:var(--ink); }}
.big {{ font-family:'Fraunces',serif; font-weight:500; font-size:4.2rem; line-height:1; letter-spacing:-.03em; }}
.big em {{ color:var(--accent); font-style:italic; }}
.pbar {{ height:10px; border-radius:99px; background:rgba(20,17,15,.08); overflow:hidden; margin:.25rem 0 .85rem; }}
.pbar > div {{ height:100%; border-radius:99px; background:var(--accent); }}
.plabel {{ display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:.78rem; }}
.note {{ font-size:.9rem; color:#6b635b; border-left:2px solid var(--accent); padding-left:.8rem; margin:1rem 0; line-height:1.6; }}
.note b {{ color:var(--ink); }}
.flow {{ display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }}
.flow span.step {{ border:1px solid rgba(20,17,15,.15); border-radius:999px; padding:.35rem .8rem; background:rgba(255,255,255,.35); color:var(--ink); }}
hr.soft {{ border:none; border-top:1px solid rgba(20,17,15,.1); margin:1.8rem 0; }}
/* nav = buttons rendered as mono text pills */
div[class*="st-key-nav_"] button {{ background:transparent !important; border:none !important; border-radius:999px !important; padding:.45rem 1rem !important; min-height:0 !important; box-shadow:none !important; }}
div[class*="st-key-nav_"] button p {{ font-family:'JetBrains Mono',monospace !important; font-size:.76rem !important; letter-spacing:.14em; text-transform:uppercase; color:#6b635b !important; }}
div[class*="st-key-nav_"] button:hover {{ background:rgba(20,17,15,.05) !important; }}
div.st-key-nav_{PAGE} button {{ background:rgba(20,17,15,.09) !important; }}
div.st-key-nav_{PAGE} button p {{ color:var(--ink) !important; }}
/* primary buttons = black pill, ghost buttons = outlined pill */
.stButton > button {{ background:var(--ink); color:#fff; border:1px solid var(--ink); border-radius:999px; padding:.75rem 1.6rem; font-family:'Inter',sans-serif; font-weight:500; transition:all .2s; }}
.stButton > button:hover {{ background:#2c2621; color:#fff; border-color:#2c2621; }}
.stButton > button p {{ font-size:1rem; }}
div[class*="st-key-ghost_"] button {{ background:rgba(255,255,255,.3) !important; color:var(--ink) !important; border:1px solid rgba(20,17,15,.2) !important; }}
div[class*="st-key-ghost_"] button p {{ color:var(--ink) !important; }}
div[class*="st-key-ghost_"] button:hover {{ background:rgba(255,255,255,.65) !important; }}
[data-testid="stWidgetLabel"] p {{ font-family:'JetBrains Mono',monospace !important; font-size:.7rem !important; letter-spacing:.12em; text-transform:uppercase; color:var(--muted) !important; }}
div[data-baseweb="select"] > div, .stNumberInput input {{ background:rgba(255,255,255,.55) !important; border-radius:14px !important; }}
div[data-testid="stDataFrame"] {{ border-radius:16px; overflow:hidden; }}
[data-testid="stTabs"] button p {{ font-family:'JetBrains Mono',monospace; font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; }}
[data-testid="stExpander"] {{ background:rgba(255,255,255,.35); border-radius:18px; border:1px solid rgba(20,17,15,.09); }}
@media (max-width: 640px) {{ .bgword {{ font-size: 40vw; }} .big {{ font-size:3rem; }} }}
</style>
<div class="bgword">RIDES</div>
"""
st.markdown("\n".join(l for l in CSS.splitlines() if l.strip()), unsafe_allow_html=True)

# Hero illustration: a stylised city map with a ride in progress (pickup pin -> route -> drop pin),
# driver-search rings, nearby drivers and a fare meter tag.
HERO_SVG = f"""
<svg viewBox="0 0 520 520" width="100%" style="max-width:520px;display:block;margin:0 auto" fill="none" font-family="JetBrains Mono, monospace">
  <g stroke="{INK}" stroke-opacity=".10">
    {''.join(f'<line x1="{x}" y1="30" x2="{x}" y2="490"/>' for x in range(40, 500, 60))}
    {''.join(f'<line x1="20" y1="{y}" x2="500" y2="{y}"/>' for y in range(50, 500, 60))}
  </g>
  <path d="M20 330 C 140 300, 220 380, 330 300 S 470 210, 505 230" stroke="{GREEN}" stroke-opacity=".35" stroke-width="14" stroke-linecap="round"/>
  <path d="M110 400 C 120 330, 170 300, 220 290 S 300 250, 320 200 S 360 120, 410 110" stroke="{ACCENT}" stroke-width="3.2" stroke-dasharray="9 8" stroke-linecap="round"/>
  <circle cx="110" cy="400" r="34" stroke="{ACCENT}" stroke-opacity=".35"/>
  <circle cx="110" cy="400" r="62" stroke="{ACCENT}" stroke-opacity=".2"/>
  <circle cx="110" cy="400" r="92" stroke="{ACCENT}" stroke-opacity=".1"/>
  <g transform="translate(110 400)"><path d="M0 0 C -14 -18 -18 -26 -18 -34 A 18 18 0 1 1 18 -34 C 18 -26 14 -18 0 0 Z" fill="{INK}"/><circle cy="-34" r="6.5" fill="#f1ebe2"/></g>
  <g transform="translate(410 110)"><path d="M0 0 C -14 -18 -18 -26 -18 -34 A 18 18 0 1 1 18 -34 C 18 -26 14 -18 0 0 Z" fill="{ACCENT}"/><circle cy="-34" r="6.5" fill="#f1ebe2"/></g>
  <g transform="translate(262 262) rotate(-38)">
    <rect x="-22" y="-12" width="44" height="24" rx="8" fill="{INK}"/>
    <rect x="-9" y="-9" width="16" height="18" rx="3" fill="#f1ebe2" fill-opacity=".85"/>
    <circle cx="-14" cy="13" r="4.5" fill="{ACCENT}"/><circle cx="14" cy="13" r="4.5" fill="{ACCENT}"/>
  </g>
  {''.join(f'<g transform="translate({x} {y})"><rect x="-8" y="-5" width="16" height="10" rx="3.5" fill="{TAN}" fill-opacity=".75"/></g>' for x, y in [(200,140),(330,420),(450,330),(70,190),(250,470),(470,470)])}
  <g transform="translate(330 300)">
    <rect x="0" y="0" width="150" height="74" rx="18" fill="#fffaf3" fill-opacity=".85" stroke="{INK}" stroke-opacity=".15"/>
    <text x="18" y="26" font-size="10" letter-spacing="2" fill="{MUTED}">FARE ESTIMATE</text>
    <text x="18" y="58" font-size="26" font-family="Fraunces, serif" fill="{INK}">$ 12.40</text>
  </g>
  <g transform="translate(40 60)">
    <rect x="0" y="0" width="150" height="64" rx="18" fill="#fffaf3" fill-opacity=".85" stroke="{INK}" stroke-opacity=".15"/>
    <text x="18" y="25" font-size="10" letter-spacing="2" fill="{MUTED}">BOOKING</text>
    <circle cx="24" cy="44" r="5" fill="{GREEN}"/><text x="36" y="49" font-size="14" font-family="Inter, sans-serif" fill="{INK}">Driver assigned</text>
  </g>
  {''.join(f'<circle cx="{x}" cy="{y}" r="2.6" fill="{c}" opacity=".55"/>' for x, y, c in [(150,250,INK),(380,170,GREEN),(480,90,ACCENT),(300,140,INK),(60,300,GREEN),(420,420,ACCENT),(180,480,INK)])}
</svg>
"""


# =============================================================== DATA + MODELS
@st.cache_data(show_spinner=False)
def load_csv(name, folder="data"):
    return pd.read_csv((DATA if folder == "data" else MODELS) / name)


@st.cache_resource(show_spinner=False)
def load_model(name):
    return joblib.load(MODELS / name)


def load_json(name):
    return json.loads((MODELS / name).read_text())


def safe(fn, *a):
    try:
        return fn(*a), None
    except Exception as e:  # missing file or scikit-learn version mismatch
        return None, str(e)


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


# =============================================================== HELPERS
def style_fig(fig, height=360, legend=True):
    fig.update_layout(height=height + (40 if legend else 0), margin=dict(l=10, r=10, t=50, b=90 if legend else 10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=INK, size=12), colorway=COLORWAY, showlegend=legend,
                      title_font=dict(family="Fraunces, serif", size=18), legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="top", y=-0.28, title_text=""),
                      hoverlabel=dict(bgcolor="#fffaf3", font_family="Inter"))
    fig.update_xaxes(gridcolor="#e2d8cc", zerolinecolor="#e2d8cc", linecolor=SAND)
    fig.update_yaxes(gridcolor="#e2d8cc", zerolinecolor="#e2d8cc", linecolor=SAND)
    return fig


def chart(fig):
    try:  # newer Streamlit
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    except Exception:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def table(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True)


def kpis(items):
    cols = st.columns(len(items))
    for c, (val, lab) in zip(cols, items):
        c.markdown(f'<div class="card"><div class="kpi">{val}</div><div class="kpilabel">{lab}</div></div>', unsafe_allow_html=True)


def header(num, dataset, title_html, lead):
    st.markdown(f'<div style="margin-top:1.6rem"><span class="chip">{num} · {dataset}</span></div>'
                f'<h2 class="sec">{title_html}</h2><p class="lead">{lead}</p>', unsafe_allow_html=True)


def sub(t):
    st.markdown(f'<h3 class="sub">{t}</h3>', unsafe_allow_html=True)


def missing(track, err):
    st.error(f"The {track} model files could not be loaded. Run notebooks/{track}.ipynb once from the notebooks folder "
             f"(this also fixes scikit-learn version mismatches), then refresh. Details: {err}")


# =============================================================== NAV
top = st.columns([1.5, 5.2, 1.3], vertical_alignment="center")
top[0].markdown('<div class="brand">gig <b>::</b> analytics</div>', unsafe_allow_html=True)
with top[1]:
    nav = st.columns(len(PAGES))
    for c, p in zip(nav, PAGES):
        c.button(p, key=f"nav_{p}", on_click=go_to, args=(p,))
top[2].button("Team 6", key="ghost_badge", on_click=go_to, args=("About",))


# =============================================================== PAGES
def page_overview():
    left, right = st.columns([1.3, 1], vertical_alignment="center")
    with left:
        st.markdown('<div style="margin-top:1.8rem"><span class="chip">Gig worker analytics · ML capstone</span></div>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero">What a ride<br>is worth, how<br>it ends, and <em>who<br>takes it.</em></h1>', unsafe_allow_html=True)
        st.markdown('<p class="lead">Three ride-hailing datasets, three machine-learning questions that matter to gig drivers: '
                    '<b>how much a trip pays</b>, <b>whether a booking will be completed</b>, and <b>which kinds of rides</b> '
                    'make up the work. Every chart here is produced from the data and models in our notebooks.</p>', unsafe_allow_html=True)
        b = st.columns([1, 1, 1.2])
        b[0].button("Explore fares  →", key="cta_fares", on_click=go_to, args=("Fares",))
        b[1].button("See bookings", key="ghost_bookings", on_click=go_to, args=("Bookings",))
    with right:
        st.markdown(HERO_SVG, unsafe_allow_html=True)

    st.markdown('<hr class="soft"><div class="eyebrow">Three datasets · three tracks</div>', unsafe_allow_html=True)
    rm, _ = safe(load_json, "regression_meta.json")
    cm, _ = safe(load_json, "classification_meta.json")
    km, _ = safe(load_json, "clustering_meta.json")
    cards = [
        ("01 · Regression", "Uber Fares", "Predict <b>fare_amount</b> from trip distance and time.",
         f"Best: <b>{rm['best_model']}</b><br>Test R² <b>{rm['test_r2']:.3f}</b> · MAE <b>${rm['test_mae']:.2f}</b>" if rm else "Run regression.ipynb", "Fares"),
        ("02 · Classification", "Ola & Uber Bookings", "Predict <b>Booking_Status</b> (4 outcomes) from vehicle, value, hour and day.",
         f"Best: <b>{cm['best_model']}</b><br>Weighted F1 <b>{cm['test_f1_weighted']:.3f}</b> · ROC-AUC <b>{cm['test_roc_auc_ovr']:.2f}</b>" if cm else "Run classification.ipynb", "Bookings"),
        ("03 · Clustering", "NCR Ride Bookings", "Group completed rides into <b>segments</b> without labels.",
         f"Best: <b>{km['best_algorithm']}</b><br>{km['K']} segments" if km else "Run clustering.ipynb", "Segments"),
    ]
    cols = st.columns(3)
    for c, (tag, name, q, res, page) in zip(cols, cards):
        with c:
            st.markdown(f'<div class="card"><div class="eyebrow">{tag}</div><div class="track">{name}</div>'
                        f'<div class="track-meta">{q}<br><br>{res}</div></div>', unsafe_allow_html=True)
            st.button(f"Open {page.lower()}  →", key=f"ghost_open_{page}", on_click=go_to, args=(page,))
    st.markdown('<hr class="soft"><div class="eyebrow">The pipeline behind every page</div><div class="flow" style="margin-top:.8rem">'
                '<span class="step">Audit & EDA</span>→<span class="step">Clean & engineer</span>→<span class="step">Scale / encode on train only</span>'
                '→<span class="step">Train every algorithm</span>→<span class="step">GridSearchCV tuning</span>→<span class="step">Cross-validated selection</span>'
                '→<span class="step">Export → this app</span></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- FARES (regression)
@st.cache_data(show_spinner=False)
def reg_test_predictions():
    df = load_csv("uber_regression_10k.csv")
    feats = ["distance_km", "passenger_count", "pickup_hour", "is_weekend", "pickup_day_of_week", "pickup_month"]
    _, Xte, _, yte = train_test_split(df[feats], df["fare_amount"], test_size=0.2, random_state=RANDOM_STATE)
    return pd.DataFrame({"actual": yte.values, "predicted": load_model("best_regression_model.joblib").predict(Xte), "distance_km": Xte["distance_km"].values})


def page_fares():
    header("01 · Regression", "Uber Fares dataset", "What is a trip <em>worth?</em>",
           "10,000 Uber trips. Fare is predicted from trip distance (engineered with the Haversine formula from GPS points), passenger count and pickup time.")
    df = load_csv("uber_regression_10k.csv")
    meta, err = safe(load_json, "regression_meta.json")
    if err:
        missing("regression", err); return
    res = load_csv("regression_results.csv", "models")
    kpis([(f"{len(df):,}", "trips"), (f"${df.fare_amount.median():.2f}", "median fare"), (f"{meta['test_r2']:.3f}", "best test R²"),
          (f"${meta['test_mae']:.2f}", "mean abs. error"), (f"<em>{meta['best_model'].replace(' Regressor', '')}</em>", "selected model")])

    t1, t2, t3 = st.tabs(["The data", "10-model comparison", "Best model diagnostics"])
    with t1:
        c = st.columns(2)
        with c[0]:
            f = px.histogram(df[df.fare_amount <= 60], x="fare_amount", nbins=60, title="Fare distribution (≤ $60 shown)")
            f.update_traces(marker_color=ACCENT); chart(style_fig(f, legend=False))
        with c[1]:
            s = df.sample(3000, random_state=RANDOM_STATE)
            f = px.scatter(s, x="distance_km", y="fare_amount", opacity=.45, title="Fare vs distance — the dominant signal")
            f.update_traces(marker=dict(color=INK, size=5)); f.update_xaxes(range=[0, 25]); f.update_yaxes(range=[0, 80]); chart(style_fig(f, legend=False))
        c = st.columns(2)
        with c[0]:
            h = df.groupby("pickup_hour").fare_amount.mean().reset_index()
            f = px.line(h, x="pickup_hour", y="fare_amount", markers=True, title="Average fare by pickup hour")
            f.update_traces(line_color=GREEN); chart(style_fig(f, legend=False))
        with c[1]:
            corr = df.select_dtypes("number").corr()["fare_amount"].drop("fare_amount").sort_values()
            f = px.bar(x=corr.values, y=corr.index, orientation="h", title="Correlation with fare")
            f.update_traces(marker_color=[ACCENT if v == corr.max() else SAND for v in corr.values]); chart(style_fig(f, legend=False))
    with t2:
        r = res.sort_values("R2")
        f = go.Figure()
        f.add_bar(y=r["Model"], x=r["R2"], orientation="h", name="Test R²", marker_color=[ACCENT if m == meta["best_model"] else SAND for m in r["Model"]])
        f.add_bar(y=r["Model"], x=r["CV R2 (5-fold)"], orientation="h", name="5-fold CV R²", marker_color=INK, opacity=.75)
        f.update_layout(barmode="group", title="All 10 algorithms after GridSearchCV tuning"); f.update_xaxes(range=[0.6, 0.85])
        chart(style_fig(f, height=520))
        c = st.columns(2)
        with c[0]:
            r2 = res.sort_values("RMSE", ascending=False)
            f = px.bar(r2, x="RMSE", y="Model", orientation="h", title="RMSE ($, lower is better)"); f.update_traces(marker_color=SAND); chart(style_fig(f, 420, False))
        with c[1]:
            r2 = res.sort_values("MAE", ascending=False)
            f = px.bar(r2, x="MAE", y="Model", orientation="h", title="MAE ($, lower is better)"); f.update_traces(marker_color=GREEN); chart(style_fig(f, 420, False))
        table(res)
        st.markdown(f'<div class="note">Selection rule (decided before looking at the test set): repeated 5×3 cross-validation on the training data; models within 0.01 R² of the best are tied, and the tie is broken by test R². '
                    f'Result: <b>{meta["best_model"]}</b> (repeated-CV R² {meta["cv_r2"]:.3f}).</div>', unsafe_allow_html=True)
    with t3:
        pr, perr = safe(reg_test_predictions)
        if perr:
            missing("regression", perr)
        else:
            c = st.columns(2)
            with c[0]:
                f = px.scatter(pr, x="actual", y="predicted", opacity=.45, title=f"Predicted vs actual — {meta['best_model']} (test set)")
                f.update_traces(marker=dict(color=ACCENT, size=5)); lim = float(min(pr[["actual", "predicted"]].max().max(), 80))
                f.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(color=INK, dash="dash")); f.update_xaxes(range=[0, lim]); f.update_yaxes(range=[0, lim])
                chart(style_fig(f, 420, False))
            with c[1]:
                pr["residual"] = pr.actual - pr.predicted
                f = px.histogram(pr, x="residual", nbins=80, title="Residuals (actual − predicted)"); f.update_traces(marker_color=INK); f.update_xaxes(range=[-20, 20])
                chart(style_fig(f, 420, False))
            pr["band"] = pd.cut(pr.distance_km, [-0.01, 1, 2, 4, 8, 1000], labels=["<1 km", "1–2 km", "2–4 km", "4–8 km", ">8 km"])
            mb = pr.assign(err=pr.residual.abs()).groupby("band", observed=True).err.mean().reset_index()
            f = px.bar(mb, x="band", y="err", title="Mean absolute error by trip length"); f.update_traces(marker_color=TAN); chart(style_fig(f, 340, False))

    sub("Try the fare model")
    model, merr = safe(load_model, "best_regression_model.joblib")
    if merr:
        missing("regression", merr); return
    left, right = st.columns([1.2, 1])
    with left:
        c = st.columns(2)
        dist = c[0].slider("Trip distance (km)", 0.5, 30.0, 5.0, 0.5)
        pax = c[1].slider("Passengers", 1, 6, 1)
        c = st.columns(3)
        hour = c[0].slider("Pickup hour", 0, 23, 18)
        day = c[1].selectbox("Day", DAYS, index=4)
        month = c[2].selectbox("Month", MONTHS, index=5)
    row = pd.DataFrame([{"distance_km": dist, "passenger_count": pax, "pickup_hour": hour, "is_weekend": int(day in ("Saturday", "Sunday")),
                         "pickup_day_of_week": day, "pickup_month": month}])
    fare = float(model.predict(row)[0])
    with right:
        st.markdown(f'<div class="card"><div class="eyebrow">Estimated fare</div><div class="big"><em>$</em>{fare:,.2f}</div>'
                    f'<div class="kpilabel" style="margin-top:.7rem">typical error ± ${meta["test_mae"]:.2f} · {meta["best_model"]}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="note">Fares are in US dollars because the Uber Fares dataset comes from New York City trips (2009–2015).</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- BOOKINGS (classification)
@st.cache_data(show_spinner=False)
def clf_confusion():
    df = load_csv("bookings_classification_10k.csv")
    X, y = df[["Booking_Value", "hour", "Vehicle_Type", "day_of_week"]], df["Booking_Status"]
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    m = load_model("best_classification_model.joblib")
    return confusion_matrix(yte, m.predict(Xte), labels=m.classes_), list(m.classes_)


def page_bookings():
    header("02 · Classification", "Ola & Uber Bookings dataset", "Will the booking <em>happen?</em>",
           "10,000 bookings with four outcomes. Only information known at booking time is used: vehicle type, booking value, hour and weekday.")
    df = load_csv("bookings_classification_10k.csv")
    meta, err = safe(load_json, "classification_meta.json")
    if err:
        missing("classification", err); return
    res = load_csv("classification_results.csv", "models")
    lift = (meta["test_accuracy"] - meta["majority_baseline_accuracy"]) * 100
    kpis([(f"{len(df):,}", "bookings"), (f"{meta['majority_baseline_accuracy']:.1%}", "succeed (baseline)"), (f"<em>{meta['best_model']}</em>", "selected model"),
          (f"{meta['test_f1_weighted']:.3f}", "weighted F1"), (f"{meta['test_roc_auc_ovr']:.2f}", "ROC-AUC (OvR)")])

    t1, t2, t3 = st.tabs(["The data", "10-model comparison", "Best model diagnostics"])
    order = df.Booking_Status.value_counts().index.tolist()
    with t1:
        c = st.columns(2)
        with c[0]:
            vc = df.Booking_Status.value_counts()
            f = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=.55, marker=dict(colors=[GREEN, ACCENT, TAN, INK]), sort=False))
            f.update_layout(title="Booking outcomes — clearly imbalanced"); chart(style_fig(f))
        with c[1]:
            ct = (pd.crosstab(df.Vehicle_Type, df.Booking_Status, normalize="index") * 100)[order].reset_index().melt(id_vars="Vehicle_Type", var_name="Outcome", value_name="%")
            f = px.bar(ct, x="Vehicle_Type", y="%", color="Outcome", title="Outcome mix by vehicle type (%)", color_discrete_sequence=[GREEN, ACCENT, TAN, INK])
            chart(style_fig(f))
        c = st.columns(2)
        with c[0]:
            hr = df.assign(ok=df.Booking_Status.eq("Success")).groupby("hour").ok.mean().mul(100).reset_index()
            f = px.line(hr, x="hour", y="ok", markers=True, title="Success rate by hour (%)"); f.update_traces(line_color=GREEN)
            f.update_yaxes(range=[40, 80]); chart(style_fig(f, legend=False))
        with c[1]:
            f = px.box(df[df.Booking_Value <= 1500], x="Booking_Status", y="Booking_Value", category_orders={"Booking_Status": order},
                       title="Booking value by outcome (≤ 1,500 shown)", color="Booking_Status", color_discrete_sequence=[GREEN, ACCENT, TAN, INK])
            chart(style_fig(f, legend=False))
        st.markdown('<div class="note">The outcome mix is almost identical for every vehicle type, hour and fare level, so these features carry very little information about the outcome. The model results below reflect that.</div>', unsafe_allow_html=True)
    with t2:
        r = res.sort_values("F1 (weighted)")
        f = go.Figure()
        for col, colr in [("Accuracy", SAND), ("F1 (weighted)", ACCENT), ("F1 (macro)", INK)]:
            f.add_bar(y=r["Model"], x=r[col], orientation="h", name=col, marker_color=colr)
        f.add_vline(x=meta["majority_baseline_accuracy"], line_dash="dash", line_color=GREEN, annotation_text="always-Success accuracy", annotation_position="top")
        f.update_layout(barmode="group", title="All 10 algorithms (tuned) + majority-class baseline"); chart(style_fig(f, height=640))
        r2 = res.dropna(subset=["CV F1"]).sort_values("ROC-AUC (OvR)")
        f = px.bar(r2, x="ROC-AUC (OvR)", y="Model", orientation="h", title="ROC-AUC (0.5 = chance)"); f.update_traces(marker_color=TAN)
        f.add_vline(x=0.5, line_dash="dash", line_color=INK); f.update_xaxes(range=[0.4, max(0.6, r2["ROC-AUC (OvR)"].max() + .02)]); chart(style_fig(f, 460, False))
        table(res)
    with t3:
        cmx, cerr = safe(clf_confusion)
        if cerr:
            missing("classification", cerr)
        else:
            m, labels = cmx
            f = px.imshow(m, x=labels, y=labels, text_auto=True, color_continuous_scale=[[0, "#f7f2ea"], [1, ACCENT]],
                          labels=dict(x="Predicted", y="Actual", color="bookings"), title=f"Confusion matrix — {meta['best_model']} (test set)")
            chart(style_fig(f, 480, False))
        verdict = "beats" if lift > 0 else "does not beat"
        st.markdown(f'<div class="note">Honest reading: the selected model reaches {meta["test_accuracy"]:.1%} accuracy against {meta["majority_baseline_accuracy"]:.1%} for always guessing “Success” '
                    f'({lift:+.1f} points), so it <b>{verdict}</b> the baseline on accuracy; ROC-AUC {meta["test_roc_auc_ovr"]:.2f} is close to chance (0.5). '
                    f'Booking outcomes depend on driver supply, traffic and surge, which this dataset does not record.</div>', unsafe_allow_html=True)

    sub("Try the booking model")
    model, merr = safe(load_model, "best_classification_model.joblib")
    if merr:
        missing("classification", merr); return
    left, right = st.columns([1.1, 1])
    with left:
        veh = st.selectbox("Vehicle type", meta["vehicle_types"])
        c = st.columns(2)
        day = c[0].selectbox("Day of week", meta["days"], index=4)
        hour = c[1].slider("Hour", 0, 23, 18)
        val = st.slider("Booking value", int(meta["booking_value_min"]), int(meta["booking_value_max"]), int(meta["booking_value_median"]))
    proba = model.predict_proba(pd.DataFrame([{"Booking_Value": val, "hour": hour, "Vehicle_Type": veh, "day_of_week": day}]))[0]
    classes = list(model.classes_)
    idx = sorted(range(len(classes)), key=lambda i: -proba[i])
    bars = "".join(f'<div class="plabel"><span>{classes[i]}</span><span>{proba[i]:.1%}</span></div><div class="pbar"><div style="width:{proba[i] * 100:.1f}%"></div></div>' for i in idx)
    with right:
        st.markdown(f'<div class="card"><div class="eyebrow">Most likely outcome</div><div class="big" style="font-size:2.8rem"><em>{classes[idx[0]]}</em></div>'
                    f'<div class="kpilabel" style="margin:.5rem 0 1.1rem">{meta["best_model"]} · probabilities, not promises</div>{bars}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- SEGMENTS (clustering)
def page_segments():
    header("03 · Clustering", "NCR Ride Bookings dataset", "Which kinds of rides <em>make up the work?</em>",
           "10,000 completed rides grouped without labels using fare, distance, pickup / trip times and the two ratings. K-Means and Agglomerative clustering are compared.")
    df = load_csv("ncr_clustering_10k.csv")
    meta, err = safe(load_json, "clustering_meta.json")
    if err:
        missing("clustering", err); return
    res = load_csv("clustering_results.csv", "models")
    best = res.set_index("Algorithm").loc[meta["best_algorithm"]]
    kpis([(f"{len(df):,}", "rides"), (f"{meta['K']}", "segments (k)"), (f"<em>{meta['best_algorithm'].split(' (')[0]}</em>", "selected algorithm"),
          (f"{best['Silhouette']:.3f}", "silhouette"), (f"{best['Davies-Bouldin']:.2f}", "Davies-Bouldin")])

    t1, t2, t3 = st.tabs(["Segments", "Choosing k & comparing algorithms", "The data"])
    feats = meta["features"]
    prof = pd.DataFrame(meta["profiles"]).set_index("Cluster")
    with t1:
        pca = load_csv("clustering_pca_sample.csv", "models")
        algo = st.radio("Colour by", ["kmeans", "agglomerative"], horizontal=True, format_func=lambda a: "K-Means" if a == "kmeans" else f"Agglomerative ({meta['best_linkage']})")
        f = px.scatter(pca.assign(Segment=pca[algo].astype(str)), x="PC1", y="PC2", color="Segment", opacity=.6,
                       title=f"PCA 2-D view of the segments ({meta['pca_variance_2d']:.0%} of variance shown)", category_orders={"Segment": [str(i) for i in range(meta["K"])]}, color_discrete_sequence=[ACCENT, INK, GREEN, "#d9a441", TAN, "#6d5a8a", "#8aa0b4", "#b04a6a", "#5c8d89", "#9c9184"])
        f.update_traces(marker=dict(size=5)); chart(style_fig(f, 480))
        z = (prof[feats] - df[feats].mean()) / df[feats].std()
        f = px.imshow(z, x=feats, y=[f"Segment {i}" for i in z.index], text_auto=".2f", color_continuous_scale=[[0, GREEN], [.5, "#f7f2ea"], [1, ACCENT]], zmin=-2, zmax=2,
                      aspect="auto", title="Segment profiles — how each segment differs from the average ride (std units)")
        chart(style_fig(f, 140 + 46 * meta["K"], False))
        show = prof.rename(columns={"Rides": "Rides"}).round(2).reset_index()
        table(show)
    with t2:
        ks = load_csv("clustering_k_search.csv", "models")
        c = st.columns(2)
        with c[0]:
            f = px.line(ks, x="k", y="Inertia", markers=True, title="Elbow curve (K-Means inertia)"); f.update_traces(line_color=INK)
            f.add_vline(x=meta["K"], line_dash="dash", line_color=ACCENT, annotation_text=f"chosen k = {meta['K']}"); chart(style_fig(f, legend=False))
        with c[1]:
            f = px.line(ks, x="k", y="Silhouette", markers=True, title="Silhouette vs k (higher = better)"); f.update_traces(line_color=GREEN)
            chart(style_fig(f, legend=False))
        lk = load_csv("clustering_linkage.csv", "models")
        f = px.bar(lk, x="Linkage", y="Calinski-Harabasz", text="Smallest cluster", title="Agglomerative linkage comparison (bar label = size of smallest cluster)")
        f.update_traces(marker_color=[ACCENT if l == meta["best_linkage"] else SAND for l in lk["Linkage"]]); chart(style_fig(f, legend=False))
        table(res)
        st.markdown(f'<div class="note">All three mandatory metrics favour <b>{meta["best_algorithm"]}</b>. Silhouette values are modest, meaning the rides form a continuum rather than sharply separated groups, '
                    f'so the segments are best read as useful profiles. Agreement between the two algorithms (ARI) = {meta["ari_kmeans_vs_agglomerative"]:.2f}.</div>', unsafe_allow_html=True)
    with t3:
        c = st.columns(3)
        for i, col in enumerate(feats):
            with c[i % 3]:
                f = px.histogram(df, x=col, nbins=40, title=col); f.update_traces(marker_color=[ACCENT, INK, GREEN, TAN, "#6d5a8a", "#5c8d89"][i]); chart(style_fig(f, 280, False))

    sub("Which segment does a ride belong to?")
    model, merr = safe(load_model, "clustering_kmeans.joblib")
    if merr:
        missing("clustering", merr); return
    rng = meta["feature_ranges"]
    cols = st.columns(3)
    vals = {}
    for i, col in enumerate(feats):
        lo, med, hi = rng[col]
        step = 0.1 if hi <= 50 else 1.0
        vals[col] = cols[i % 3].slider(col, float(lo), float(hi), float(round(med, 1)), step)
    seg = int(model.predict(pd.DataFrame([vals]))[0])
    zrow = ((prof.loc[seg, feats] - df[feats].mean()) / df[feats].std())
    top = zrow.abs().sort_values(ascending=False).index[:2]
    desc = ", ".join(f"{t} {'above' if zrow[t] > 0 else 'below'} average" for t in top)
    st.markdown(f'<div class="card"><div class="eyebrow">Assigned segment</div><div class="big" style="font-size:2.8rem"><em>Segment {seg}</em></div>'
                f'<div class="track-meta" style="margin-top:.5rem">{int(prof.loc[seg, "Rides"]):,} similar rides · typical profile: <b>{desc}</b></div></div>', unsafe_allow_html=True)


def page_about():
    header("04 · About", "Team 6 · 23CSE301", "The project, <em>honestly</em>", "How the data, models and this dashboard fit together.")
    c = st.columns(2)
    c[0].markdown('<div class="card"><div class="eyebrow">Team</div><div class="track-meta" style="margin-top:.6rem">'
                  '<b>Harsha Vardhan</b> — CB.SC.U4CSE24719<br><b>M Pranav</b> — CB.SC.U4CSE24730<br><b>S Rahul Kumar</b> — CB.SC.U4CSE24766<br><br>'
                  '23CSE301 Machine Learning · Capstone Review 2</div></div>', unsafe_allow_html=True)
    c[1].markdown('<div class="card"><div class="eyebrow">Datasets (Kaggle, 10,000-row cleaned samples)</div><div class="track-meta" style="margin-top:.6rem">'
                  '<b>Uber Fares</b> → regression (fare_amount)<br><b>Ola &amp; Uber Bookings</b> → classification (Booking_Status)<br>'
                  '<b>NCR Ride Bookings</b> → clustering (ride segments)<br><br>Each dataset is used only on its own page and by its own model.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-top:1rem"><div class="eyebrow">Method</div><div class="track-meta" style="margin-top:.6rem">'
                'scikit-learn pipelines (scaler + one-hot encoder fitted on training data only) · 80/20 split with random_state = 42 · '
                '10 regression and 10 classification algorithms, one notebook cell each · GridSearchCV tuning · repeated cross-validation for model selection · '
                'K-Means (elbow curve) and Agglomerative clustering (dendrogram, linkage comparison) with Silhouette, Davies-Bouldin and Calinski-Harabasz scores, PCA and t-SNE plots.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="note">AI assistance: parts of the notebooks and this dashboard were drafted with Claude (Anthropic). The team reviewed and ran everything and is responsible for the results.</div>', unsafe_allow_html=True)


{"Overview": page_overview, "Fares": page_fares, "Bookings": page_bookings, "Segments": page_segments, "About": page_about}[PAGE]()
st.markdown('<hr class="soft"><div class="eyebrow" style="text-align:center">gig :: analytics · Team 6 · 23CSE301 Machine Learning Capstone</div>', unsafe_allow_html=True)

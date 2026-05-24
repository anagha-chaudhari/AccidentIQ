import streamlit as st
import numpy as np
import joblib
import math

st.set_page_config(
    page_title="AccidentIQ",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&family=Barlow:wght@700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container {
    background: #f5f6f8 !important;
    color: #1a2340;
    font-family: 'Inter', 'DM Sans', sans-serif;
}

section[data-testid="stMain"] > div { padding-top: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header,
[data-testid="stSidebar"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── TOP BAR — Indian tricolour stripe + navy ── */
.aiq-topbar {
    background: #0a2252;
    border-bottom: 4px solid #FF6200;
    padding: 0 36px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 64px;
    position: sticky;
    top: 0;
    z-index: 200;
}

.aiq-logo-wrap { display: flex; align-items: center; gap: 13px; }

.aiq-logo-icon {
    width: 38px; height: 38px;
    background: #FF6200;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

.aiq-wordmark {
    font-family: 'Inter', sans-serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.02em;
}

.aiq-wordmark em {
    color: #FF6200;
    font-style: normal;
}

.aiq-subtitle-bar {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.45);
    font-weight: 400;
    letter-spacing: 0.01em;
    margin-top: 1px;
}

.aiq-topbar-tags { display: flex; gap: 8px; align-items: center; }

.aiq-tag {
    font-family: 'Inter', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    padding: 4px 11px;
    border-radius: 4px;
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
}

.aiq-tag.accent {
    border-color: rgba(255,98,0,0.5);
    color: #FF6200;
    background: rgba(255,98,0,0.08);
}

/* ── LAYOUT ── */
.aiq-layout { display: grid; grid-template-columns: 1fr 400px; min-height: calc(100vh - 68px); gap: 0; }

.panel-left {
    padding: 28px 32px 40px;
    background: #ffffff;
    border-right: 1px solid #e2e6ef;
    overflow-y: auto;
}

.panel-right {
    padding: 28px 28px 40px;
    background: #f5f6f8;
    overflow-y: auto;
}

/* ── SECTION HEADING ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
    margin-top: 4px;
}

.sec-head-bar {
    width: 4px; height: 18px;
    background: #0a2252;
    border-radius: 2px;
    flex-shrink: 0;
}

.sec-head-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: #0a2252;
    text-transform: uppercase;
}

.sec-head-line { flex: 1; height: 1px; background: #e2e6ef; }

/* ── FORM CONTROLS ── */
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.73rem !important;
    font-weight: 600 !important;
    color: #6b7a99 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-bottom: 5px !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #f8f9fc !important;
    border: 1.5px solid #dde1ed !important;
    border-radius: 7px !important;
    color: #1a2340 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: border-color 0.18s !important;
    box-shadow: none !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: #0a2252 !important;
}

[data-baseweb="popover"], [data-baseweb="menu"] {
    background: #ffffff !important;
    border: 1.5px solid #dde1ed !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(10,34,82,0.12) !important;
}

[data-baseweb="menu"] li {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    color: #1a2340 !important;
    font-weight: 400 !important;
}

[data-baseweb="menu"] li:hover { background: #f0f3fa !important; }

[data-testid="stSlider"] > div > div > div > div { background: #0a2252 !important; }
[data-testid="stSlider"] > div > div > div { background: #dde1ed !important; }

/* ── BUTTON ── */
[data-testid="stButton"] button {
    background: #0a2252 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 11px 28px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.18s, box-shadow 0.18s, transform 0.1s !important;
    box-shadow: 0 3px 12px rgba(10,34,82,0.22) !important;
}

[data-testid="stButton"] button:hover {
    background: #0d2d6b !important;
    box-shadow: 0 5px 18px rgba(10,34,82,0.3) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stButton"] button:active { transform: scale(0.99) !important; }

/* ── DIVIDER ── */
.div { height: 1px; background: #e2e6ef; margin: 22px 0; }

/* ── RIGHT PANEL CARDS ── */

/* Empty state */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 64px 24px; gap: 14px;
}

.empty-icon {
    width: 60px; height: 60px;
    border-radius: 14px;
    background: #eef0f7;
    border: 1.5px solid #dde1ed;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 6px;
}

.empty-title { font-size: 1rem; font-weight: 600; color: #8892aa; }
.empty-sub { font-size: 0.83rem; color: #adb5c8; line-height: 1.65; max-width: 220px; }

/* Risk big card */
.risk-big-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1.5px solid #e2e6ef;
    padding: 24px 22px 20px;
    margin-bottom: 14px;
    border-top-width: 4px;
}

.risk-big-card.HIGH   { border-top-color: #c0000a; }
.risk-big-card.MEDIUM { border-top-color: #FF6200; }
.risk-big-card.LOW    { border-top-color: #138808; }

.risk-meta-row {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 14px;
}

.risk-meta-label {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.12em; color: #9aa3ba;
    text-transform: uppercase;
}

.risk-badge {
    font-size: 0.66rem; font-weight: 700;
    padding: 4px 12px; border-radius: 100px;
    letter-spacing: 0.09em; text-transform: uppercase;
}

.risk-badge.HIGH   { background: #fff0f0; color: #c0000a; border: 1.5px solid #f5c0c0; }
.risk-badge.MEDIUM { background: #fff4ec; color: #b84500; border: 1.5px solid #fcd5b0; }
.risk-badge.LOW    { background: #edf7ee; color: #138808; border: 1.5px solid #b5d9b6; }

/* Big number */
.risk-number {
    font-family: 'Barlow', 'Inter', sans-serif;
    font-size: 5.8rem;
    font-weight: 900;
    line-height: 0.88;
    letter-spacing: -0.03em;
}

.risk-number sup {
    font-size: 2rem; font-weight: 800;
    vertical-align: super; letter-spacing: 0;
}

.risk-number.HIGH   { color: #c0000a; }
.risk-number.MEDIUM { color: #FF6200; }
.risk-number.LOW    { color: #138808; }

.risk-baseline {
    font-size: 0.8rem; color: #8892aa;
    margin-top: 10px; font-weight: 400; line-height: 1.5;
}

.risk-baseline strong { color: #3d4a68; font-weight: 600; }

/* Progress bar */
.prog-wrap { margin-top: 16px; }

.prog-labels {
    display: flex; justify-content: space-between;
    font-size: 0.6rem; color: #adb5c8;
    font-weight: 600; letter-spacing: 0.07em;
    margin-bottom: 6px; text-transform: uppercase;
}

.prog-track {
    height: 7px; background: #eef0f7;
    border-radius: 100px; overflow: hidden;
}

.prog-fill { height: 100%; border-radius: 100px; transition: width 0.7s ease; }
.prog-fill.HIGH   { background: linear-gradient(90deg, #FF6200 0%, #c0000a 100%); }
.prog-fill.MEDIUM { background: linear-gradient(90deg, #138808 0%, #FF6200 100%); }
.prog-fill.LOW    { background: linear-gradient(90deg, #2e7d32 0%, #4caf50 100%); }

/* Explain box */
.explain-box {
    background: #f8f9fc;
    border-left: 3px solid #0a2252;
    border-radius: 0 8px 8px 0;
    padding: 13px 15px;
    margin-bottom: 18px;
}

.explain-box p { font-size: 0.83rem; color: #4a5578; line-height: 1.7; }

/* Stat chips */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin-bottom: 16px; }

.stat-chip {
    background: #ffffff;
    border: 1.5px solid #e2e6ef;
    border-radius: 8px;
    padding: 14px 16px;
}

.stat-chip-val {
    font-family: 'Barlow', 'Inter', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    color: #0a2252; line-height: 1; margin-bottom: 4px;
}

.stat-chip-lbl {
    font-size: 0.68rem; font-weight: 600;
    color: #9aa3ba; letter-spacing: 0.07em; text-transform: uppercase;
}

.stat-note {
    background: #f8f9fc;
    border-left: 3px solid #dde1ed;
    border-radius: 0 8px 8px 0;
    padding: 11px 14px;
    font-size: 0.79rem; color: #6b7a99; line-height: 1.65; margin-bottom: 18px;
}

/* SHAP bars */
.shap-legend { display: flex; gap: 18px; margin-bottom: 14px; }

.shap-legend-item {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.73rem; color: #6b7a99; font-weight: 500;
}

.shap-legend-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
.shap-legend-dot.up { background: #FF6200; }
.shap-legend-dot.dn { background: #0a2252; }

.shap-row {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 9px; padding: 8px 12px;
    background: #ffffff; border-radius: 7px;
    border: 1px solid #eef0f7;
}

.shap-name {
    font-size: 0.73rem; font-weight: 500;
    color: #3d4a68; width: 120px; flex-shrink: 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

.shap-track { flex: 1; height: 6px; background: #eef0f7; border-radius: 100px; overflow: hidden; }
.shap-bar { height: 100%; border-radius: 100px; }
.shap-bar.up { background: #FF6200; }
.shap-bar.dn { background: #0a2252; }

.shap-val { font-family: 'Barlow', 'Inter', sans-serif; font-size: 0.78rem; font-weight: 700; min-width: 46px; text-align: right; flex-shrink: 0; }
.shap-val.up { color: #b84500; }
.shap-val.dn { color: #0a2252; }

/* Footer */
.aiq-footer {
    font-size: 0.65rem; color: #adb5c8; text-align: center;
    padding: 20px 0 30px; letter-spacing: 0.08em; text-transform: uppercase;
    border-top: 1px solid #e2e6ef; margin-top: 16px;
    background: #ffffff;
}
</style>
""", unsafe_allow_html=True)

from pathlib import Path

@st.cache_resource
def load_model():

    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR.parent / "models" / "xgb_fatal_predictor.pkl"

    return joblib.load(MODEL_PATH)

model = load_model()

def time_to_sincos(hour, minute):
    m = hour * 60 + minute
    return math.sin(2 * math.pi * m / 1440), math.cos(2 * math.pi * m / 1440)

def day_to_sincos(day):
    return math.sin(2 * math.pi * day / 7), math.cos(2 * math.pi * day / 7)

def get_risk(prob):
    if prob >= 0.40: return "HIGH"
    if prob >= 0.20: return "MEDIUM"
    return "LOW"

def sec(label):
    st.markdown(f"""
    <div class="sec-head">
        <div class="sec-head-bar"></div>
        <div class="sec-head-label">{label}</div>
        <div class="sec-head-line"></div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="aiq-topbar">
    <div class="aiq-logo-wrap">
        <div class="aiq-logo-icon">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                <path d="M11 3L20 18H2L11 3Z" fill="white" fill-opacity="0.95"/>
                <rect x="10" y="10" width="2" height="4.5" rx="1" fill="#FF6200"/>
                <circle cx="11" cy="16" r="1.1" fill="#FF6200"/>
            </svg>
        </div>
        <div>
            <div class="aiq-wordmark">Accident<em>IQ</em></div>
            <div class="aiq-subtitle-bar">Road accident fatal risk predictor · UK STATS19</div>
        </div>
    </div>
    <div class="aiq-topbar-tags">
        <div class="aiq-tag">106K Records</div>
        <div class="aiq-tag">Dept for Transport</div>
        <div class="aiq-tag accent">XGBoost</div>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    sec("Accident Conditions")

    c1, c2 = st.columns(2)
    with c1:
        speed_limit = st.selectbox("Speed Limit (mph)", [20,30,40,50,60,70], index=2)
        road_type = st.selectbox("Road Type",
            options=[1,2,3,6,7,9],
            format_func=lambda x: {1:"Roundabout",2:"One-way street",3:"Dual carriageway",
                                    6:"Single carriageway",7:"Slip road",9:"Unknown"}[x],
            index=3)
        urban_or_rural = st.selectbox("Area Type",
            options=[1,2,3],
            format_func=lambda x: {1:"Urban",2:"Rural",3:"Unallocated"}[x])
        light_conditions = st.selectbox("Light Conditions",
            options=[1,4,5,6,7],
            format_func=lambda x: {1:"Daylight",4:"Dark — lights on",5:"Dark — lights off",
                                    6:"Dark — no lighting",7:"Dark — unknown"}[x])
        weather = st.selectbox("Weather",
            options=[1,2,3,4,5,6,7,8],
            format_func=lambda x: {1:"Fine, no wind",2:"Raining",3:"Snowing",
                                    4:"Fine + high wind",5:"Raining + high wind",
                                    6:"Fog or mist",7:"Other",8:"Unknown"}[x])

    with c2:
        road_surface = st.selectbox("Road Surface",
            options=[1,2,3,4,5,9],
            format_func=lambda x: {1:"Dry",2:"Wet / Damp",3:"Snow",
                                    4:"Frost / Ice",5:"Flood",9:"Unknown"}[x])
        junction_detail = st.selectbox("Junction Type",
            options=[0,13,16,17,18,19],
            format_func=lambda x: {0:"Not at junction",13:"T-junction",16:"Crossroads",
                                    17:"Multiple junction",18:"Roundabout",19:"Private drive"}[x])
        junction_control = st.selectbox("Junction Control",
            options=[-1,1,2,3,4],
            format_func=lambda x: {-1:"Not at junction",1:"Authorised person",
                                    2:"Auto signal",3:"Stop sign",
                                    4:"Give way / Uncontrolled"}[x])
        num_vehicles   = st.slider("Number of Vehicles",   1, 10, 2)
        num_casualties = st.slider("Number of Casualties", 1, 10, 1)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    sec("Vehicle & Time")

    c3, c4 = st.columns(2)
    with c3:
        vehicle_type = st.selectbox("Primary Vehicle Type",
            options=[1,2,3,4,5,8,9,10,11,16,17,19,20,21,22,23],
            format_func=lambda x: {
                1:"Bicycle",2:"Motorcycle <50cc",3:"Motorcycle 50–125cc",
                4:"Motorcycle 125–500cc",5:"Motorcycle 500cc+",8:"Taxi",
                9:"Car",10:"Minibus",11:"Bus / Coach",16:"Horse",
                17:"Agricultural",19:"Van",20:"HGV",
                21:"Motorcycle unknown",22:"Electric motorcycle",23:"E-scooter"}[x],
            index=6)
        any_skidding = st.selectbox("Skidding / Overturning",
            options=[0,1,2],
            format_func=lambda x: {0:"No",1:"Yes — skidded",2:"Yes — overturned"}[x])
        carriageway_hazards = st.selectbox("Carriageway Hazard",
            options=[0,1,2,7],
            format_func=lambda x: {0:"None",1:"Object in road",
                                    2:"Other vehicle",7:"Animal"}[x])

    with c4:
        avg_vehicle_age = st.slider("Avg Vehicle Age (years)", 0, 30, 8)
        hour = st.slider("Hour of Day", 0, 23, 14)
        minute = st.selectbox("Minute", [0,15,30,45], index=0)
        day_of_week = st.selectbox("Day of Week",
            options=[1,2,3,4,5,6,7],
            format_func=lambda x: {1:"Monday",2:"Tuesday",3:"Wednesday",
                                    4:"Thursday",5:"Friday",6:"Saturday",7:"Sunday"}[x],
            index=4)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Analyse Risk →")

with right:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    sec("Risk Assessment")

    if predict_btn:
        t_sin, t_cos = time_to_sincos(hour, minute)
        d_sin, d_cos = day_to_sincos(day_of_week)

        features = np.array([[
            road_type, speed_limit, urban_or_rural,
            junction_detail, junction_control,
            light_conditions, weather, road_surface,
            carriageway_hazards, num_vehicles, num_casualties,
            vehicle_type, any_skidding, avg_vehicle_age,
            t_sin, t_cos, d_sin, d_cos
        ]])

        prob = model.predict_proba(features)[0][1]
        risk = get_risk(prob)
        prob_pct = round(prob * 100, 1)
        meter_w = min(round(prob * 100), 100)
        multiplier = round(prob * 100 / 5.8, 1)

        risk_labels = {"HIGH": "High Risk", "MEDIUM": "Medium Risk", "LOW": "Low Risk"}

        st.markdown(f"""
        <div class="risk-big-card {risk}">
            <div class="risk-meta-row">
                <div class="risk-meta-label">Fatal Risk Probability</div>
                <div class="risk-badge {risk}">{risk_labels[risk]}</div>
            </div>
            <div class="risk-number {risk}">{prob_pct}<sup>%</sup></div>
            <div class="risk-baseline">
                Population baseline: <strong>5.8%</strong> &nbsp;·&nbsp;
                This scenario is <strong>{multiplier}× higher</strong> than average
            </div>
            <div class="prog-wrap">
                <div class="prog-labels">
                    <span>0%</span><span>Low</span><span>Medium</span><span>High</span><span>100%</span>
                </div>
                <div class="prog-track">
                    <div class="prog-fill {risk}" style="width:{meter_w}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        explanations = {
            "HIGH": (
                "The model identifies this combination as genuinely high-risk. "
                "Across 106,000 historical STATS19 accidents, similar conditions have produced "
                "fatal outcomes at a significantly elevated rate. "
                "This is not a guarantee — it is a signal that warrants serious attention."
            ),
            "MEDIUM": (
                "Mixed signals — some conditions raise risk, others reduce it. "
                "The model carries uncertainty here, which is itself informative. "
                "You are not in the clear, but not in the most dangerous category either. "
                "Review the SHAP breakdown below to see what is driving the concern."
            ),
            "LOW": (
                "These conditions closely resemble non-fatal accidents in the historical record. "
                "The model is reasonably confident this is a lower-risk scenario. "
                "Road conditions can change rapidly — treat this as guidance, not a guarantee."
            )
        }

        st.markdown(f"""
        <div class="explain-box"><p>{explanations[risk]}</p></div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='div'></div>", unsafe_allow_html=True)
        sec("Model Performance")

        st.markdown("""
        <div class="stat-grid">
            <div class="stat-chip">
                <div class="stat-chip-val">0.800</div>
                <div class="stat-chip-lbl">AUC – ROC</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-val">53%</div>
                <div class="stat-chip-lbl">Fatal Recall</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-val">106K</div>
                <div class="stat-chip-lbl">Training Records</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-val">XGBoost</div>
                <div class="stat-chip-lbl">Algorithm</div>
            </div>
        </div>
        <div class="stat-note">
            AUC 0.800 means the model ranks a fatal accident above a non-fatal one 80% of the time
            (random baseline = 0.500). Fatal recall of 53% means roughly half of all real fatalities
            are detected — versus zero by a naïve baseline.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='div'></div>", unsafe_allow_html=True)
        sec("What Drives Fatal Risk")

        st.markdown("""
        <div class="shap-legend">
            <div class="shap-legend-item"><div class="shap-legend-dot up"></div> Raises risk</div>
            <div class="shap-legend-item"><div class="shap-legend-dot dn"></div> Lowers risk</div>
        </div>
        """, unsafe_allow_html=True)

        top_features = [
            ("speed_limit", 0.530, True),
            ("no. of vehicles", 0.450, True),
            ("any_skidding", 0.436, True),
            ("time_cos", 0.290, True),
            ("time_sin", 0.277, True),
            ("urban_or_rural", 0.256, False),
            ("avg_vehicle_age", 0.204, True),
            ("road_type", 0.207, False),
        ]

        max_val = max(f[1] for f in top_features)
        for name, val, positive in top_features:
            cls  = "up" if positive else "dn"
            sign = "+" if positive else "−"
            w    = int((val / max_val) * 100)
            st.markdown(f"""
            <div class="shap-row">
                <div class="shap-name">{name}</div>
                <div class="shap-track">
                    <div class="shap-bar {cls}" style="width:{w}%"></div>
                </div>
                <div class="shap-val {cls}">{sign}{val:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">
                <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
                    <path d="M15 5L27 24H3L15 5Z" stroke="#c0c8df" stroke-width="1.8" stroke-linejoin="round"/>
                    <rect x="14" y="12" width="2" height="6" rx="1" fill="#c0c8df"/>
                    <circle cx="15" cy="21" r="1.2" fill="#c0c8df"/>
                </svg>
            </div>
            <div class="empty-title">No analysis run yet</div>
            <div class="empty-sub">Set road and vehicle conditions on the left, then click <strong>Analyse Risk →</strong></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="aiq-footer">
    AccidentIQ · UK STATS19 Data · Department for Transport · Model: XGBoost · Not for operational use
</div>
""", unsafe_allow_html=True)
"""Control-room dashboard (Streamlit).

Run:  streamlit run app/app.py
Upload a crowd photo for a zone; the dashboard updates with the count, density,
risk level and a heatmap. Each upload is one snapshot, so zone cards, trend and
alert log build up as photos come in.
"""
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.heatmap import overlay_heatmap  # noqa: E402
from src.predict import backend_name, ensure_model, predict  # noqa: E402
from src.risk import LEVEL_COLORS, LEVELS, density_from_count, level_for_density, load_thresholds  # noqa: E402

# Placeholder areas in square metres. Measure the real visible ground area per camera.
ZONES = {"Ramkund": 200.0, "Panchvati": 300.0, "Trimbakeshwar": 250.0}
IMAGE_TYPES = ["jpg", "jpeg", "png"]
MAX_RECORDS = 30
VIEW_MODES = ["Heatmap overlay", "Heat only", "Original"]

st.set_page_config(page_title="Simhastha Suraksha", page_icon="🔱", layout="wide")

CSS = """
<style>
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden;height:0}
.block-container {padding-top:1.6rem;max-width:1400px}
.hero {display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:1.1rem}
.hero h1 {margin:0;font-size:2.3rem;font-weight:800;letter-spacing:-.5px;
  background:linear-gradient(90deg,#FFB25B,#FF5E3A);-webkit-background-clip:text;color:transparent}
.hero p {margin:.2rem 0 0;color:#8E99B3;font-size:.92rem}
.pill {padding:5px 14px;border-radius:99px;font-size:.78rem;font-weight:600;border:1px solid}
.pill.on {color:#3DDC97;border-color:#3DDC9755;background:#3DDC9714}
.pill.off {color:#FFB25B;border-color:#FFB25B55;background:#FFB25B14}
.kpi {background:linear-gradient(160deg,#161E33,#10162A);border:1px solid #232D4A;border-radius:14px;
  padding:16px 18px;height:100%}
.kpi .l {color:#8E99B3;font-size:.74rem;text-transform:uppercase;letter-spacing:1.2px}
.kpi .v {font-size:2rem;font-weight:800;line-height:1.15;margin-top:4px}
.kpi .s {color:#8E99B3;font-size:.8rem}
.zone {background:#131A2B;border:1px solid #232D4A;border-left:5px solid var(--c);border-radius:12px;
  padding:14px 16px;margin-bottom:.6rem}
.zone .n {font-weight:700;font-size:1.02rem}
.zone .b {float:right;background:var(--c);color:#0B0F1A;padding:2px 10px;border-radius:6px;
  font-weight:700;font-size:.76rem}
.zone .m {color:#8E99B3;font-size:.84rem;margin-top:6px}
.zone .m b {color:#E8ECF4;font-size:1.05rem}
.big {border-radius:16px;padding:22px;border:1px solid var(--c);background:linear-gradient(160deg,var(--c)22,#10162A 70%)}
.big .lv {font-size:2.6rem;font-weight:800;color:var(--c);line-height:1}
.big .row {display:flex;justify-content:space-between;margin-top:14px;color:#8E99B3;font-size:.85rem}
.big .row b {color:#E8ECF4;font-size:1.25rem;display:block}
.meter {height:10px;border-radius:99px;margin-top:16px;position:relative;
  background:linear-gradient(90deg,#1D9E75 0%,#FAC775 33%,#EF9F27 62%,#E24B4A 100%)}
.meter i {position:absolute;top:-5px;width:4px;height:20px;border-radius:3px;background:#fff;
  box-shadow:0 0 8px #000}
.empty {text-align:center;padding:70px 20px;border:1px dashed #2C3858;border-radius:16px;color:#8E99B3}
.empty h3 {color:#E8ECF4;margin-bottom:.3rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _init_model():
    # Not st.cache_resource: a failed first attempt (e.g. secret not set yet) must be retried.
    if not st.session_state.get("model_ok"):
        st.session_state["model_ok"] = ensure_model()
    return st.session_state["model_ok"]


def _shrink(img, max_w=900):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)


def analyse(file, zone, area, thresholds):
    img = np.asarray(Image.open(file).convert("RGB"))
    count, dens_map = predict(img)
    density = density_from_count(count, area)
    level = level_for_density(density, thresholds)
    img_s = _shrink(img)
    dens_s = dens_map
    return {
        "zone": zone, "name": file.name, "ts": datetime.now(), "area": area,
        "count": float(count), "density": density, "level": level,
        "original": img_s,
        "overlay": overlay_heatmap(img_s, dens_s, colormap=cv2.COLORMAP_INFERNO, alpha=0.6),
        "heat": overlay_heatmap(img_s, dens_s, colormap=cv2.COLORMAP_INFERNO, alpha=1.0, min_fraction=0.0),
    }


def kpi(label, value, sub="", color="#E8ECF4"):
    return ('<div class="kpi"><div class="l">%s</div><div class="v" style="color:%s">%s</div>'
            '<div class="s">%s</div></div>' % (label, color, value, sub))


_init_model()
thresholds = load_thresholds()
st.session_state.setdefault("records", [])
st.session_state.setdefault("alerts", [])
st.session_state.setdefault("uploader_n", 0)
records = st.session_state["records"]

# ---- sidebar: the only input is a photo ---------------------------------
with st.sidebar:
    st.markdown("### 📷 Add a camera photo")
    zone = st.selectbox("Zone", list(ZONES))
    area = st.number_input("Visible ground area (m²)", min_value=1.0, value=ZONES[zone], step=10.0,
                           key="area_" + zone, help="Measure the ground area this camera actually sees.")
    files = st.file_uploader("Upload one or more photos", type=IMAGE_TYPES, accept_multiple_files=True,
                             key="up_%d" % st.session_state["uploader_n"])
    st.markdown("---")
    st.markdown("**Risk thresholds** (people per m²)")
    st.markdown("Medium ≥ %.1f · High ≥ %.1f · Critical ≥ %.1f" %
                (thresholds["medium"], thresholds["high"], thresholds["critical"]))
    st.caption("Placeholders. Edit config/risk_thresholds.yaml.")
    if records and st.button("Clear dashboard", use_container_width=True):
        st.session_state["records"], st.session_state["alerts"] = [], []
        st.rerun()

if files:
    with st.spinner("Analysing %d photo(s)..." % len(files)):
        for f in files:
            rec = analyse(f, zone, area, thresholds)
            prev = next((r for r in reversed(records) if r["zone"] == zone), None)
            if prev and prev["level"] != rec["level"]:
                st.session_state["alerts"].append(
                    {"Time": rec["ts"].strftime("%H:%M:%S"), "Zone": zone,
                     "Change": "%s → %s" % (prev["level"], rec["level"]), "People": round(rec["count"])})
            records.append(rec)
    del records[:-MAX_RECORDS]
    st.session_state["focus"] = zone
    st.session_state["uploader_n"] += 1
    st.rerun()

latest = {}
for r in records:
    latest[r["zone"]] = r

# ---- header ------------------------------------------------------------
live = backend_name() == "onnx"
st.markdown(
    '<div class="hero"><div><h1>सिंहस्थ Suraksha</h1>'
    '<p>Crowd density and risk alerts for the control room. Decision-support aid only; '
    'it does not replace administrative or police judgement.</p></div>'
    '<span class="pill %s">%s</span></div>' %
    ("on" if live else "off", "● CSRNet model live" if live else "● Demo mode: simulated counts"),
    unsafe_allow_html=True)

if not live:
    st.warning("No trained model found, so counts are simulated. Set SS_MODEL_URL (or add models/csrnet.onnx) "
               "and reboot the app for real predictions.")

# ---- KPIs --------------------------------------------------------------
worst = max(latest.values(), key=lambda r: LEVELS.index(r["level"]))["level"] if latest else "-"
total = sum(r["count"] for r in latest.values())
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("People in view", "{:,}".format(round(total)), "latest photo per zone"), unsafe_allow_html=True)
c2.markdown(kpi("Zones reporting", "%d / %d" % (len(latest), len(ZONES)), "upload a photo to add one"),
            unsafe_allow_html=True)
c3.markdown(kpi("Highest risk", worst, "across zones", LEVEL_COLORS.get(worst, "#E8ECF4")), unsafe_allow_html=True)
c4.markdown(kpi("Alerts", len(st.session_state["alerts"]), "level changes this session"), unsafe_allow_html=True)
st.write("")

if not records:
    st.markdown('<div class="empty"><h3>No photos yet</h3>Pick a zone in the sidebar and upload a crowd photo. '
                'The count, risk level and heatmap appear here instantly.</div>', unsafe_allow_html=True)
    st.stop()

# ---- focus zone: heatmap + risk card -------------------------------------
focus = st.session_state.get("focus", zone)
if focus not in latest:
    focus = list(latest)[-1]
names = list(latest)
focus = st.radio("Zone", names, index=names.index(focus), horizontal=True, label_visibility="collapsed")
cur = latest[focus]
col = LEVEL_COLORS[cur["level"]]

left, right = st.columns([2, 1])
with left:
    mode = st.radio("View", VIEW_MODES, horizontal=True, label_visibility="collapsed")
    key = {"Heatmap overlay": "overlay", "Heat only": "heat", "Original": "original"}[mode]
    st.image(cur[key], use_container_width=True,
             caption="%s · %s · heat is relative to the densest spot in this photo" %
                     (focus, cur["ts"].strftime("%H:%M:%S")))
with right:
    pos = min(cur["density"] / (thresholds["critical"] * 1.25), 1.0) * 100
    st.markdown(
        '<div class="big" style="--c:%s"><div class="l" style="color:#8E99B3;font-size:.74rem;'
        'letter-spacing:1.2px">RISK LEVEL · %s</div><div class="lv">%s</div>'
        '<div class="meter"><i style="left:calc(%.1f%% - 2px)"></i></div>'
        '<div class="row"><div><b>%s</b>people</div><div><b>%.2f</b>per m²</div>'
        '<div><b>%d m²</b>area</div></div></div>' %
        (col, focus.upper(), cur["level"], pos, "{:,}".format(round(cur["count"])), cur["density"], cur["area"]),
        unsafe_allow_html=True)
    st.write("")
    st.markdown("**Zones**")
    for z in ZONES:
        r = latest.get(z)
        if r:
            st.markdown('<div class="zone" style="--c:%s"><span class="b">%s</span><div class="n">%s</div>'
                        '<div class="m"><b>%s</b> people · <b>%.2f</b>/m²</div></div>' %
                        (LEVEL_COLORS[r["level"]], r["level"], z, "{:,}".format(round(r["count"])), r["density"]),
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="zone" style="--c:#2C3858"><div class="n">%s</div>'
                        '<div class="m">No photo yet</div></div>' % z, unsafe_allow_html=True)

# ---- trend + alerts ----------------------------------------------------------
t1, t2 = st.columns([2, 1])
with t1:
    st.markdown("**Density over time** (people per m², each point is one photo)")
    df = pd.DataFrame([{"n": i + 1, "zone": r["zone"], "density": r["density"]} for i, r in enumerate(records)])
    if len(df) > 1:
        st.line_chart(df.pivot_table(index="n", columns="zone", values="density"), height=240)
    else:
        st.caption("Upload a second photo to see the trend.")
with t2:
    st.markdown("**Alert log**")
    if st.session_state["alerts"]:
        st.dataframe(pd.DataFrame(st.session_state["alerts"][::-1]), hide_index=True, use_container_width=True)
    else:
        st.caption("No risk-level changes yet.")

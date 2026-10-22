"""Control-room dashboard (Streamlit).

Run:  streamlit run app/app.py
Input is a recorded video or an image per zone, standing in for a camera feed.
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.heatmap import overlay_heatmap  # noqa: E402
from src.pipeline import analyze_frames  # noqa: E402
from src.predict import backend_name, ensure_model, to_rgb_array  # noqa: E402
from src.risk import LEVEL_COLORS, load_thresholds  # noqa: E402
from src.video import sample_frames  # noqa: E402

# Placeholder areas in square metres. Measure the real visible ground area per camera.
ZONES = {"Ramkund": 200.0, "Panchvati": 300.0, "Trimbakeshwar": 250.0}
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
IMAGE_TYPES = ["jpg", "jpeg", "png"]

st.set_page_config(page_title="Simhastha Suraksha", layout="wide")


def _init_model():
    # Not st.cache_resource: a failed first attempt (e.g. secret not set yet) must be retried.
    if not st.session_state.get("model_ok"):
        st.session_state["model_ok"] = ensure_model()
    return st.session_state["model_ok"]


def badge(level):
    color = LEVEL_COLORS[level]
    return ('<span style="background:%s;color:#1a1a1a;padding:4px 14px;border-radius:6px;'
            'font-weight:600">%s</span>' % (color, level))


def frames_from_upload(uploaded, every_sec, max_frames):
    suffix = Path(uploaded.name).suffix.lower().lstrip(".")
    if suffix in IMAGE_TYPES:
        import numpy as np
        from PIL import Image
        yield 0.0, np.asarray(Image.open(uploaded).convert("RGB"))
        return
    with tempfile.NamedTemporaryFile(suffix="." + suffix, delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
    yield from sample_frames(tmp.name, every_sec=every_sec, max_frames=max_frames)


_init_model()
thresholds = load_thresholds()
st.session_state.setdefault("latest", {})
st.session_state.setdefault("alerts", [])
st.session_state.setdefault("history", {})

st.title("सिंहस्थ Suraksha")
st.caption("AI-based crowd density and risk alerts for control rooms. Decision-support aid only; "
           "it does not replace administrative or police judgement.")
if backend_name() == "stub":
    st.warning("Demo mode: no trained model found, so counts are simulated. "
               "Add models/csrnet.onnx (or set SS_MODEL_URL) for real predictions.")

with st.sidebar:
    st.header("Settings")
    every_sec = st.slider("Sample one frame every (seconds)", 1.0, 10.0, 2.0, 0.5)
    max_frames = st.slider("Max frames per video", 5, 120, 30)
    st.subheader("Risk thresholds (people per m²)")
    st.table(pd.DataFrame({"Level": ["Medium from", "High from", "Critical from"],
                           "Density": [thresholds["medium"], thresholds["high"], thresholds["critical"]]}))
    st.caption("Placeholders. Edit config/risk_thresholds.yaml.")

tabs = st.tabs(["Overview"] + list(ZONES))

with tabs[0]:
    st.subheader("All zones")
    if st.session_state["latest"]:
        rows = [{"Zone": z, "People": round(v["count"]), "Density (per m²)": round(v["density"], 2), "Risk": v["level"]}
                for z, v in st.session_state["latest"].items()]
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("Analyse a zone to see it here.")
    st.subheader("Alert log")
    if st.session_state["alerts"]:
        st.dataframe(pd.DataFrame(st.session_state["alerts"]))
    else:
        st.write("No alerts yet.")

for tab, (zone, default_area) in zip(tabs[1:], ZONES.items()):
    with tab:
        left, right = st.columns([1, 2])
        with left:
            area = st.number_input("Visible ground area (m²)", min_value=1.0, value=default_area, step=10.0,
                                   key="area_" + zone)
            uploaded = st.file_uploader("Camera feed stand-in (video or image)", type=VIDEO_TYPES + IMAGE_TYPES,
                                        key="file_" + zone)
            run = st.button("Analyse " + zone, key="run_" + zone)
        if run:
            if uploaded is None:
                st.error("Choose a video or image first.")
            else:
                with st.spinner("Analysing frames..."):
                    result = analyze_frames(frames_from_upload(uploaded, every_sec, max_frames), area, thresholds)
                if not result["history"]:
                    st.error("Could not read any frames from that file.")
                else:
                    last = result["history"][-1]
                    st.session_state["latest"][zone] = last
                    st.session_state["history"][zone] = result["history"]
                    for a in result["alerts"]:
                        st.session_state["alerts"].append({"Zone": zone, "Time (s)": a["time_s"],
                                                           "Change": "%s to %s" % (a["from"], a["to"])})
                    st.session_state["frame_" + zone] = overlay_heatmap(result["last_frame"], result["last_density_map"])
        history = st.session_state["history"].get(zone)
        if history:
            last = history[-1]
            with right:
                st.image(st.session_state["frame_" + zone], caption="Heatmap overlay (relative to the densest spot in this frame)")
            c1, c2, c3 = st.columns(3)
            c1.metric("People counted", round(last["count"]))
            c2.metric("Density (per m²)", round(last["density"], 2))
            c3.markdown("**Risk level**<br>" + badge(last["level"]), unsafe_allow_html=True)
            if len(history) > 1:
                st.line_chart(pd.DataFrame(history).set_index("time_s")[["density"]])

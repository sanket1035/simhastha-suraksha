"""Control-room dashboard (Streamlit).

Run:  streamlit run app/app.py
Two views over one shared store:
  Pilgrim view   public, read-only: crowd level and a heatmap per zone, no raw photos.
  Control room   staff login (SS_STAFF_PASSWORD): upload camera photos, full detail, alerts.
Every uploaded photo, or frame from the camera-ingest worker (python -m src.ingest), updates the
pilgrim view for all visitors within seconds. State is shared through SS_LIVE_DIR.
"""
import hmac
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.live import LiveStore, build_record  # noqa: E402
from src.predict import backend_name, ensure_model, predict  # noqa: E402
from src.risk import LEVELS, load_thresholds  # noqa: E402

# Placeholder areas in square metres. Measure the real visible ground area per camera.
ZONES = {"Ramkund": 200.0, "Panchvati": 300.0, "Trimbakeshwar": 250.0}
# Centre points (lat, lon): Ramkund = Ram Ghat on the Godavari, Panchvati = Kalaram Temple area, Trimbakeshwar =
# the temple. Looked up from public sources; set them to the real camera locations before use.
ZONE_GEO = {"Ramkund": (20.0077, 73.7921), "Panchvati": (20.0069, 73.7950), "Trimbakeshwar": (19.9322, 73.5309)}
IMAGE_TYPES = ["jpg", "jpeg", "png"]
VIEW_MODES = ["Heatmap overlay", "Heat only", "Original"]
# Level colours chosen to sit on the cream/aubergine palette while staying distinguishable.
LEVEL_COLORS = {"Low": "#2E8B6A", "Medium": "#E39A00", "High": "#D4571C", "Critical": "#B3261E"}


st.set_page_config(page_title="Simhastha Suraksha", page_icon="🔱", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,700;0,9..144,900;1,9..144,400;1,9..144,700&family=Inter:wght@400;500;600&family=Tiro+Devanagari+Hindi&display=swap');
:root {--ink:#2B0E27;--aub:#3A0F34;--gold:#EAA000;--ice:#DDE3E8;--cream:#EDE6DA;--mute:#7A6A74;--rule:#2B0E2733}
html, body, [class*="css"] {font-family:'Inter',sans-serif}
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden;height:0}
.block-container {padding-top:1.4rem;max-width:1320px}
[data-testid="stSidebar"] {background:#E3DACB;border-right:1px solid var(--rule)}
.top {display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--ink);
  padding-bottom:10px;font-size:.68rem;letter-spacing:.28em;text-transform:uppercase;color:var(--ink)}
.top .dot {display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;background:var(--gold)}
.top .dot.live {background:#2E8B6A}
.hero {padding:34px 0 10px;position:relative}
.hero .kick {font-size:.68rem;letter-spacing:.32em;color:var(--mute);text-transform:uppercase}
.hero h1 {margin:10px 0 0;font-family:'Tiro Devanagari Hindi','Fraunces',serif;font-weight:400;
  font-size:clamp(3rem,8vw,6.4rem);line-height:.98;letter-spacing:-.02em;color:var(--aub)}
.hero h1 em {font-family:'Fraunces',serif;font-style:italic;font-weight:900;
  background:linear-gradient(95deg,var(--aub) 10%,var(--gold) 90%);-webkit-background-clip:text;color:transparent;
  filter:drop-shadow(0 0 .5px var(--aub))}
.hero p {max-width:560px;color:#54404f;font-size:.95rem;line-height:1.6;margin:14px 0 0}
.cols {display:grid;grid-template-columns:repeat(3,1fr);margin-top:18px;border-top:1px solid var(--ink)}
.col {padding:18px 20px 12px;border-left:1px solid var(--rule)}
.col:first-child {border-left:0;padding-left:0}
.col .z {font-family:'Fraunces',serif;font-size:1.5rem;font-weight:700;color:var(--aub)}
.col .lv {font-family:'Fraunces',serif;font-style:italic;font-size:2.7rem;font-weight:900;line-height:1.05;color:var(--c)}
.col .adv {color:#54404f;font-size:.88rem;line-height:1.5;margin:6px 0 10px;min-height:2.6em}
.col .ts, .lab {font-size:.66rem;letter-spacing:.24em;text-transform:uppercase;color:var(--mute)}
.col .meter {height:3px;background:var(--rule);margin:12px 0 4px;position:relative}
.col .meter i {position:absolute;left:0;top:0;bottom:0;background:var(--c)}
.stats {display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--ink);border-bottom:1px solid var(--rule);margin:14px 0 22px}
.stat {padding:14px 18px;border-left:1px solid var(--rule)}
.stat:first-child {border-left:0;padding-left:0}
.stat .v {font-family:'Fraunces',serif;font-size:2.4rem;font-weight:900;line-height:1.1;color:var(--c,var(--aub))}
.stat .s {font-size:.78rem;color:var(--mute)}
.big {background:var(--aub);color:var(--cream);padding:24px;border-radius:2px;border-top:6px solid var(--c)}
.big .lab {color:#cbb8c6}
.big .lv {font-family:'Fraunces',serif;font-style:italic;font-weight:900;font-size:3.2rem;line-height:1;color:var(--c);margin-top:6px}
.big .row {display:flex;justify-content:space-between;margin-top:18px;font-size:.78rem;color:#cbb8c6}
.big .row b {display:block;font-family:'Fraunces',serif;font-size:1.5rem;color:var(--cream)}
.big .meter {height:4px;background:#ffffff33;margin-top:18px;position:relative}
.big .meter i {position:absolute;top:-4px;width:3px;height:12px;background:var(--c)}
.zline {border-top:1px solid var(--rule);padding:10px 0;display:flex;justify-content:space-between;align-items:baseline}
.zline .n {font-family:'Fraunces',serif;font-weight:700;color:var(--aub)}
.zline .b {font-size:.68rem;letter-spacing:.2em;text-transform:uppercase;font-weight:600;color:var(--c)}
.zline .m {font-size:.8rem;color:var(--mute)}
.empty {text-align:center;padding:70px 20px;border-top:1px solid var(--ink);border-bottom:1px solid var(--rule);color:var(--mute)}
.empty h3 {font-family:'Fraunces',serif;font-size:1.8rem;color:var(--aub);margin-bottom:.3rem}
.stRadio label p, .stButton button {font-family:'Inter',sans-serif}
.stButton button {border-radius:2px;border:1px solid var(--ink);background:transparent;color:var(--ink)}
.stButton button:hover {background:var(--aub);color:var(--cream);border-color:var(--aub)}
div[data-testid="stImage"] img {border-radius:2px}
@media (max-width:800px) {.cols,.stats {grid-template-columns:1fr} .col,.stat {border-left:0;padding-left:0}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _init_model():
    # Not st.cache_resource: a failed first attempt (e.g. secret not set yet) must be retried.
    if not st.session_state.get("model_ok"):
        st.session_state["model_ok"] = ensure_model()
    return st.session_state["model_ok"]


@st.cache_resource
def get_store():
    return LiveStore()


def staff_password():
    pw = os.environ.get("SS_STAFF_PASSWORD", "")
    if not pw:
        try:
            pw = st.secrets.get("SS_STAFF_PASSWORD", "")
        except Exception:
            pw = ""
    return pw


# Plain-language advice for the public. Wording is a placeholder for the control room to approve.
ADVICE = {"Low": "Clear. Good time to visit.",
          "Medium": "Moderately busy. Keep moving and stay with your group.",
          "High": "Crowded. Consider waiting or using another ghat.",
          "Critical": "Very crowded. Please avoid this area now and follow police directions."}


def header(live, kicker, title_html, blurb):
    st.markdown(
        '<div class="top"><span>Simhastha Kumbh Mela · Nashik</span>'
        '<span><span class="dot %s"></span>%s</span></div>'
        '<div class="hero"><div class="kick">%s</div><h1>%s</h1><p>%s</p></div>' %
        ("live" if live else "", "Model live" if live else "Demo mode, simulated counts", kicker, title_html, blurb),
        unsafe_allow_html=True)


def stat(label, value, sub, color=None):
    style = ' style="--c:%s"' % color if color else ""
    return ('<div class="stat"%s><div class="lab">%s</div><div class="v">%s</div><div class="s">%s</div></div>'
            % (style, label, value, sub))


def zone_line(z, r):
    if not r:
        return ('<div class="zline" style="--c:#7A6A74"><span class="n">%s</span>'
                '<span class="m">waiting for first photo</span></div>' % z)
    return ('<div class="zline" style="--c:%s"><span class="n">%s</span><span><span class="m">%s people · '
            '%.2f/m² &nbsp;</span><span class="b">%s</span></span></div>' %
            (LEVEL_COLORS[r["level"]], z, "{:,}".format(round(r["count"])), r["density"], r["level"]))


VENDOR = Path(__file__).resolve().parent / "vendor"
MAP_HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="referrer" content="strict-origin-when-cross-origin"><style>%(css)s
html,body,#m{height:100%%;margin:0;background:#E3DACB;font-family:Inter,system-ui,sans-serif}
.leaflet-container{background:#E3DACB}
.pin{background:none;border:0;font:600 12px Inter,system-ui,sans-serif;color:#2B0E27;white-space:nowrap;
 text-shadow:0 0 3px #EDE6DA,0 0 3px #EDE6DA,0 0 3px #EDE6DA}
</style></head><body><div id="m"></div><script>%(js)s</script><script>
var Z=%(zones)s;
var map=L.map('m',{zoomControl:true,scrollWheelZoom:false});
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19,attribution:'&copy; OpenStreetMap contributors'}).addTo(map);
var pts=[];
Z.forEach(function(z){
  pts.push([z.lat,z.lon]);
  L.circleMarker([z.lat,z.lon],{radius:z.radius,color:z.color,weight:2,fillColor:z.color,fillOpacity:.28}).addTo(map);
  L.circleMarker([z.lat,z.lon],{radius:7,color:'#fff',weight:2,fillColor:z.color,fillOpacity:1})
    .bindTooltip('<b>'+z.zone+'</b><br>'+z.info).addTo(map);
  L.marker([z.lat,z.lon],{interactive:false,icon:L.divIcon({className:'pin',html:z.zone,iconAnchor:[-14,6]})}).addTo(map);
});
function fit(list){map.fitBounds(list,{padding:[60,60],maxZoom:15});}
var core=pts.filter(function(p){return p[1]>73.7;});
fit(core.length>1?core:pts);
var bar=L.control({position:'topright'});
bar.onAdd=function(){var d=L.DomUtil.create('div','leaflet-bar');
  d.innerHTML='<a href="#" style="width:auto;padding:0 10px;font:600 12px/30px Inter,sans-serif" id="b1">Nashik ghats</a>'+
  '<a href="#" style="width:auto;padding:0 10px;font:600 12px/30px Inter,sans-serif" id="b2">All zones</a>';
  L.DomEvent.disableClickPropagation(d);
  d.querySelector('#b1').onclick=function(e){e.preventDefault();fit(core);};
  d.querySelector('#b2').onclick=function(e){e.preventDefault();fit(pts);};return d;};
bar.addTo(map);
</script></body></html>"""


def crowd_map(latest, height=420):
    """Map of the zones: circle colour = risk level, size = people counted, grey = no data yet."""
    import json
    import streamlit.components.v1 as components
    zones = []
    for z, (lat, lon) in ZONE_GEO.items():
        r = latest.get(z)
        zones.append({
            "zone": z, "lat": lat, "lon": lon, "color": LEVEL_COLORS[r["level"]] if r else "#7A6A74",
            "radius": round(16 + min(r["count"], 1500) / 1500.0 * 30) if r else 10,
            "info": ("%s · %s people · %.2f per m²" % (r["level"], "{:,}".format(round(r["count"])), r["density"]))
                    if r else "No data yet"})
    html = MAP_HTML % {"css": (VENDOR / "leaflet.css").read_text(), "js": (VENDOR / "leaflet.js").read_text(),
                       "zones": json.dumps(zones)}
    components.html(html, height=height)


def pilgrim_view(store):
    @st.fragment(run_every=10)
    def live_board():
        _, _, latest = store.snapshot()
        if not latest:
            st.markdown('<div class="empty"><h3>No live data yet</h3>Crowd updates appear here as soon as '
                        'the cameras report.</div>', unsafe_allow_html=True)
            return
        st.markdown('<div style="border-top:1px solid var(--ink);margin:18px 0 8px"></div>'
                    '<div class="lab">Where is it crowded</div>', unsafe_allow_html=True)
        crowd_map(latest)
        st.markdown('<div style="border-top:1px solid var(--ink);margin:14px 0 6px"></div>', unsafe_allow_html=True)
        cols = st.columns(len(ZONES), gap="large")
        for col, z in zip(cols, ZONES):
            with col:
                r = latest.get(z)
                if r:
                    pct = min(r["density"] / (thresholds["critical"] * 1.25), 1.0) * 100
                    st.markdown(
                        '<div class="col" style="--c:%s;border:0;padding:0"><div class="z">%s</div>'
                        '<div class="lv">%s</div><div class="adv">%s</div>'
                        '<div class="meter"><i style="width:%.0f%%"></i></div>'
                        '<div class="ts">Updated %s</div></div>' %
                        (LEVEL_COLORS[r["level"]], z, r["level"], ADVICE[r["level"]], pct,
                         r["ts"].strftime("%H:%M")), unsafe_allow_html=True)
                    # Heat only: no raw photo is ever shown to the public.
                    st.image(r["heat"], use_container_width=True, caption="Brighter means more crowded")
                else:
                    st.markdown('<div class="col" style="border:0;padding:0"><div class="z">%s</div>'
                                '<div class="adv">Waiting for the first camera update.</div></div>' % z,
                                unsafe_allow_html=True)
        st.caption("This page refreshes by itself. Always follow the instructions of police and volunteers on site.")

    live_board()


def control_view(store):
    with st.container(border=True):
        st.markdown('<div class="lab">Add a camera photo</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        zone = c1.selectbox("Zone", list(ZONES))
        area = c2.number_input("Visible ground area (m²)", min_value=1.0, value=ZONES[zone], step=10.0,
                               key="area_" + zone, help="Measure the ground area this camera actually sees.")
        files = st.file_uploader("Upload one or more photos", type=IMAGE_TYPES, accept_multiple_files=True,
                                 key="up_%d" % st.session_state.setdefault("uploader_n", 0))
    with st.expander("Settings and account"):
        st.markdown("**Risk thresholds** (people per m²): Medium ≥ %.1f · High ≥ %.1f · Critical ≥ %.1f" %
                    (thresholds["medium"], thresholds["high"], thresholds["critical"]))
        st.caption("Placeholders. Edit config/risk_thresholds.yaml.")
        b1, b2 = st.columns(2)
        if b1.button("Clear dashboard", use_container_width=True):
            store.clear()
            st.rerun()
        if b2.button("Sign out", use_container_width=True):
            st.session_state["staff"] = False
            st.rerun()

    if files:
        with st.spinner("Analysing %d photo(s)..." % len(files)):
            for f in files:
                store.add(build_record(np.asarray(Image.open(f).convert("RGB")), zone, area, thresholds, name=f.name))
        st.session_state["focus"] = zone
        st.session_state["uploader_n"] += 1
        st.rerun()

    records, alerts, latest = store.snapshot()
    worst = max(latest.values(), key=lambda r: LEVELS.index(r["level"]))["level"] if latest else "-"
    total = sum(r["count"] for r in latest.values())
    st.markdown('<div class="stats">%s%s%s%s</div>' % (
        stat("People in view", "{:,}".format(round(total)), "latest photo per zone"),
        stat("Zones reporting", "%d / %d" % (len(latest), len(ZONES)), "upload to add one"),
        stat("Highest risk", worst, "across zones", LEVEL_COLORS.get(worst)),
        stat("Alerts", len(alerts), "level changes")), unsafe_allow_html=True)
    if not records:
        st.markdown('<div class="empty"><h3>No photos yet</h3>Pick a zone above and upload a crowd '
                    'photo. The public dashboard updates at the same time.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="lab" style="margin-bottom:6px">Zone map</div>', unsafe_allow_html=True)
    crowd_map(latest, height=360)
    names = list(latest)
    focus = st.session_state.get("focus", names[-1])
    if focus not in latest:
        focus = names[-1]
    focus = st.radio("Zone", names, index=names.index(focus), horizontal=True, label_visibility="collapsed")
    cur = latest[focus]
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
            '<div class="big" style="--c:%s"><div class="lab">Risk level · %s</div><div class="lv">%s</div>'
            '<div class="meter"><i style="left:calc(%.1f%% - 2px)"></i></div>'
            '<div class="row"><div><b>%s</b>people</div><div><b>%.2f</b>per m²</div>'
            '<div><b>%d</b>m² area</div></div></div>' %
            (LEVEL_COLORS[cur["level"]], focus, cur["level"], pos, "{:,}".format(round(cur["count"])),
             cur["density"], cur["area"]), unsafe_allow_html=True)
        st.write("")
        st.markdown("".join(zone_line(z, latest.get(z)) for z in ZONES), unsafe_allow_html=True)

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
        if alerts:
            st.dataframe(pd.DataFrame(alerts[::-1]), hide_index=True, use_container_width=True)
        else:
            st.caption("No risk-level changes yet.")


_init_model()
thresholds = load_thresholds()
store = get_store()
live = backend_name() == "onnx"

def staff_login():
    """Shown in the page body (not the sidebar, which is hidden on phones)."""
    pw = staff_password()
    with st.expander("🔒 Staff login (police, volunteers, control room)"):
        if not pw:
            st.caption("Staff login is not configured. Set SS_STAFF_PASSWORD on the server.")
            return
        entered = st.text_input("Password", type="password")
        if st.button("Sign in"):
            if hmac.compare_digest(entered.encode(), pw.encode()):
                st.session_state["staff"] = True
                st.rerun()
            st.caption("Wrong password.")


if st.session_state.get("staff"):
    header(live, "Control room", "Crowd <em>command</em>",
           "Upload camera photos here. The public dashboard updates for every visitor within seconds.")
    if not live:
        st.warning("No trained model found, so counts are simulated. Set SS_MODEL_URL and reboot the app.")
    control_view(store)
else:
    header(live, "Live crowd status", "सिंहस्थ <em>Suraksha</em>",
           "Know how busy each ghat is before you go. Updated from the cameras, in plain words.")
    staff_login()
    pilgrim_view(store)

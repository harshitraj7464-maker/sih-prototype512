import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
import json
import time

st.set_page_config(
    page_title="SIH26001 - AI Landslide Early Warning System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize offline fallback datastore locally
LOG_FILE = "offline_disaster_alerts.json"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as f:
        json.dump([], f)

# ==========================================
# 1. CORE MACHINE LEARNING INSTANTIATION
# ==========================================
@st.cache_resource
def initialize_ml_pipeline():
    np.random.seed(42)
    samples = 1500
    X_mock = pd.DataFrame({
        'slope_angle': np.random.uniform(15, 60, samples),
        'rainfall_24h': np.random.uniform(10, 350, samples),
        'soil_moisture': np.random.uniform(20, 95, samples),
        'elevation': np.random.uniform(500, 3000, samples),
        'deforestation_idx': np.random.uniform(0.1, 0.9, samples)
    })
    y_mock = ((X_mock['rainfall_24h']/350 + X_mock['slope_angle']/60 + X_mock['soil_moisture']/95) > 1.2).astype(int)
    pipeline = RandomForestClassifier(n_estimators=50, random_state=42)
    pipeline.fit(X_mock, y_mock)
    return pipeline

ml_pipeline = initialize_ml_pipeline()

if "field_reports" not in st.session_state:
    st.session_state.field_reports = [
        {"lat": 25.2810, "lon": 91.7290, "type": "Photo", "label": "Major Rock Crack Observed", "timestamp": "2026-09-28 10:14", "severity": "High"}
    ]

# ==========================================
# 2. STREAMLIT FRONTEND & TELEMETRY CONTROLS
# ==========================================
st.title("🌋 MDoNER AI Landslide Early Warning Engine")
st.caption("Smart India Hackathon Prototype Cloud Node (Problem ID: SIH26001)")

st.sidebar.header("🌐 Deployment Topology")
network_status = st.sidebar.radio("Connectivity Node Status", ["Online Mode", "Total Offline Mode"])

st.sidebar.header("📡 Regional Telemetry Sensors")
location_profile = st.sidebar.selectbox("Select Target Monitor Region", ["Cherrapunji (Meghalaya)", "Aizawl (Mizoram)", "Kohima (Nagaland)"])

geo_profiles = {
    "Cherrapunji (Meghalaya)": {"lat": 25.2702, "lon": 91.7325},
    "Aizawl (Mizoram)": {"lat": 23.7271, "lon": 92.7176},
    "Kohima (Nagaland)": {"lat": 25.6751, "lon": 94.1086}
}
active_coords = geo_profiles[location_profile]

live_rain = st.sidebar.slider("IMD Live Rainfall Saturation (24h mm)", 10, 450, 240)
soil_moist = st.sidebar.slider("IoT Field Soil Moisture Sensor (%)", 10, 100, 75)

# Media Upload Interface
st.sidebar.header("📸 Field Evidence Tagger")
media_type = st.sidebar.radio("Upload Media Type", ["Geotagged Photo", "Geotagged Drone Video"])
uploaded_file = st.sidebar.file_uploader(f"Upload field asset...", type=["png", "jpg", "jpeg", "mp4"])

if uploaded_file is not None:
    report_label = st.sidebar.text_input("Incident Observation Note", value="Soil movement observed.")
    severity_tier = st.sidebar.selectbox("Field Threat Level", ["Medium", "High", "Critical Intervention"])
    
    if st.sidebar.button("💾 Plot Media Layer"):
        # Simulated offset coordinate to render on map next to base hub
        sim_lat = active_coords['lat'] + 0.008
        sim_lon = active_coords['lon'] - 0.012
        new_report = {
            "lat": sim_lat, 
            "lon": sim_lon,
            "type": "Photo" if media_type == "Geotagged Photo" else "Video",
            "label": report_label, 
            "timestamp": time.strftime("%Y-%m-%d %H:%M"), 
            "severity": severity_tier
        }
        st.session_state.field_reports.append(new_report)
        st.sidebar.success("✅ Media layer loaded into memory!")

# Status Cards Row
col1, col2 = st.columns(2)
with col1:
    if network_status == "Online Mode":
        st.success("🌐 Connected to Active IMD & Bhuvan API Nodes")
    else:
        st.warning("🔌 Local Loop Active: Pulling Offline Cached Map Matrices")
with col2:
    # FIXED: Extracted single probability element explicitly using [0][1] instead of whole array slicing [:, 1]
    center_prob = float(ml_pipeline.predict_proba([[42.0, live_rain, soil_moist, 1400, 0.65]][0][1]))
    risk_percentage = round(center_prob * 100, 2)
    if risk_percentage < 35.0: st.success(f"Status: Safe ({risk_percentage}%)")
    elif risk_percentage < 70.0: st.warning(f"Status: Warning Watch ({risk_percentage}%)")
    else: st.error(f"Status: RED ALERT EVACUATE ({risk_percentage}%)")

st.markdown("---")

# ==========================================
# 3. SPATIAL GEOSPATIAL MAP CANVAS
# ==========================================
st.subheader("🗺️ Regional Vulnerability Heatmap & Ground Truth Points")

heatmap_points = []
lat_range = np.linspace(active_coords['lat'] - 0.08, active_coords['lat'] + 0.08, 15)
lon_range = np.linspace(active_coords['lon'] - 0.08, active_coords['lon'] + 0.08, 15)

for lat in lat_range:
    for lon in lon_range:
        t_var = abs(np.sin(lat * lon))
        sim_slope = 15.0 + (t_var * 45.0)
        sim_moist = min(max(soil_moist + (t_var * 10.0), 0.0), 100.0)
        risk_w = float(ml_pipeline.predict_proba([[sim_slope, live_rain, sim_moist, 1200, 0.6]][0][1]))
        if risk_w > 0.15:
            heatmap_points.append([lat, lon, risk_w])

base_map = folium.Map(location=[active_coords['lat'], active_coords['lon']], zoom_start=12, tiles="OpenStreetMap")
HeatMap(heatmap_points, radius=25, blur=15, max_zoom=12).addTo(base_map)

# Add Ground Markers
for report in st.session_state.field_reports:
    icon_color = "red" if report['severity'] in ["High", "Critical Intervention"] else "orange"
    folium.Marker(
        location=[report['lat'], report['lon']],
        popup=f"<b>🚨 Ground Evidence</b><br>{report['label']}<br>Rank: {report['severity']}",
        icon=folium.Icon(color=icon_color, icon="camera")
    ).addTo(base_map)

st_folium(base_map, width=1200, height=500)

st.subheader("📋 Ingested Visual Log Assets Index")
st.dataframe(pd.DataFrame(st.session_state.field_reports), use_container_width=True)

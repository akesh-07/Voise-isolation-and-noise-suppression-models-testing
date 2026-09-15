import streamlit as st
import requests
import base64
import json
import time

# --- Setup ---
st.set_page_config(
    page_title="ClearerVoice Target Speaker Extraction",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CSS Styling ---
st.markdown("""
<style>
    /* Global Background and Fonts */
    body {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #238636;
        color: white;
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 6px;
        font-weight: 500;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: rgba(240, 246, 252, 0.1);
    }
    
    /* Panels */
    .metric-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
        margin-bottom: 16px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #58a6ff;
    }
    
    .metric-label {
        font-size: 14px;
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🎙️ ClearerVoice TSE")
st.markdown("**Target Speaker Extraction (Audio-Only) using SpEx+**")

st.markdown("""
<div style="background-color: #1f2937; padding: 15px; border-radius: 8px; border-left: 4px solid #00cc66; margin-bottom: 20px;">
    <h3 style="margin-top: 0; color: #00cc66;">🚀 New Feature: Real-time Streaming</h3>
    <p>We've added a sub-50ms latency WebSocket stream for the GTCRN pipeline.</p>
    <a href="http://localhost:8000/static/realtime.html" target="_blank" style="display: inline-block; background-color: #00cc66; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; font-weight: bold;">Launch Real-time Live Stream</a>
</div>
""", unsafe_allow_html=True)

# --- Configuration ---
API_URL = "http://localhost:8000/process"

# --- Main Interface ---
st.write("Upload a call recording containing multiple speakers. The system will automatically detect the first speaker's voice to use as an enrollment reference, and then extract their voice from the entire recording.")

uploaded_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])

if uploaded_file is not None:
    # Display Original Audio
    st.subheader("1. Original Audio")
    st.audio(uploaded_file, format='audio/wav')
    
    st.markdown("### Processing Settings")
    pipeline_options = {
        "SpEx+ Only (Isolation)": "spex",
        "DeepFilterNet3 → SpEx+ (Pre-Denoising + Isolation)": "deepfilter_spex",
        "SpEx+ + MossFormerGAN (Isolation + Noise Suppression)": "mossformer",
        "GTCRN (Streaming Noise Suppression)": "gtcrn"
    }
    selected_pipeline_label = st.radio("Select Pipeline:", list(pipeline_options.keys()))
    processing_mode = pipeline_options[selected_pipeline_label]
    
    if st.button("Extract Target Speaker"):
        with st.spinner("Processing... This may take a moment. (Auto-detecting enrollment segment & running inference)"):
            try:
                # Send to backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
                payload = {"processing_mode": processing_mode}
                start_req = time.time()
                response = requests.post(API_URL, files=files, data=payload)
                req_time = time.time() - start_req
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("Extraction Complete!")
                    
                    if processing_mode == "gtcrn":
                        st.subheader("2. Cleaned Audio (GTCRN)")
                        st.write(f"Sample Rate: **{data['sample_rate']} Hz**")
                        # Decode and play extracted audio (which is the GTCRN audio)
                        extracted_bytes = base64.b64decode(data['extracted_audio_b64'])
                        st.audio(extracted_bytes, format='audio/wav')
                    else:
                        if data.get('denoised_audio_b64'):
                            st.subheader("1.5 Pre-Denoised Audio (DeepFilterNet3)")
                            st.write(f"Sample Rate: **{data['sample_rate']} Hz**")
                            denoised_bytes = base64.b64decode(data['denoised_audio_b64'])
                            st.audio(denoised_bytes, format='audio/wav')
                        
                        # Layout results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("2. Auto-Detected Enrollment")
                            if "enrollment_start" in data and "enrollment_end" in data:
                                st.write(f"Segment from **{data['enrollment_start']}s** to **{data['enrollment_end']}s**")
                            # Decode and play enrollment
                            if data.get('enrollment_audio_b64'):
                                enrollment_bytes = base64.b64decode(data['enrollment_audio_b64'])
                                st.audio(enrollment_bytes, format='audio/wav')
                        
                        with col2:
                            st.subheader("3a. Extracted (Isolated)")
                            st.write(f"Sample Rate: **{data['sample_rate']} Hz**")
                            # Decode and play extracted audio
                            extracted_bytes = base64.b64decode(data['extracted_audio_b64'])
                            st.audio(extracted_bytes, format='audio/wav')
                            
                        with col3:
                            if data.get('enhanced_audio_b64'):
                                st.subheader("3b. Enhanced (Denoised)")
                                st.write(f"Sample Rate: **{data['sample_rate']} Hz**")
                                # Decode and play enhanced audio
                                enhanced_bytes = base64.b64decode(data['enhanced_audio_b64'])
                                st.audio(enhanced_bytes, format='audio/wav')
                            else:
                                st.write("")
                    
                    # Metrics
                    import textwrap
                    stage_times = data.get("stage_times", {})
                    metrics_html = f"""
<div class="metric-container" style="display: flex; gap: 20px; flex-wrap: wrap;">
    <div>
        <div class="metric-label">Total Processing Time</div>
        <div class="metric-value">{data['processing_time']}s</div>
    </div>
"""
                    if "denoise_time" in stage_times:
                        metrics_html += f"""
    <div>
        <div class="metric-label">Pre-Denoise Time</div>
        <div class="metric-value">{stage_times['denoise_time']}s</div>
    </div>"""
                    if "vad_time" in stage_times:
                        metrics_html += f"""
    <div>
        <div class="metric-label">VAD Time</div>
        <div class="metric-value">{stage_times['vad_time']}s</div>
    </div>"""
                    if "spex_time" in stage_times:
                        metrics_html += f"""
    <div>
        <div class="metric-label">SpEx+ Time</div>
        <div class="metric-value">{stage_times['spex_time']}s</div>
    </div>"""
                    if "gtcrn_time" in stage_times:
                        metrics_html += f"""
    <div>
        <div class="metric-label">GTCRN Time</div>
        <div class="metric-value">{stage_times['gtcrn_time']}s</div>
    </div>"""
                    if "enhancement_time" in stage_times:
                        metrics_html += f"""
    <div>
        <div class="metric-label">MossFormer Time</div>
        <div class="metric-value">{stage_times['enhancement_time']}s</div>
    </div>"""
                        
                    metrics_html += "\n</div>"
                    st.markdown(metrics_html, unsafe_allow_html=True)
                    
                else:
                    st.error(f"Backend Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

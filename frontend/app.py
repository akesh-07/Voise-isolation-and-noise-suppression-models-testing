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
    processing_mode = st.radio("Select Pipeline:", ["SpEx+ Only (Isolation)", "SpEx+ + MossFormerGAN (Isolation + Noise Suppression)"])
    enable_enhancement = processing_mode == "SpEx+ + MossFormerGAN (Isolation + Noise Suppression)"
    
    if st.button("Extract Target Speaker"):
        with st.spinner("Processing... This may take a moment. (Auto-detecting enrollment segment & running SpEx+)"):
            try:
                # Send to backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
                payload = {"enable_enhancement": enable_enhancement}
                start_req = time.time()
                response = requests.post(API_URL, files=files, data=payload)
                req_time = time.time() - start_req
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("Extraction Complete!")
                    
                    # Layout results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("2. Auto-Detected Enrollment")
                        st.write(f"Segment from **{data['enrollment_start']}s** to **{data['enrollment_end']}s**")
                        # Decode and play enrollment
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
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Total Processing Time</div>
                        <div class="metric-value">{data['processing_time']}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error(f"Backend Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

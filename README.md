# ClearerVoice TSE Web Application

This project provides an offline, self-hosted web application for Audio-Only Target Speaker Extraction (TSE) using the **SpEx+** model from the [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) repository.

The application uses **FastAPI** as the backend for audio processing and **Streamlit** for a premium, dark-themed frontend user interface.

## Prerequisites

- **Python 3.9+**
- **FFmpeg**: Required for audio processing. Make sure it is installed and added to your system `PATH`.
  - Windows: `winget install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

## Installation

1. **Clone the repository** (or navigate to this directory).
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Backend Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Install Frontend Dependencies**:
   ```bash
   cd ../frontend
   pip install -r requirements.txt
   ```

## Running the Application

You will need to start both the backend and the frontend servers.

### 1. Start the Backend

Open a terminal and run:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
*Note: On the first run, the backend will automatically download the SpEx+ pre-trained weights from Hugging Face (`alibabasglab/log_wsj0-2mix_speech_SpEx-plus_2spk`) and the Silero VAD model. Subsequent runs will use the cached weights.*

### 2. Start the Frontend

Open a new terminal, activate your virtual environment, and run:
```bash
cd frontend
streamlit run app.py
```

### 3. Usage
- Open your browser to the local Streamlit URL (usually `http://localhost:8501`).
- Upload a `.wav` file containing multiple overlapping speakers.
- The system will use Silero VAD to automatically detect the first speaker's voice (the enrollment audio) and use SpEx+ to extract that specific speaker from the rest of the mix.

## Acknowledgements

This application directly leverages the SpEx+ implementation and pre-trained checkpoints provided by [ModelScope / ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) under the Apache 2.0 license.

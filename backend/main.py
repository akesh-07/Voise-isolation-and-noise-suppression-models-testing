import io
import time
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import soundfile as sf
import uvicorn
import base64
from inference import SpExPlusInference

app = FastAPI(title="ClearerVoice SpEx+ Backend")
engine = None

@app.on_event("startup")
def startup_event():
    global engine
    # Load model on startup
    engine = SpExPlusInference()
    try:
        engine.load_model()
    except Exception as e:
        print(f"Error loading model on startup: {e}")

@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    enable_enhancement: bool = Form(None),
    processing_mode: str = Form("spex")
):
    global engine
    if engine is None:
        engine = SpExPlusInference()
    if engine.model is None or engine.se_model is None:
        try:
            engine.load_model()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": f"Failed to load model: {str(e)}"})

    try:
        start_time = time.time()
        
        # Read uploaded audio file to bytes
        audio_bytes = await file.read()
        
        import tempfile
        import os
        
        # Write to temporary file for torchaudio to read properly
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, 'wb') as f:
            f.write(audio_bytes)
            
        # Fallback for backward compatibility
        if enable_enhancement is True:
            processing_mode = "mossformer"
            
        # Run inference
        result = engine.process(temp_path, enable_enhancement=False, processing_mode=processing_mode)
        
        extracted_audio = result["extracted_audio"].squeeze().numpy()
        # Normalize extracted audio to prevent clipping/static noise
        max_val = np.max(np.abs(extracted_audio))
        if max_val > 0:
            extracted_audio = extracted_audio / max_val
            
        enrollment_audio = result.get("enrollment_audio")
        if enrollment_audio is not None:
            enrollment_audio = enrollment_audio.squeeze().numpy()
            
        sr = result["sample_rate"]
        
        # Convert extracted audio to base64
        extracted_io = io.BytesIO()
        sf.write(extracted_io, extracted_audio, sr, format="wav")
        extracted_io.seek(0)
        extracted_b64 = base64.b64encode(extracted_io.read()).decode('utf-8')
        
        # Convert enrollment audio to base64 if present
        enrollment_b64 = None
        if enrollment_audio is not None:
            enrollment_io = io.BytesIO()
            sf.write(enrollment_io, enrollment_audio, sr, format="wav")
            enrollment_io.seek(0)
            enrollment_b64 = base64.b64encode(enrollment_io.read()).decode('utf-8')
        
        enhanced_b64 = None
        if "enhanced_audio" in result:
            enhanced_audio = result["enhanced_audio"].squeeze().numpy()
            max_val = np.max(np.abs(enhanced_audio))
            if max_val > 0:
                enhanced_audio = enhanced_audio / max_val
            enhanced_io = io.BytesIO()
            sf.write(enhanced_io, enhanced_audio, sr, format="wav")
            enhanced_io.seek(0)
            enhanced_b64 = base64.b64encode(enhanced_io.read()).decode('utf-8')
        
        processing_time = time.time() - start_time
        
        response_payload = {
            "success": True,
            "processing_time": round(processing_time, 2),
            "stage_times": {k: round(v, 2) for k, v in result.get("stage_times", {}).items()},
            "sample_rate": sr,
            "extracted_audio_b64": extracted_b64
        }
        
        if "enrollment_start_time" in result:
            response_payload["enrollment_start"] = round(result["enrollment_start_time"], 2)
        if "enrollment_end_time" in result:
            response_payload["enrollment_end"] = round(result["enrollment_end_time"], 2)
        if enrollment_b64:
            response_payload["enrollment_audio_b64"] = enrollment_b64
        
        if enhanced_b64:
            response_payload["enhanced_audio_b64"] = enhanced_b64
            
        if "denoised_audio" in result:
            denoised_audio = result["denoised_audio"].squeeze().numpy()
            max_val = np.max(np.abs(denoised_audio))
            if max_val > 0:
                denoised_audio = denoised_audio / max_val
            denoised_io = io.BytesIO()
            sf.write(denoised_io, denoised_audio, sr, format="wav")
            denoised_io.seek(0)
            response_payload["denoised_audio_b64"] = base64.b64encode(denoised_io.read()).decode('utf-8')
            
        return response_payload
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import numpy as np
import soundfile as sf
import torch
import time
from backend.inference import SpExPlusInference
from backend.gtcrn import GTCRNInference, GTCRNStreamSession
import sys

def test_stream_equivalence():
    print("Loading model...")
    engine = GTCRNInference()
    stream_session = GTCRNStreamSession(engine)
    
    print("Generating test noise (sine wave + random noise)...")
    sr = 16000
    t = np.linspace(0, 3, sr * 3, endpoint=False) # 3 seconds
    clean_audio = np.sin(2 * np.pi * 440 * t) * 0.5 # 440 Hz tone
    noise = np.random.randn(len(t)) * 0.1
    audio = (clean_audio + noise).astype(np.float32)
    
    # Save input
    sf.write('test_input.wav', audio, sr)
    
    print("Processing batch...")
    # engine.process pads the audio, processes, and writes to file
    engine.process('test_input.wav', 'test_batch_out.wav')
    
    print("Processing stream...")
    stream_out = []
    # Simulate chunk by chunk on UNPADDED audio, process_chunk handles the padding intrinsically (via prev_chunk)
    for i in range(0, len(audio), 256):
        chunk = audio[i:i+256]
        if len(chunk) < 256:
            chunk = np.pad(chunk, (0, 256 - len(chunk)))
        out_chunk = stream_session.process_chunk(chunk)
        stream_out.append(out_chunk)
        
    stream_out_full = np.concatenate(stream_out)
    
    # We also need to process one last zero-chunk to flush the overlap buffer
    final_chunk = stream_session.process_chunk(np.zeros(256, dtype=np.float32))
    # The first chunk out_chunk_0 corresponds to the padding, so we discard it.
    stream_out_full = np.concatenate(stream_out[1:])
    
    sf.write('test_stream_out.wav', stream_out_full, sr)
    
    # Compare
    batch_out, _ = sf.read('test_batch_out.wav')
    
    min_len = min(len(batch_out), len(stream_out_full))
    diff = np.abs(batch_out[:min_len] - stream_out_full[:min_len]).max()
    
    print(f"Max difference between batch and stream: {diff}")
    
if __name__ == "__main__":
    test_stream_equivalence()

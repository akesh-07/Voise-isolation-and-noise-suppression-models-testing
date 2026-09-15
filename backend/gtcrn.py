import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
import os
import torch

class GTCRNInference:
    def __init__(self):
        base_dir = Path(__file__).parent.resolve()
        model_path = base_dir / "models" / "gtcrn" / "gtcrn_simple.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"GTCRN model not found at {model_path}. Please download it.")
            
        print(f"Loading GTCRN ONNX model from {model_path}")
        # Initialize singleton session
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        
        # Precompute the square-root Hann window for overlap-add
        self.window = torch.hann_window(512).pow(0.5).numpy()

    def process(self, audio_path: str, output_path: str):
        # 1. Load audio and enforce 16kHz mono using soundfile and librosa
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1) # Convert to mono
            
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        audio = audio.astype(np.float32)
        
        # 2. Setup recurrent caches for streaming ONNX
        conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)
        
        hop_length = 256
        win_length = 512
        
        # Pad audio to ensure overlap-add works correctly at boundaries
        padding = win_length - hop_length
        audio = np.pad(audio, (padding, padding), mode='constant')
        
        num_frames = (len(audio) - win_length) // hop_length + 1
        output_audio = np.zeros_like(audio)
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + win_length
            
            # Apply analysis window
            frame = audio[start:end] * self.window
            
            # FFT (Real to Complex)
            frame_tensor = torch.from_numpy(frame)
            stft_frame = torch.fft.rfft(frame_tensor, n=512)
            
            # Format to [1, 257, 1, 2] expected by GTCRN
            mix_real = stft_frame.real.numpy()
            mix_imag = stft_frame.imag.numpy()
            mix_np = np.stack([mix_real, mix_imag], axis=-1)
            mix_np = np.reshape(mix_np, (1, 257, 1, 2)).astype(np.float32)
            
            inputs = {
                'mix': mix_np,
                'conv_cache': conv_cache,
                'tra_cache': tra_cache,
                'inter_cache': inter_cache
            }
            
            # Streaming Inference Step
            out = self.session.run(None, inputs)
            enh, conv_cache, tra_cache, inter_cache = out
            
            # Parse output back to complex tensor
            enh_real = enh[0, :, 0, 0]
            enh_imag = enh[0, :, 0, 1]
            enh_complex = torch.complex(torch.from_numpy(enh_real), torch.from_numpy(enh_imag))
            
            # Inverse FFT
            enh_frame = torch.fft.irfft(enh_complex, n=512).numpy()
            
            # Synthesis window & Overlap-Add
            output_audio[start:end] += enh_frame * self.window
            
        # Remove padding
        output_audio = output_audio[padding : len(output_audio) - padding]
        
        # Save to disk
        sf.write(output_path, output_audio, 16000, format='wav')
        
        return output_path

class GTCRNStreamSession:
    def __init__(self, inference_engine: GTCRNInference):
        self.engine = inference_engine
        self.conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        self.tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        self.inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)
        
        self.prev_chunk = np.zeros(256, dtype=np.float32)
        self.overlap_buffer = np.zeros(256, dtype=np.float32)
        
    def process_chunk(self, new_chunk: np.ndarray) -> np.ndarray:
        assert len(new_chunk) == 256, f"Expected chunk length 256, got {len(new_chunk)}"
        
        # Form 512-sample frame
        frame = np.concatenate([self.prev_chunk, new_chunk])
        self.prev_chunk = new_chunk.copy()
        
        # Apply analysis window
        frame = frame * self.engine.window
        
        # FFT
        frame_tensor = torch.from_numpy(frame)
        stft_frame = torch.fft.rfft(frame_tensor, n=512)
        
        # Format for GTCRN input
        mix_real = stft_frame.real.numpy()
        mix_imag = stft_frame.imag.numpy()
        mix_np = np.stack([mix_real, mix_imag], axis=-1)
        mix_np = np.reshape(mix_np, (1, 257, 1, 2)).astype(np.float32)
        
        inputs = {
            'mix': mix_np,
            'conv_cache': self.conv_cache,
            'tra_cache': self.tra_cache,
            'inter_cache': self.inter_cache
        }
        
        # Streaming Inference Step
        out = self.engine.session.run(None, inputs)
        enh, self.conv_cache, self.tra_cache, self.inter_cache = out
        
        # Parse output back to complex tensor
        enh_real = enh[0, :, 0, 0]
        enh_imag = enh[0, :, 0, 1]
        enh_complex = torch.complex(torch.from_numpy(enh_real), torch.from_numpy(enh_imag))
        
        # Inverse FFT
        enh_frame = torch.fft.irfft(enh_complex, n=512).numpy()
        
        # Synthesis window & Overlap-Add
        enh_frame = enh_frame * self.engine.window
        
        out_chunk = enh_frame[:256] + self.overlap_buffer
        self.overlap_buffer = enh_frame[256:]
        
        return out_chunk

import torch
import torchaudio
import soundfile as sf
import numpy as np

def get_enrollment_segment(wav_path, target_sr=8000, duration_sec=5.0):
    """
    Load an audio file, detect speech using Silero VAD, and return the first robust
    continuous speech segment as the enrollment audio.
    Returns: (enrollment_waveform, start_sec, end_sec)
    """
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Read audio safely using soundfile
    audio_array, sr = sf.read(wav_path)
    wav = torch.from_numpy(audio_array).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.transpose(0, 1)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True) # mono
        
    if sr != 16000:
        resampler_16k = torchaudio.transforms.Resample(sr, 16000)
        wav_16k = resampler_16k(wav)
    else:
        wav_16k = wav

    # Get speech timestamps with a higher threshold to avoid noise
    timestamps = get_speech_timestamps(wav_16k.squeeze(0), model, sampling_rate=16000, threshold=0.6)
    
    if not timestamps:
        # fallback: take first N seconds
        start_samples = 0
        end_samples = int(target_sr * duration_sec)
    else:
        # Find the best segment: ideally the first one that is >= 2.0 seconds
        best_segment = None
        for ts in timestamps:
            duration = (ts['end'] - ts['start']) / 16000.0
            if duration >= 2.0:
                best_segment = ts
                break
                
        # If none >= 2.0s, pick the longest one available
        if best_segment is None:
            best_segment = max(timestamps, key=lambda t: t['end'] - t['start'])
            
        start_16k = best_segment['start']
        end_16k = best_segment['end']
        
        start_sec = start_16k / 16000.0
        end_sec = end_16k / 16000.0
        
        # Limit to duration_sec if it's too long
        if end_sec - start_sec > duration_sec:
            end_sec = start_sec + duration_sec
            
        start_samples = int(start_sec * target_sr)
        end_samples = int(end_sec * target_sr)

    # resample original to target_sr (8000)
    if sr != target_sr:
        resampler_target = torchaudio.transforms.Resample(sr, target_sr)
        wav_target = resampler_target(wav)
    else:
        wav_target = wav

    # slice
    if end_samples > wav_target.shape[1]:
        end_samples = wav_target.shape[1]
    
    enrollment_wav = wav_target[:, start_samples:end_samples]
    return enrollment_wav, start_samples / target_sr, end_samples / target_sr

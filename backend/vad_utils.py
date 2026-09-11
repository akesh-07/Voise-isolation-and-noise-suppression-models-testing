import torch
import torchaudio

def get_enrollment_segment(wav_path, target_sr=8000, duration_sec=5.0):
    """
    Load an audio file, detect speech using Silero VAD, and return the first N seconds
    of continuous speech as the enrollment audio.
    Returns: (enrollment_waveform, start_sec, end_sec)
    """
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Read audio and resample to 16000 for silero-vad
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True) # mono
    if sr != 16000:
        resampler_16k = torchaudio.transforms.Resample(sr, 16000)
        wav_16k = resampler_16k(wav)
    else:
        wav_16k = wav

    # Get speech timestamps
    timestamps = get_speech_timestamps(wav_16k.squeeze(0), model, sampling_rate=16000)
    
    if not timestamps:
        # fallback: take first N seconds
        start_samples = 0
        end_samples = int(target_sr * duration_sec)
    else:
        # find the first timestamp that can give us duration_sec of speech, or combine them?
        # Let's just start at the first timestamp, and take duration_sec.
        start_16k = timestamps[0]['start']
        start_sec = start_16k / 16000.0
        start_samples = int(start_sec * target_sr)
        end_samples = start_samples + int(target_sr * duration_sec)

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

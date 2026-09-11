import os
import sys
import torch
import torchaudio
from huggingface_hub import snapshot_download
import numpy as np

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEARVOICE_REPO_DIR = os.path.join(BASE_DIR, "ClearerVoice-Studio")
TSE_DIR = os.path.join(CLEARVOICE_REPO_DIR, "train", "target_speaker_extraction")

# Add the target_speaker_extraction path to sys.path so we can import its modules
if TSE_DIR not in sys.path:
    sys.path.insert(0, TSE_DIR)

from networks import network_wrapper

# Configuration classes to mock args for SpEx+
class NetworkAudioConfig:
    def __init__(self):
        self.backbone = 'SpEx-plus'
        self.L = 20
        self.N = 256
        self.X = 8
        self.R = 4
        self.B = 256
        self.H = 512
        self.P = 3
        self.norm = 'gLN'
        self.non_linear = 'relu'
        self.speakers = 101

class NetworkReferenceConfig:
    def __init__(self):
        self.cue = 'speech'

class ModelArgs:
    def __init__(self, device):
        self.network_audio = NetworkAudioConfig()
        self.network_reference = NetworkReferenceConfig()
        self.causal = 0
        self.device = device
        self.speaker_no = 2

class SpExPlusInference:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vad_model = None
        self.vad_utils = None
        self.target_sr = 8000
        
    def load_model(self):
        if self.model is not None:
            return

        print("Loading Silero VAD...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=False,
                                               onnx=False,
                                               trust_repo=True)
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        self.vad_model = self.vad_model.to(self.device)

        print("Downloading SpEx+ weights from Hugging Face...")
        checkpoint_dir = snapshot_download(repo_id="alibabasglab/log_wsj0-2mix_speech_SpEx-plus_2spk", 
                                           allow_patterns=["*.pt", "*.yaml"])
        
        # The checkpoint is usually inside the directory "checkpoints/log_wsj0-2mix_speech_SpEx-plus_2spk/"
        # Let's find the last_best_checkpoint.pt
        best_ckpt_path = None
        for root, dirs, files in os.walk(checkpoint_dir):
            if "last_best_checkpoint.pt" in files:
                best_ckpt_path = os.path.join(root, "last_best_checkpoint.pt")
                break
        
        if not best_ckpt_path:
            raise RuntimeError(f"Could not find last_best_checkpoint.pt in {checkpoint_dir}")

        print(f"Loading SpEx+ from {best_ckpt_path}...")
        args = ModelArgs(self.device)
        self.model = network_wrapper(args)
        
        # Load state dict
        checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Clean up "module." prefix if it was saved with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def find_enrollment_segment(self, audio_tensor, sr):
        # Silero VAD works best at 16000Hz. Let's resample to 16kHz for VAD if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio_16k = resampler(audio_tensor)
        else:
            audio_16k = audio_tensor

        # Ensure audio is 1D for VAD
        if audio_16k.dim() > 1:
            audio_16k = audio_16k.mean(dim=0)
            
        speech_timestamps = self.get_speech_timestamps(audio_16k, self.vad_model, sampling_rate=16000)
        
        if not speech_timestamps:
            raise ValueError("No speech detected in the audio for enrollment.")
            
        # Find the first sufficiently long segment (e.g., > 2 seconds), or just use the first one
        # VAD timestamps are in samples for 16kHz
        target_length_samples = 16000 * 2 # 2 seconds
        
        chosen_segment = None
        for ts in speech_timestamps:
            length = ts['end'] - ts['start']
            if length > target_length_samples:
                chosen_segment = ts
                break
                
        # Fallback to the longest segment if none is > 2 seconds
        if not chosen_segment:
            chosen_segment = max(speech_timestamps, key=lambda x: x['end'] - x['start'])
            
        # Convert 16kHz sample indices back to the original sr indices
        start_orig = int(chosen_segment['start'] * sr / 16000)
        end_orig = int(chosen_segment['end'] * sr / 16000)
        
        # Max length of enrollment audio in SpEx+ training is often around 4s-10s. Let's cap at 5 seconds.
        max_samples = sr * 5
        if end_orig - start_orig > max_samples:
            end_orig = start_orig + max_samples

        return start_orig, end_orig

    def process(self, audio_path):
        if self.model is None:
            self.load_model()
            
        # Load audio using soundfile directly to bypass torchaudio/torchcodec completely
        import soundfile as sf
        audio_array, sr = sf.read(audio_path)
        waveform = torch.from_numpy(audio_array).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)
        
        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Resample to 8000Hz for SpEx+
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            mixture = resampler(waveform)
        else:
            mixture = waveform

        # Find enrollment segment using original waveform to retain quality before 8k downsampling
        start_idx, end_idx = self.find_enrollment_segment(waveform, sr)
        
        # Calculate timestamps in seconds
        start_time = start_idx / sr
        end_time = end_idx / sr
        
        # Extract enrollment in 8kHz domain
        start_8k = int(start_time * self.target_sr)
        end_8k = int(end_time * self.target_sr)
        
        aux = mixture[:, start_8k:end_8k]
        
        # Prepare inputs for model
        # model expects mixture (B, T)
        # aux (B, T_aux)
        # aux_len (B,)
        mixture_input = mixture.to(self.device)
        aux_input = aux.to(self.device)
        aux_len_input = torch.tensor([aux_input.shape[1]], dtype=torch.long).to(self.device)
        speakers = torch.tensor([-1]).to(self.device) # -1 to bypass speaker classification accuracy calculation
        
        ref = (aux_input, aux_len_input, speakers)
        
        with torch.no_grad():
            output = self.model(mixture_input, ref)
            # output is ests (B, T)
            extracted_audio = output.cpu()
            
        return {
            "extracted_audio": extracted_audio,
            "enrollment_audio": aux.cpu(),
            "sample_rate": self.target_sr,
            "enrollment_start_time": start_time,
            "enrollment_end_time": end_time
        }
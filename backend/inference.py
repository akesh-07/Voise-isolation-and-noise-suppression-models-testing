import os
import sys
import torch
import torchaudio
import soundfile as sf
from huggingface_hub import snapshot_download
import numpy as np
import importlib.util

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEARVOICE_REPO_DIR = os.path.join(BASE_DIR, "ClearerVoice-Studio")

# 1. Load SpEx_plus directly from target_speaker_extraction to avoid namespace collisions
spex_path = os.path.join(CLEARVOICE_REPO_DIR, "train", "target_speaker_extraction", "models", "SpEx_plus", "SpEx_plus.py")
if not os.path.exists(spex_path):
    raise FileNotFoundError(f"SpEx+ model file not found at: {spex_path}")

spec = importlib.util.spec_from_file_location("spex_plus_module", spex_path)
spex_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spex_mod)
SpEx_plus = spex_mod.SpEx_plus

# 2. Add speech_enhancement to sys.path for MossFormerGAN_SE_16K
SE_DIR = os.path.join(CLEARVOICE_REPO_DIR, "train", "speech_enhancement")
if SE_DIR not in sys.path:
    sys.path.insert(0, SE_DIR)

from models.mossformer_gan.generator import MossFormerGAN_SE_16K

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

class SpExPlusWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sep_network = SpEx_plus(args)

    def forward(self, mixture, ref=None):
        aux, aux_len, speakers = ref
        aux = aux.to(self.args.device)
        aux_len = aux_len.to(self.args.device)
        speakers = speakers.to(self.args.device)

        ests, ests2, ests3, spk_pred = self.sep_network(mixture, aux, aux_len)
        if torch.sum(speakers) >= 0:
            return (ests, ests2, ests3, spk_pred, speakers)
        else:
            return ests

class NetworkSEConfig:
    def __init__(self):
        self.network = 'MossFormerGAN_SE_16K'
        self.mode = 'test'
        self.use_cuda = 1 if torch.cuda.is_available() else 0
        self.sampling_rate = 16000
        self.one_time_decode_length = 10
        self.decode_window = 1
        self.win_type = 'hamming'
        self.win_len = 400
        self.win_inc = 100
        self.fft_len = 400

# Self-contained MossFormerGAN decoding functions to avoid PESQ/external evaluation dependencies
def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)

def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)

def stft(x, args, center=False):
    window = torch.hamming_window(args.win_len, periodic=False).to(x.device)
    return torch.stft(x, args.fft_len, args.win_inc, args.win_len, center=center, window=window, return_complex=False)

def istft(x, args):
    window = torch.hamming_window(args.win_len, periodic=False).to(x.device)
    x_complex = torch.view_as_complex(x)
    return torch.istft(x_complex, n_fft=args.fft_len, hop_length=args.win_inc, win_length=args.win_len, window=window, center=False, return_complex=False)

def decode_mossformergan(model, device, inputs, args):
    window = args.sampling_rate * args.decode_window
    stride = int(window * 0.75)
    b, t = inputs.shape
    decode_do_segment = t > args.sampling_rate * args.one_time_decode_length
    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], 1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)

    inputs = torch.from_numpy(np.float32(inputs)).to(device)
    b, t = inputs.shape

    def _decode_chunk(chunk):
        input_len = chunk.size(-1)
        nframe = int(np.ceil(input_len / args.win_inc))
        padded_len = nframe * args.win_inc
        padding_len = padded_len - input_len
        chunk = torch.cat([chunk, chunk[:, :padding_len]], dim=-1)
        c = torch.sqrt(chunk.size(-1) / torch.sum((chunk ** 2.0), dim=-1))
        chunk = torch.transpose(chunk, 0, 1)
        chunk = torch.transpose(chunk * c, 0, 1)
        inputs_spec = stft(chunk, args, center=True).to(torch.float32)
        inputs_spec = power_compress(inputs_spec).permute(0, 1, 3, 2)
        out_list = model(inputs_spec)
        pred_real, pred_imag = out_list[0].permute(0, 1, 3, 2), out_list[1].permute(0, 1, 3, 2)
        pred_spec_uncompress = power_uncompress(pred_real, pred_imag).squeeze(1)
        outputs = istft(pred_spec_uncompress, args).squeeze(0) / c
        return outputs[:input_len].detach().cpu().numpy()

    if decode_do_segment:
        outputs = np.zeros(t)
        give_up_length = (window - stride) // 2
        current_idx = 0
        total_chunks = int(np.ceil((t - window) / stride)) + 1 if t > window else 1
        chunk_idx = 1
        
        while current_idx + window <= t:
            print(f"Decoding MossFormerGAN chunk {chunk_idx}/{total_chunks}...")
            tmp_input = inputs[:, current_idx:current_idx + window]
            tmp_output = _decode_chunk(tmp_input)
            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = tmp_output[:-give_up_length]
            else:
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = tmp_output[give_up_length:-give_up_length]
            current_idx += stride
            chunk_idx += 1
            
        return outputs
    else:
        return _decode_chunk(inputs)

class SpExPlusInference:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.se_model = None
        self.se_args = NetworkSEConfig()
        self.vad_model = None
        self.vad_utils = None
        self.target_sr = 8000
        
        # Initialize noisereduce/deepfilter logic locally
        try:
            from deepfilter import DeepFilterInference
            self.df_engine = DeepFilterInference()
        except ImportError:
            self.df_engine = None
            
        # Initialize GTCRN logic locally
        try:
            from gtcrn import GTCRNInference
            self.gtcrn_engine = GTCRNInference()
        except ImportError:
            self.gtcrn_engine = None
        
    def load_model(self):
        if self.model is not None and self.se_model is not None:
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
        
        best_ckpt_path = None
        for root, dirs, files in os.walk(checkpoint_dir):
            if "last_best_checkpoint.pt" in files:
                best_ckpt_path = os.path.join(root, "last_best_checkpoint.pt")
                break
        
        if not best_ckpt_path:
            raise RuntimeError(f"Could not find last_best_checkpoint.pt in {checkpoint_dir}")

        print(f"Loading SpEx+ from {best_ckpt_path}...")
        args = ModelArgs(self.device)
        self.model = SpExPlusWrapper(args)
        
        # Load state dict
        checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

        print("Downloading MossFormerGAN_SE_16K weights from Hugging Face...")
        se_ckpt_dir = snapshot_download(repo_id="alibabasglab/MossFormerGAN_SE_16K", allow_patterns=["*.pt", "*.yaml"])
        se_best_ckpt = None
        for root, dirs, files in os.walk(se_ckpt_dir):
            if "last_best_checkpoint.pt" in files:
                se_best_ckpt = os.path.join(root, "last_best_checkpoint.pt")
                break
                
        if not se_best_ckpt:
            raise RuntimeError(f"Could not find last_best_checkpoint.pt in {se_ckpt_dir}")
            
        print(f"Loading MossFormerGAN_SE_16K from {se_best_ckpt}...")
        self.se_model = MossFormerGAN_SE_16K(self.se_args).model
        
        se_checkpoint = torch.load(se_best_ckpt, map_location='cpu')
        if 'model' in se_checkpoint:
            se_state_dict = se_checkpoint['model']
        elif 'model_state_dict' in se_checkpoint:
            se_state_dict = se_checkpoint['model_state_dict']
        else:
            se_state_dict = se_checkpoint
            
        se_new_state_dict = {}
        for k, v in se_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            se_new_state_dict[name] = v
            
        self.se_model.load_state_dict(se_new_state_dict)
        self.se_model.to(self.device)
        self.se_model.eval()
        
        print("Models loaded successfully.")

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
            
        # Find the first sufficiently long continuous speech segment (>= 2 seconds)
        target_length_samples = 16000 * 2 # 2 seconds
        
        chosen_segment = None
        for ts in speech_timestamps:
            length = ts['end'] - ts['start']
            if length >= target_length_samples:
                chosen_segment = ts
                break
                
        # Fallback to the longest segment if none is >= 2 seconds
        if not chosen_segment:
            chosen_segment = max(speech_timestamps, key=lambda x: x['end'] - x['start'])
            
        # Convert 16kHz sample indices back to the original sr indices
        start_orig = int(chosen_segment['start'] * sr / 16000)
        end_orig = int(chosen_segment['end'] * sr / 16000)
        
        # Max length of enrollment audio in SpEx+ is capped at 5 seconds
        max_samples = sr * 5
        if end_orig - start_orig > max_samples:
            end_orig = start_orig + max_samples

        return start_orig, end_orig

    def process(self, audio_path: str, enable_enhancement=False, processing_mode="spex"):
        """
        Process the uploaded audio file:
        1. Read audio and use VAD to detect enrollment
        2. Run SpEx+ to extract the target speaker (8kHz)
        3. Optionally run MossFormerGAN_SE_16K to enhance the extracted speech (8k -> 16k -> 8k)
        """
        import time
        if self.model is None or self.se_model is None:
            self.load_model()
            
        # Load audio using soundfile directly to bypass torchaudio/torchcodec
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
            
        stage_times = {}

        # GTCRN mode (Independent Pipeline)
        if processing_mode == "gtcrn":
            t0 = time.time()
            if self.gtcrn_engine:
                import tempfile
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                with os.fdopen(fd, 'wb') as f:
                    pass
                self.gtcrn_engine.process(audio_path, temp_path)
                gtcrn_array, gtcrn_sr = sf.read(temp_path)
                gtcrn_waveform = torch.from_numpy(gtcrn_array).float()
                if gtcrn_waveform.dim() == 1:
                    gtcrn_waveform = gtcrn_waveform.unsqueeze(0)
                else:
                    gtcrn_waveform = gtcrn_waveform.transpose(0, 1)
                os.remove(temp_path)
            else:
                raise RuntimeError("GTCRN engine could not be loaded.")
            stage_times["gtcrn_time"] = time.time() - t0
            
            return {
                "sample_rate": 16000,
                "stage_times": stage_times,
                "extracted_audio": gtcrn_waveform.cpu()
            }
            
        # Pre-denoising stage for DeepFilterNet3
        denoised_waveform = waveform
        current_sr = sr
        if processing_mode == "deepfilter_spex":
            t0 = time.time()
            if self.df_engine:
                import tempfile
                fd, df_temp_path = tempfile.mkstemp(suffix=".wav")
                with os.fdopen(fd, 'wb') as f:
                    pass
                self.df_engine.process(audio_path, df_temp_path)
                denoised_array, denoised_sr = sf.read(df_temp_path)
                denoised_waveform = torch.from_numpy(denoised_array).float()
                current_sr = denoised_sr
                if denoised_waveform.dim() == 1:
                    denoised_waveform = denoised_waveform.unsqueeze(0)
                else:
                    denoised_waveform = denoised_waveform.transpose(0, 1)
                if denoised_waveform.size(0) > 1:
                    denoised_waveform = denoised_waveform.mean(dim=0, keepdim=True)
                os.remove(df_temp_path)
            stage_times["denoise_time"] = time.time() - t0
            
        # Find enrollment segment using the (possibly denoised) waveform
        t0 = time.time()
        start_idx, end_idx = self.find_enrollment_segment(denoised_waveform, current_sr)
        stage_times["vad_time"] = time.time() - t0
        
        # Calculate timestamps in seconds
        start_time = start_idx / current_sr
        end_time = end_idx / current_sr
        
        # Resample to 8000Hz for SpEx+
        if current_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=current_sr, new_freq=self.target_sr)
            mixture = resampler(denoised_waveform)
        else:
            mixture = denoised_waveform
            
        if sr != self.target_sr:
            resampler_orig = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            orig_resampled = resampler_orig(waveform)
        else:
            orig_resampled = waveform
        
        # Extract enrollment in 8kHz domain
        start_8k = int(start_time * self.target_sr)
        end_8k = int(end_time * self.target_sr)
        
        orig_aux = orig_resampled[:, start_8k:end_8k]
        
        # Use the original raw audio for SpEx+ isolation!
        # DeepFilterNet applies non-linear phase/spectral distortions that confuse SpEx+'s time-domain 
        # 1D-CNN encoder. By using the DeepFilterNet output exclusively for VAD timestamp detection, 
        # we ensure perfect enrollment, while giving SpEx+ the raw linear audio it requires to isolate properly.
        aux_input = orig_aux.to(self.device)
        if processing_mode == "deepfilter_spex":
            # For the mixture, passing the highly non-linear denoised audio breaks the isolation mask.
            # We must pass the original resampled audio to SpEx+.
            mixture_input = orig_resampled.to(self.device)
        else:
            mixture_input = mixture.to(self.device)
            
        aux_len_input = torch.tensor([aux_input.shape[1]], dtype=torch.long).to(self.device)
        speakers = torch.tensor([-1]).to(self.device) # -1 to bypass speaker classification accuracy calculation
        
        ref = (aux_input, aux_len_input, speakers)
        
        t0 = time.time()
        with torch.no_grad():
            output = self.model(mixture_input, ref)
            extracted_audio = output.cpu()
        stage_times["spex_time"] = time.time() - t0
            
        result_payload = {
            "sample_rate": self.target_sr,
            "enrollment_start_time": start_time,
            "enrollment_end_time": end_time,
            "stage_times": stage_times
        }
        
        if enable_enhancement or processing_mode == "mossformer":
            t0 = time.time()
            # First, upsample SpEx+ output (8kHz) to 16kHz
            resampler_16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
            enhanced_input_16k = resampler_16k(extracted_audio) # (1, time)
            enhanced_input_16k_np = enhanced_input_16k.numpy()
            
            with torch.no_grad():
                enhanced_output_16k_np = decode_mossformergan(self.se_model, self.device, enhanced_input_16k_np, self.se_args)
            
            # Resample enhanced output back to 8kHz
            enhanced_output_16k_tensor = torch.from_numpy(enhanced_output_16k_np).float()
            if enhanced_output_16k_tensor.dim() == 1:
                enhanced_output_16k_tensor = enhanced_output_16k_tensor.unsqueeze(0)
                
            resampler_8k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
            enhanced_audio_8k = resampler_8k(enhanced_output_16k_tensor)
            
            result_payload["enhanced_audio"] = enhanced_audio_8k.cpu()
            stage_times["enhancement_time"] = time.time() - t0
            
        result_payload["extracted_audio"] = extracted_audio
        result_payload["enrollment_audio"] = orig_aux.cpu()
        if processing_mode == "deepfilter_spex":
            result_payload["denoised_audio"] = mixture.cpu()
            
        return result_payload
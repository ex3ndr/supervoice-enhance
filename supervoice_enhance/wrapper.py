import torch
from .model import EnhanceModel
from .config import config
from .audio import spectogram

class SuperVoiceEnhance(torch.nn.Module):
    def __init__(self, flow, vocoder):
        super(SuperVoiceEnhance, self).__init__()
        self.diffusion = EnhanceModel(flow, config)
        self.vocoder = vocoder

    def enhance(self, waveform, *, steps = 8, alpha = None):

        # Convert to spectogram
        device = self._device()
        spec = self._do_spectogram(waveform)
        spec = spec.to(device)

        # Enhance
        spec = self._audio_normalize(spec).to(torch.float32)
        enhanced = self.diffusion.sample(source = spec.transpose(0, 1), steps = steps, alpha = alpha).transpose(0, 1)
        enhanced = self._audio_denormalize(enhanced).to(torch.float32)

        # Vocoder
        reconstructed = vocoder.generate(enhanced.unsqueeze(0)).squeeze(0)
        reconstructed.to(waveform.device)
        return reconstructed

    def _do_spectogram(self, waveform):
        return spectogram(waveform, config.audio.n_fft, config.audio.n_mels, config.audio.hop_size, config.audio.win_size, config.audio.mel_norm, config.audio.mel_scale, config.audio.sample_rate)
    
    def _audio_normalize(self, src):
        return (src - config.audio.norm_mean) / config.audio.norm_std

    def _audio_denormalize(self, src):
        return (src * config.audio.norm_std) + config.audio.norm_mean

    def _device(self):
        return next(self.parameters()).device

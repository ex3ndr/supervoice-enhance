import torch
from .model import EnhanceModel
from .config import config

class SuperVoiceEnhance(torch.nn.Module):
    def __init__(self, flow, vocoder):
        super(SupervoiceEnhance, self).__init__()
        self.diffusion = EnhanceModel(flow, config)
        self.vocoder = vocoder

    def enhance(self, waveform, *, steps = 8, alpha = None):

        # Convert to spectogram
        spec = spectogram(waveform, 
            n_fft = config.audio.n_fft, 
            n_mels = config.audio.n_mels, 
            n_hop = config.audio.hop_size, 
            n_window = config.audio.win_size,  
            mel_norm = config.audio.mel_norm, 
            mel_scale = config.audio.mel_scale, 
            sample_rate = config.audio.sample_rate
        )

        # Enhance
        spec = (spec - config.audio.norm_mean) / config.audio.norm_std # Normalize
        enhanced = self.diffusion.sample(source = spec.to(torch.float32), steps = steps, alpha = alpha)
        enhanced = ((enhanced * config.audio.norm_std) + config.audio.norm_mean).to(torch.float32) # Denormalize

        # Vocoder
        return vocoder.generate(enhanced)

from .misc import dict_to_object

config = dict_to_object({
    "audio": {
        "sample_rate": 24000,
        "n_mels": 100,
        "n_fft": 1024,
        "hop_size": 256,
        "win_size": 256 * 4,
        "mel_norm": "slaney",
        "mel_scale": "slaney",
        "norm_std": 2.2615,
        "norm_mean": -5.8843
    }
})
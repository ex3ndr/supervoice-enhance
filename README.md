# ✨ SuperVoice Enhance [BETA]

Enhancing diffusion neural network for a single speaker speech based on Speech Flow architecture. [Evaluation notebook](/eval.ipynb).

> [!IMPORTANT]  
> Network was trained using 5s intrevals, but it can work with any length of audio with slightly reduced quality.

# Features

* ⚡️ Restoring and improving audio
* 🎤 24khz mono audio
* 🚀 Can work directly with spectograms for speedup and tight pipelining
* 🤹‍♂️ Can work with unknown languages

https://github.com/ex3ndr/supervoice-enhance/assets/400659/c378a824-b6c0-4ec2-bee7-df5e27116c1e

# Usage

Supervoice Enhance consists of multiple networks, but they are all loaded using a single command and published using Torch Hub, so you can use it as follows:

```python
import torch
import torchaudio

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load(repo_or_dir='ex3ndr/supervoice-enhance', model='enhance', vocoder = True) # vocoder = False if you don't need vocoder
model.to(device)
model.eval()

# Load audio
def load_mono_audio(path):
    audio, sr = torchaudio.load(path)
    if sr != model.sample_rate:
        audio = torchaudio.transforms.Resample(sr, model.sample_rate)(audio)
        sr = model.sample_rate
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio[0]
audio = load_mono_audio("./eval/eval_2.wav")

# Enhance
enhanced = model.enhance(waveform = audio, steps = 8) # 8 is optimal, 32 is higer quality but sometimes it halluciantes
enhanced_spec = model.enhance(waveform = audio, steps = 8, vocoder = False) # Return spectogram without running vocoder

```

# License

MIT

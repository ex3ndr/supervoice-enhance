# ✨ Supervoice Enhance [BETA]

Enhancing diffusion neural network for a single speaker speech.

> [!IMPORTANT]  
> Network was trained using 5s intrevals, but it can work with any length of audio with slightly reduced quality.

# Features

* ⚡️ Restoring and improving audio
* 🎤 24khz mono audio
* 🤹‍♂️ Can work with unknown languages

# Usage
Supervoice Enhance consists of multiple networks, but they are all loaded using a single command and published using Torch Hub, so you can use it as follows:

```python
import torch
import torchaudio

# Load audio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enhance = torch.hub.load(repo_or_dir='ex3ndr/supervoice-enhance', model='enhance')
enhance.to(device)
enhance.eval()

# Enhance

```

# License

MIT
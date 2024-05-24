# ✨ Supervoice Enhance [BETA]

Enhancing diffusion neural network for a single speaker speech.

# Features

* ⚡️ Restoring and improving audio
* 🎤 16khz mono audio
* 🤹‍♂️ Can work with unknown languages

# Usage
Supervoice Enhance consists of multiple networks, but they are all loaded using a single command and published using Torch Hub, so you can use it as follows:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enhance = torch.hub.load(repo_or_dir='ex3ndr/supervoice-enhance', model='enhance')
enhance.to(device)
enhance.eval()
```

# License

MIT
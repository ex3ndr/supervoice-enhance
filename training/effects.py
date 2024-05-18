import torch
import torchaudio

def maybe(effect, p):
    return lambda: _resolve(effect) if random.random() < p else None

def one_of(*args):
    return lambda:_resolve(random.choice(list(args)))
import torchaudio

def do_reverbrate(waveforms, rir):
    assert len(waveforms.shape) == 1 # Only single dimension is allowed
    assert len(rir.shape) == 1 # Only single dimension is allowed

    # Source length
    source_len = waveforms.shape[0]

    # NOTE: THIS ALL NOT NEEDED for fftconvolve
    # Flip for convolution (we are adding previous values (aka "echo") for each point
    # rir = torch.flip(rir,[0])

    # Pad with zeros to match output time
    # waveforms = torch.cat((torch.zeros(rir.shape[0]-1,dtype=waveforms.dtype), waveforms), 0)

    # Calculate convolution
    waveforms = waveforms.unsqueeze(0).unsqueeze(0)
    rir = rir.unsqueeze(0).unsqueeze(0)
    # waveforms = torch.nn.functional.conv1d(waveforms, rir)
    waveforms = torchaudio.functional.fftconvolve(waveforms, rir)
    waveforms = waveforms.squeeze(dim=0).squeeze(dim=0)
    waveforms = waveforms[0:source_len]

    return waveforms
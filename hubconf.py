dependencies = ['torch', 'torchaudio']

def enhance(pretrained = True, vocoder = True):

    # Imports
    import torch
    import os
    from supervoice_enhance.model import EnhanceModel
    from supervoice_enhance.wrapper import SuperVoiceEnhance
    from supervoice_enhance.config import config

    # Load vocoder
    if vocoder is False:
        vocoder = None
    elif vocoder is True:
        vocoder = torch.hub.load(repo_or_dir='ex3ndr/supervoice-vocoder', model='bigvsan')

    # Model
    flow = torch.hub.load(repo_or_dir='ex3ndr/supervoice-flow', model='flow')
    model = SuperVoiceEnhance(flow, vocoder)

    # Load checkpoint
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/supervoice-enhance-60000.pt", map_location="cpu")
        model.diffusion.load_state_dict(checkpoint['model'])

    return model
            
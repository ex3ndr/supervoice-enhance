import torch
from torchdiffeq import odeint

class EnhanceModel(torch.nn.Module):
    def __init__(self, flow, config):
        super(EnhanceModel, self).__init__()
        self.flow = flow
        self.config = config

    def sample(self, *, source, steps, alpha = None, return_trajectory = False):
        
        #
        # Prepare
        #

        # Create noise
        noise = torch.randn_like(source)

        # Create time interpolation
        times = torch.linspace(0, 1, steps, device = source.device)

        #
        # Solver
        # 

        # Overwrite audio segment with predicted audio according to mask
        def merge_predicted(predicted):
            if mask is None:
                return predicted
            return merge_mask(source = source_audio, replacement = predicted, mask = mask)

        def solver(t, z):

            # If alpha is not provided
            if alpha is None:
                return self.forward(source = source.unsqueeze(0), noise = z.unsqueeze(0), times = t.unsqueeze(0)).squeeze(0)

            # If alpha is provided - zero out tokens and audio and mix together
            audio_empty = torch.zeros_like(source)

            # Mix together
            audio_t = torch.stack([audio_empty, source], dim = 0)
            noise_t = torch.stack([z, z], dim = 0) # Just double it
            t_t = torch.stack([t, t], dim = 0) # Just double it

            # Inference
            predicted_mix = self.forward(
                audio = audio_t, 
                noise = noise_t, 
                times = t_t
            )
            predicted_conditioned = predicted_mix[1]
            predicted_unconditioned = predicted_mix[0]
            
            # CFG prediction

            # There are different ways to do CFG, this is my very naive version, which worked for me:
            # prediction = (1 + alpha) * predicted_conditioned - alpha * predicted_unconditioned

            # Original paper uses a different one, but i found that it simply creates overexposed values
            # prediction = predicted_unconditioned + (predicted_conditioned - predicted_unconditioned) * alpha

            # This is from the latest paper that rescales original formula (https://arxiv.org/abs/2305.08891):
            prediction = predicted_conditioned + (predicted_conditioned - predicted_unconditioned) * alpha
            prediction_rescaled = predicted_conditioned.std() * (prediction / prediction.std())

            return prediction


        trajectory = odeint(solver, noise, times, atol = 1e-5, rtol = 1e-5, method = 'midpoint')

        #
        # Output sample and full trajectory
        #

        if return_trajectory:
            return trajectory[-1], trajectory
        else:
            return trajectory[-1]

    def forward(self, *, source, noise, times, target = None):
        return self.flow(
            audio = source,
            noise = noise,
            times = times,
            mask = None,
            target = target
        )
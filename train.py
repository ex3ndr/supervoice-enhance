# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

# Base
import itertools
from glob import glob
import textgrid
from tqdm import tqdm
import time
from contextlib import nullcontext
import shutil
from pathlib import Path
import math
import random
from tqdm import tqdm

# ML
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import wandb

# Local
from supervoice_enhance.config import config
from supervoice_enhance.model import EnhanceModel
from training.tensors import probability_binary_mask, drop_using_mask
from training.dataset import load_distorted_loader

# Train parameters
train_experiment = "ft-05"
train_project="supervoice-enhance"
train_datasets = ["./external_datasets/hifi-tts/audio"]
train_eval_datasets = ["./external_datasets/libritts-r/test-clean/"]
train_duration = 10
train_source_experiment = None
train_auto_resume = True
train_batch_size = 5 # Per GPU
train_drop_prob = 0.3
train_grad_accum_every = 1
train_steps = 60000
train_loader_workers = 5
train_log_every = 1
train_save_every = 1000
train_watch_every = 1000
train_evaluate_every = 1
train_evaluate_batch_size = 10
train_lr_start = 1e-7
train_lr_max = 2e-5
train_warmup_steps = 5000
train_mixed_precision = "fp16" # "bf16" or "fp16" or None
train_clip_grad_norm = 0.2
train_sigma = 1e-5

# Train
def main():

    # Prepare accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps = train_grad_accum_every, mixed_precision=train_mixed_precision)
    device = accelerator.device
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if train_mixed_precision == "fp16" else (torch.bfloat16 if train_mixed_precision == "bf16" else torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 
    # set_seed(42) enabling this would force each GPU to have same samples
    lr_start = train_lr_start * accelerator.num_processes
    lr_max = train_lr_max * accelerator.num_processes

    # Prepare dataset
    accelerator.print("Loading dataset...")
    train_loader = load_distorted_loader(datasets = train_datasets, duration = train_duration, num_workers = train_loader_workers, batch_size = train_batch_size)
    test_loader = load_distorted_loader(datasets = train_eval_datasets, duration = train_duration, num_workers = train_loader_workers, batch_size = train_batch_size)

    # Prepare model
    accelerator.print("Loading model...")
    step = 0

    # Model
    flow = torch.hub.load(repo_or_dir='ex3ndr/supervoice-flow', model='flow')
    raw_model = EnhanceModel(flow, config)
    model = raw_model
    wd_params, no_wd_params = [], []
    for param in model.parameters():
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    optim = torch.optim.AdamW([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], lr_max, betas=[0.9, 0.99], weight_decay=0.01, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = train_steps)

    # Accelerate
    model, optim, train_loader, test_loader = accelerator.prepare(model, optim, train_loader, test_loader)
    train_cycle = cycle(train_loader)
    test_cycle = cycle(test_loader)
    test_batch = next(test_cycle)
    hps = {
        "train_lr_start": train_lr_start, 
        "train_lr_max": train_lr_max, 
        "batch_size": train_batch_size, 
        "grad_accum_every": train_grad_accum_every,
        "steps": train_steps, 
        "warmup_steps": train_warmup_steps,
        "mixed_precision": train_mixed_precision,
        "clip_grad_norm": train_clip_grad_norm,
    }
    accelerator.init_trackers(train_project, config=hps)
    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=train_watch_every * train_grad_accum_every)

    # Save
    def save():
        
        # Save step checkpoint
        fname = str(output_dir / f"{train_experiment}.pt")
        fname_step = str(output_dir / f"{train_experiment}.{step}.pt")
        torch.save({

            # Model
            'model': raw_model.state_dict(), 

            # Optimizer
            'step': step,
            'optimizer': optim.state_dict(), 
            'scheduler': scheduler.state_dict(),

        },  fname_step)

        # Overwrite main checkpoint
        shutil.copyfile(fname_step, fname)

    # Load
    source = None
    if (output_dir / f"{train_experiment}.pt").exists():
        source = train_experiment
    elif train_source_experiment and (output_dir / f"{train_source_experiment}.pt").exists():
        source = train_source_experiment

    if train_auto_resume and source is not None:
        accelerator.print("Resuming training...")
        checkpoint = torch.load(str(output_dir / f"{source}.pt"), map_location="cpu")

        # Model
        raw_model.load_state_dict(checkpoint['model'])

        # Optimizer
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        step = checkpoint['step']

        accelerator.print(f'Loaded at #{step}')
        

    # Train step
    def train_step():
        model.train()

        # Update LR
        if step < train_warmup_steps:
            lr = (lr_start + ((lr_max - lr_start) * step) / train_warmup_steps)
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            lr = lr / accelerator.num_processes
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0] / accelerator.num_processes

        # Load batch
        successful_cycles = 0
        failed_steps = 0
        while successful_cycles < train_grad_accum_every:
            with accelerator.accumulate(model):
                with accelerator.autocast():

                    # Load batch
                    spec, spec_aug = next(train_cycle)

                    # Prepare batch
                    batch_size = spec.shape[0]
                    seq_len = spec.shape[1]

                    # Normalize spectograms
                    spec = (spec - config.audio.norm_mean) / config.audio.norm_std
                    spec_aug = (spec_aug - config.audio.norm_mean) / config.audio.norm_std

                    # Prepare target flow (CFM)
                    times = torch.rand((batch_size,), dtype = spec.dtype, device = device)
                    t = rearrange(times, 'b -> b 1 1')
                    source_noise = torch.randn_like(spec, device=device)
                    noise = (1 - (1 - train_sigma) * t) * source_noise + t * spec
                    flow = spec - (1 - train_sigma) * source_noise

                    # Drop mask
                    if train_drop_prob > 0:
                        drop_mask = probability_binary_mask(shape = (batch_size,), true_prob = train_drop_prob, device = device)
                        spec_aug = drop_using_mask(source = spec_aug, replacement = 0, mask = drop_mask)

                    # Train step
                    predicted, loss = model(source = spec_aug, noise = noise, times = times, target = flow)

                    # Backprop
                    optim.zero_grad()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), train_clip_grad_norm)
                    optim.step()

                    # Log skipping step
                    if optim.step_was_skipped:
                        failed_steps = failed_steps + 1
                        if torch.isnan(loss).any():
                            accelerator.print("Step was skipped with NaN loss")
                        else:
                            accelerator.print("Step was skipped")
                        if failed_steps > 20:
                            raise Exception("Too many failed steps")
                    else:
                        successful_cycles = successful_cycles + 1
                        failed_steps = 0

        return loss, predicted, flow, lr

    #
    # Start Training
    #

    accelerator.print("Training started at step", step)
    while step < train_steps:
        start = time.time()
        loss, predicted, flow, lr = train_step()
        end = time.time()

        # Advance
        step = step + 1

        # Summary
        if step % train_log_every == 0 and accelerator.is_main_process:
            accelerator.log({
                "learning_rate": lr,
                "loss": loss,
                "predicted/mean": predicted.mean(),
                "predicted/max": predicted.max(),
                "predicted/min": predicted.min(),
                "target/mean": flow.mean(),
                "target/max": flow.max(),
                "target/min": flow.min()
            }, step=step)
            accelerator.print(f'Step {step}: loss={loss}, lr={lr}, time={end - start} sec')
        
        # Save
        if step % train_save_every == 0 and accelerator.is_main_process:
            save()

    # End training
    if accelerator.is_main_process:
        accelerator.print("Finishing training...")
        save()
    accelerator.end_training()
    accelerator.print('✨ Training complete!')

#
# Utility
#

def cycle(dl):
    while True:
        for data in dl:
            yield data    

if __name__ == "__main__":
    main()
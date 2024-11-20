# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from os import path

from tqdm import tqdm
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.optim import AdamW, Adam
from torch import autocast, GradScaler

from omegaconf import OmegaConf, open_dict

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


#%% # My block

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print("All seeds set to", seed)
    
seed_everything(42)

disable_tqdm = True
warnings.filterwarnings("ignore")

# %%
# additional packages
# !pip install hiera-transformer
# !pip install -U pytorch_warmup

# %% [markdown]
# # Hyperparameters

# %%
video_size = [36, 64]
batchsize=16

screen_chunk_size = 30
screen_sampling_rate = 30

response_chunk_size = 8
response_sampling_rate = 8

behavior_as_channels = True
replace_nans_with_means = True

dim_head = 64
num_heads = 2
drop_path_rate = 0
mlp_ratio=4

# %% [markdown]
# ### get dataloaders

# %%
paths = [
        #  'dynamic29513-3-5-Video-full',
        #  'dynamic29514-2-9-Video-full',
        #  'dynamic29755-2-8-Video-full',
        #  'dynamic29647-19-8-Video-full',
         'dynamic29156-11-10-Video-full',
        #  'dynamic29623-4-9-Video-full',
        #  'dynamic29515-10-12-Video-full',
        #  'dynamic29234-6-9-Video-full',
        #  'dynamic29712-5-9-Video-full',
        #  'dynamic29228-2-10-Video-full'
        ]

full_paths = [path.join("/home/gchakrabarty/video_foundation_model/data", f) for f in paths]

# %%
from experanto.dataloaders import get_multisession_dataloader
from experanto.configs import DEFAULT_CONFIG as cfg

# %%
cfg.dataset.global_chunk_size = None
cfg.dataset.global_sampling_rate = None

cfg.dataset.modality_config.screen.chunk_size = screen_chunk_size
cfg.dataset.modality_config.screen.sampling_rate = screen_sampling_rate
cfg.dataset.modality_config.eye_tracker.chunk_size = screen_chunk_size
cfg.dataset.modality_config.eye_tracker.sampling_rate = screen_sampling_rate
cfg.dataset.modality_config.treadmill.chunk_size = screen_chunk_size
cfg.dataset.modality_config.treadmill.sampling_rate = screen_sampling_rate

cfg.dataset.modality_config.responses.chunk_size = response_chunk_size
cfg.dataset.modality_config.responses.sampling_rate = response_sampling_rate

cfg.dataset.modality_config.screen.sample_stride = 1
cfg.dataset.modality_config.screen.include_blanks=True
cfg.dataset.modality_config.screen.valid_condition = {"tier": "train"}
cfg.dataset.modality_config.screen.transforms.Resize.size = video_size

cfg.dataloader.num_workers = 2
cfg.dataloader.prefetch_factor = 2
cfg.dataloader.batch_size = batchsize
cfg.dataloader.pin_memory = False
cfg.dataloader.shuffle = True

train_dl = get_multisession_dataloader(full_paths, cfg)

# %% [markdown]
# ### get Hiera backbone

# %%
# pip install hiera-transformer
from hiera import Hiera
tiny_hiera = Hiera(input_size=(screen_chunk_size, video_size[0], video_size[1]),
                     num_heads=1,
                     embed_dim=96,
                     stages=(2, 1,), # 3 transformer layers 
                     q_pool=1, 
                     in_chans=1,
                     q_stride=(1, 1, 1,),
                     mask_unit_size=(1, 8, 8),
                     patch_kernel=(5, 5, 5),
                     patch_stride=(3, 2, 2),
                     patch_padding=(1, 2, 2),
                     sep_pos_embed=True,
                     drop_path_rate=drop_path_rate,
                     mlp_ratio=4,)

tiny_hiera = tiny_hiera.cuda().to(torch.float32);
example_input = torch.ones(8,1,screen_chunk_size, 36,64).to("cuda", torch.float32)
out = tiny_hiera(example_input, return_intermediates=True);

hiera_output = out[-1][-1]
hiera_output.shape # (b, t, h, w, c): (8, 4, 9, 16, 192)

# %% [markdown]
# # Model definition

# %%
class MouseHiera(nn.Module):
    def __init__(self,
                 backbone,
                 dls,
                 chunk_size,
                 dim=192,
                 dim_head=32,
                 num_heads=4,
                 mlp_ratio=4,):
        super().__init__()
        self.backbone=backbone
        self.num_heads=num_heads
        self.dim_head=dim_head
        self.dim=dim
        self.dim_q = dim_head*num_heads
        self.wq = nn.Linear(self.dim_q, self.dim_q, bias=False)
        self.wk = nn.Linear(dim, self.dim_q, bias=False)
        self.wv = nn.Linear(dim, self.dim_q, bias=False)
        self.wo = nn.Linear(self.dim_q, self.dim_q, bias=False)
        
        self.neuron_proj = nn.Linear(self.dim_q, chunk_size, bias=False)
        
        
        self.kv_norm=torch.nn.RMSNorm(dim)
        self.q_norm=torch.nn.RMSNorm(self.dim_q)
        self.qkv_norm=torch.nn.RMSNorm(self.dim_q)
        self.mlp = MLP(dim=self.dim_q, hidden_dim=int(self.dim_q * mlp_ratio))
        self.readout = nn.ModuleDict()
        self.activation = nn.Softplus(beta=0.1) # probably a much better activation than ELU+1
        for k, v in dls.loaders.items():
            n_neurons = next(iter(v))["responses"].shape[-1]
            self.readout[k] = IndexedLinearReadout(n_neurons, 
                                                   in_features=dim_head*num_heads,
                                                   dim_head=dim_head, 
                                                   num_heads=num_heads, 
                                                  )
        self.init_weights()

    def forward(self, x, key):
        x = self.backbone(x, return_intermediates=True)[-1][-1]
        b, t, h, w, d = x.shape
        x = self.kv_norm(x)
        x = x.view(b, -1, d) # (B, t*h*w, D)
        k, v = self.wk(x), self.wv(x)
        q = self.q_norm(self.readout[key].query) # (1, N, D)
        q = q.repeat(b, 1, 1) # repeat query for number of batches
        q_attn = self.wq(q)
        q_attn = q_attn.view(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.dim_head).transpose(1, 2) # (B, H, S, D)
        v = v.view(b, -1, self.num_heads, self.dim_head).transpose(1, 2) # (B, H, S, D)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            o = F.scaled_dot_product_attention(q_attn, k, v)
        # (B, H, S, D) -> (B, N, D), with N = num_neurons
        o = o.transpose(1,2).contiguous().view(b, -1, self.dim_q)
        o = self.wo(o) + q
        o = self.qkv_norm(o)  
        o = self.mlp(o) + o
        o = self.neuron_proj(o) # (B, N, D) -> (B, N, t)
        o = o + self.readout[key].bias
        o = self.activation(o)
        return o
     
    def init_weights(self, std=.5, cutoff_factor: int = 3):
        """See `TorchTitan <https://github.com/pytorch/torchtitan/blob/40a10263c5b3468ffa53b3ac98d80c9267d68155/torchtitan/models/llama/model.py#L403>`__."""
        std = self.dim_q**-0.5
        for lin in (self.wq, self.wk, self.wv, self.wo):
            nn.init.trunc_normal_(
                lin.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

# %%
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.init_weights()

    def forward(self, x):
        return self.net(x)
        
    def init_weights(self, std=.5, cutoff_factor: int = 3):
        """See `TorchTitan <https://github.com/pytorch/torchtitan/blob/40a10263c5b3468ffa53b3ac98d80c9267d68155/torchtitan/models/llama/model.py#L403>`__."""
        nn.init.trunc_normal_(
            self.net[0].weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
        nn.init.trunc_normal_(
            self.net[2].weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
        self.net[0].bias.data.zero_()
        self.net[2].bias.data.zero_()
        

# %%
class IndexedLinearReadout(nn.Module):
    """
    Readout module for MTM models with selectable weights based on 
    input IDs. Based on :class:`torch.nn.Linear`.
    """
    def __init__(
        self,
        unique_ids: int,
        in_features: int = 384,
        dim_head=32,
        num_heads=4,
        bias: bool = True,
        device="cuda",
        dtype=torch.float32,
        init_std: float = 0.02,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.unique_ids = unique_ids
        self.in_features = in_features
        self.init_std = init_std
        self.query = nn.Parameter(
            torch.empty(1, unique_ids, dim_head*num_heads, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(1, unique_ids, 1, **factory_kwargs)
            )
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self, cutoff_factor: int = 3):
        """See `TorchTitan <https://github.com/pytorch/torchtitan/blob/40a10263c5b3468ffa53b3ac98d80c9267d68155/torchtitan/models/llama/model.py#L403>`__."""
        readout_std = self.in_features**-0.5
        nn.init.trunc_normal_(
            self.query,
            mean=0.0,
            std=readout_std,
            a=-cutoff_factor * readout_std,
            b=cutoff_factor * readout_std,
        )
        if self.bias is not None:
            self.bias.data.zero_()

# %% [markdown]
# ### Build Model

# %%
backbone_dim = hiera_output[-1][-1].shape[-1]
model = MouseHiera(backbone=tiny_hiera, 
                        dls=train_dl, 
                        chunk_size=response_chunk_size,
                        dim=backbone_dim, 
                        dim_head=dim_head,
                        num_heads=num_heads,
                       mlp_ratio=mlp_ratio)
model = model.to(torch.float32).cuda();

# %% [markdown]
# # Trainer

# %%
# pip install -U pytorch_warmup
import pytorch_warmup as warmup

n_epochs = 200
lr = 2e-4
gradient_clipping = 1.0
criteria = nn.PoissonNLLLoss(log_input=False, reduction='mean')
opt = AdamW(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                          T_max=1e6, 
                                                          eta_min=1e-5)
warmup_scheduler = warmup.UntunedLinearWarmup(opt)


# %% [markdown]
# # train

# %%
# the first 10 batches are slow because torch is compiling the model for each new input shape
for _ in range(n_epochs):
    for i, (key, batch) in tqdm(enumerate(train_dl), disable=disable_tqdm):
        videos = batch["screen"].to("cuda", torch.float32, non_blocking=True).transpose(1,2)
        responses = batch["responses"].to("cuda", torch.float32, non_blocking=True)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(videos, key);
        loss = criteria(out.transpose(1,2), responses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping, norm_type=2)
        opt.step()
        opt.zero_grad()
        with warmup_scheduler.dampening():
            lr_scheduler.step()

        # if i == 10:
        #     break
    print(f"Loss:{loss}")
    # %% [markdown]
    # # Validation dataloader

    # %%
    cfg.dataset.modality_config.screen.sample_stride = screen_chunk_size
    cfg.dataset.modality_config.screen.include_blanks=False
    cfg.dataset.modality_config.screen.valid_condition = {"tier": "oracle"}

    # example session
    val_dl = get_multisession_dataloader(full_paths[0:1], cfg)

    _, b = next(iter(val_dl))
    n_neurons = b["responses"].shape[-1]

    # %%
    def val_step():
        from torchmetrics.regression import PearsonCorrCoef
        r = PearsonCorrCoef(num_outputs=n_neurons).cuda()
        with torch.no_grad():
            for i, (k, b) in tqdm(enumerate(val_dl), disable=disable_tqdm):
                videos = b["screen"].to("cuda", torch.float32, non_blocking=True).permute(0,2,1,3,4)
                responses = b["responses"].to("cuda", torch.float32, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(videos, k);
                r.update(out.transpose(1,2).reshape(-1, n_neurons), responses.reshape(-1, n_neurons))
        return r.compute().cpu().numpy().mean()

    # %%
    print(val_step())

    # %% [markdown]
    # # Final Test Set Score

    # %%
    cfg.dataset.modality_config.screen.sample_stride = screen_chunk_size
    cfg.dataset.modality_config.screen.include_blanks=False
    cfg.dataset.modality_config.screen.valid_condition = {"tier": "final_test_1"}

    # example session
    val_dl = get_multisession_dataloader(full_paths[0:1], cfg)

    _, b = next(iter(val_dl))
    n_neurons = b["responses"].shape[-1]

    # %%
    def test_step():
        from torchmetrics.regression import PearsonCorrCoef
        r = PearsonCorrCoef(num_outputs=n_neurons).cuda()
        with torch.no_grad():
            for i, (k, b) in tqdm(enumerate(val_dl), disable=disable_tqdm):
                videos = b["screen"].to("cuda", torch.float32, non_blocking=True).permute(0,2,1,3,4)
                responses = b["responses"].to("cuda", torch.float32, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(videos, k);
                r.update(out.transpose(1,2).reshape(-1, n_neurons), responses.reshape(-1, n_neurons))
        return r.compute().cpu().numpy().mean()

    # %%
    print(test_step())

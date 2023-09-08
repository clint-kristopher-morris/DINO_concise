import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import datasets, transforms

import sys
dir_root = "/gv1/projects/REMAPS/remaps/Development/monodepth2/monodepth2-master/monodepth2-master/Clints/REMAPS/DINO/dino-concise"
sys.path.append(f"{dir_root}/dino")
from dino.vision_transformer import DINOHead, vit_small, vit_tiny, vit_base, vit_tinyer, vit_tiniest
from dino import utils


from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce


# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None
        
    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None, return_attention=False):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)
        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        if return_attention:
            return attn
        
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

    
# main class
class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 64,
        latent_dim = 64,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True
        
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels
        
        self.embed_dim = latent_dim*num_latents

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Identity()


    def forward(self, data, mask=None, return_embeddings=False, return_last_cross_attention=False):
        b, channel_dim, height_dim, width_dim = data.shape

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axes
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=data.device, dtype=data.dtype), [height_dim, width_dim]))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)

            # Expand the fourier encodings to batch size
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            # Reshape enc_pos to merge the last two dimensions
            enc_pos = enc_pos.reshape(b, height_dim, width_dim, -1)
            # Move the Fourier channels to the channel dimension for concatenation
            enc_pos = rearrange(enc_pos, 'b h w c -> b c h w')
            # Concatenate data and Fourier encodings
            data = torch.cat((data, enc_pos), dim=1)

            # Update the dimensions after Fourier encoding
            _, channel_dim, height_dim, width_dim = data.shape

        # Reshape data
        data = data.transpose(1, 2).transpose(2, 3).reshape(b, height_dim * width_dim, channel_dim)
        x = repeat(self.latents, 'n d -> b n d', b=b)

        # layers
        for idx, (cross_attn, cross_ff, self_attns) in enumerate(self.layers):
            # normal loop
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings
        if return_last_cross_attention:
            return x

        # to logits
        x = self.to_logits(x)
        return x
    
    def get_last_selfattention(self, data):
        return self.forward(data, mask=None, return_embeddings=False, return_last_cross_attention=False)

    
def perceiver_base(patch_size=16, fourier_encode_data=True, dim=32, **kwargs):
    model = model = Perceiver(
    num_freq_bands=64,  # This is typically half the size of the image, but can be tuned
    depth=3,  # Equivalent to the depth of VisionTransformer
    max_freq=patch_size,  # This is a hyperparameter, can be tuned. Setting it equal to patch_size as a starting point
    input_channels=3,  # Assuming images have 3 channels (RGB). Adjust if different
    input_axis=2,  # For images
    num_latents=dim,  # A hyperparameter. Adjust based on the complexity of your data
    latent_dim=dim,  # Equivalent to embed_dim of VisionTransformer
    cross_heads=1,  # Equivalent to num_heads of VisionTransformer
    latent_heads=8,  # Equivalent to num_heads of VisionTransformer
    cross_dim_head=128,  # A hyperparameter. Adjust based on your requirements
    latent_dim_head=128,  # A hyperparameter. Adjust based on your requirements
    num_classes=10,  # Adjust based on your classification task
    attn_dropout=0.4,  # Dropout for attention. Adjust based on your requirements
    ff_dropout=0.4,  # Dropout for feed-forward networks. Adjust based on your requirements
    weight_tie_layers=True,  # Whether to share weights across layers
    fourier_encode_data=fourier_encode_data,  # Whether to Fourier encode the data
    self_per_cross_attn=3  # Number of self-attention layers per cross-attention layer
    )
    return model


def perceiver_light(patch_size=16, fourier_encode_data=True, dim=32, depth=3, weight_tie_layers=False, **kwargs):
    model = model = Perceiver(
    num_freq_bands=64,  # This is typically half the size of the image, but can be tuned
    depth=depth,  # Equivalent to the depth of VisionTransformer
    max_freq=patch_size,  # This is a hyperparameter, can be tuned. Setting it equal to patch_size as a starting point
    input_channels=3,  # Assuming images have 3 channels (RGB). Adjust if different
    input_axis=2,  # For images
    num_latents=dim,  # A hyperparameter. Adjust based on the complexity of your data
    latent_dim=dim,  # Equivalent to embed_dim of VisionTransformer
    cross_heads=1,  # Equivalent to num_heads of VisionTransformer
    latent_heads=8,  # Equivalent to num_heads of VisionTransformer
    cross_dim_head=128,  # A hyperparameter. Adjust based on your requirements
    latent_dim_head=128,  # A hyperparameter. Adjust based on your requirements
    num_classes=10,  # Adjust based on your classification task
    attn_dropout=0.4,  # Dropout for attention. Adjust based on your requirements
    ff_dropout=0.4,  # Dropout for feed-forward networks. Adjust based on your requirements
    weight_tie_layers=weight_tie_layers,  # Whether to share weights across layers
    fourier_encode_data=fourier_encode_data,  # Whether to Fourier encode the data
    self_per_cross_attn=3  # Number of self-attention layers per cross-attention layer
    )
    return model


def perceiver_mid(patch_size=16, fourier_encode_data=True, dim=32, depth=3, weight_tie_layers=False, **kwargs):
    model = model = Perceiver(
    num_freq_bands=6,  # This is typically half the size of the image, but can be tuned
    depth=depth,  # Equivalent to the depth of VisionTransformer
    max_freq=patch_size,  # This is a hyperparameter, can be tuned. Setting it equal to patch_size as a starting point
    input_channels=3,  # Assuming images have 3 channels (RGB). Adjust if different
    input_axis=2,  # For images
    num_latents=dim,  # A hyperparameter. Adjust based on the complexity of your data
    latent_dim=dim,  # Equivalent to embed_dim of VisionTransformer
    cross_heads=8,  # Equivalent to num_heads of VisionTransformer
    latent_heads=8,  # Equivalent to num_heads of VisionTransformer
    cross_dim_head=256,  # A hyperparameter. Adjust based on your requirements
    latent_dim_head=256,  # A hyperparameter. Adjust based on your requirements
    num_classes=10,  # Adjust based on your classification task
    attn_dropout=0.4,  # Dropout for attention. Adjust based on your requirements
    ff_dropout=0.4,  # Dropout for feed-forward networks. Adjust based on your requirements
    weight_tie_layers=weight_tie_layers,  # Whether to share weights across layers
    fourier_encode_data=fourier_encode_data,  # Whether to Fourier encode the data
    self_per_cross_attn=3  # Number of self-attention layers per cross-attention layer
    )
    return model


"""
DINO code:
"""

class DINO(nn.Module):
    def __init__(self, out_dim=65536, use_bn=False, model_type="tiny"):
        super().__init__()
        model_map = {'tiny':vit_tiny(), 'small':vit_small(), 'base':vit_base(), 
                     
                     'tinyer': vit_tinyer(), 'tiniest': vit_tiniest(), 'tinyer8':vit_tinyer(patch_size=8),
                     'tinyer12':vit_tinyer(patch_size=12),
                     
                     'perceiver':perceiver_base(),
                     'perceiver_v1':perceiver_base(fourier_encode_data=False, dim=8),
                     'perceiver_v2':perceiver_base(fourier_encode_data=True, dim=16),
                     'perceiver_v3':perceiver_base(fourier_encode_data=True, dim=32),
                     'perceiver_v4':perceiver_light(fourier_encode_data=True, dim=8, weight_tie_layers=False),
                     'perceiver_v5':perceiver_light(fourier_encode_data=True, dim=128, weight_tie_layers=True, depth=2),
                     'perceiver_m1':perceiver_mid(fourier_encode_data=True, dim=128, weight_tie_layers=True, depth=3),
                    }
        
        # Student network
        self.student = model_map[model_type]
        embed_dim = self.student.embed_dim
        
        self.student = nn.Sequential(
            self.student,
            DINOHead(embed_dim, out_dim, use_bn)
        )
        # Teacher network
        self.teacher = model_map[model_type]
        self.teacher = nn.Sequential(
            self.teacher,
            DINOHead(embed_dim, out_dim, use_bn)
        )
        # Initialize teacher and student with same weights
        self.teacher.load_state_dict(self.student.state_dict())
        # Turn off gradients for teacher network
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student.add_module("energy", nn.Linear(out_dim, 1))
        self.teacher.add_module("energy", nn.Linear(out_dim, 1))

    def forward(self, x, is_teacher=False):
        # Forward pass through Perceiver
        batch_size = x.shape[0]
        x = self.student[0](x) if not is_teacher else self.teacher[0](x)
        # combine latents
        x = x.view(batch_size, 1, -1)
        x = self.student[1](x) if not is_teacher else self.teacher[1](x)
        # Forward pass through Energy layer
        energy_output = self.student.energy(x) if not is_teacher else self.teacher.energy(x)
        # Debugging: Print shape of energy_output and the energy layer
        # print("Shape of energy_output: ", energy_output.shape)
        # print(f"{self.student.energy if not is_teacher else self.teacher.energy}")
        return x, energy_output

    def get_last_selfattention(self, x):
        return self.student[0].get_last_selfattention(x)

    
    
class EBMDINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, student_energy, teacher_energy, epoch):
        dino_loss = self.compute_dino_loss(student_output, teacher_output, epoch)
        # Calculate the energy loss
        energy_loss = 0
        n_loss_terms = 0
        for iteacher in range(teacher_energy.shape[0]):
            for istudent in range(student_energy.shape[0]):
                if iteacher == istudent:
                    continue
                loss = abs(teacher_energy[iteacher] - student_energy[istudent])
                energy_loss += loss.mean()
                n_loss_terms += 1
        energy_loss /= n_loss_terms  # Average the energy loss terms
        
        # Combine the DINO loss and the energy loss
        total_loss = dino_loss + 0.0 * energy_loss
        self.update_center(teacher_output)
        # print("DINO loss:", dino_loss.item()); print("Energy loss:", energy_loss.item())
        return total_loss

    def compute_dino_loss(self, student_output, teacher_output, epoch):
        self.center = self.center.to(student_output.device)
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
 
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(int(224/(244/32)), scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(int(224/(244/32)), scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(int(96/(244/32)), scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    
    
class VOCSegmentationImages(datasets.VOCSegmentation):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image

    
# --- args START --- #
# Argument parsing
parser = argparse.ArgumentParser(description='DINO training script')
parser.add_argument('--model_type', default='perceiver', type=str, help='Model type for DINO (tiny, small, base)')
parser.add_argument('--total_epochs', default=300, type=int, help='Total training epochs')
parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help='Warmup teacher temperature for DINO loss')
parser.add_argument('--teacher_temp', default=0.04, type=float, help='Teacher temperature for DINO loss')
parser.add_argument('--warmup_teacher_temp_epochs', default=40, type=int, help='Number of epochs for teacher temperature warmup')
parser.add_argument('--student_temp', default=0.1, type=float, help='Student temperature for DINO loss')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
parser.add_argument('--name', default='v1', type=str, help='Name for the training run')
parser.add_argument('--lr', default=0.005, type=float, help='')
parser.add_argument('--dino_dim', default=2048, type=int, help='')
args = parser.parse_args()



model_type = args.model_type
total_epochs = args.total_epochs
warmup_teacher_temp = args.warmup_teacher_temp
teacher_temp = args.teacher_temp
warmup_teacher_temp_epochs = args.warmup_teacher_temp_epochs
student_temp = args.student_temp
batch_size = args.batch_size
name = args.name
lr = args.lr
dino_out_dim =  args.dino_dim



out_name = F"model_{args.model_type}_{args.name}"
path2models = f"{dir_root}/models_{out_name}"
if not os.path.exists(path2models):
    os.makedirs(path2models)
    
local_crops_number = 12


data_transform = DataAugmentationDINO(
    global_crops_scale=(0.4, 1.0), 
    local_crops_scale=(0.05, 0.4), 
    local_crops_number=local_crops_number
)


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform) # Dataset setup
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # Data loader setup
# Device specification
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Using GPU if available, else CPU


# Only one instance of DINO class
dino_instance = DINO(model_type=model_type, out_dim=dino_out_dim, use_bn=False).to(device)
# Optimizer initialization
optimizer = AdamW(dino_instance.student.parameters(), lr=lr, weight_decay=0.01)


# Instantiate the EBM DINO loss
ebm_dino_loss = EBMDINOLoss(out_dim=dino_out_dim, ncrops=(2+local_crops_number), warmup_teacher_temp=warmup_teacher_temp, 
                            teacher_temp=teacher_temp, warmup_teacher_temp_epochs=warmup_teacher_temp_epochs, nepochs=total_epochs)


# New: Create a validation dataset and loader
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


def evaluate(model, val_loader, device, epoch):  # Added epoch as an argument
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for images, _ in val_loader:
            student_outputs = []
            teacher_outputs = []
            student_energies = []
            teacher_energies = []
            
            for j, image in enumerate(images):
                image = image.to(device)
                
                # Forward pass through student networks
                student_output, student_energy = model(image)
                student_outputs.append(student_output)
                student_energies.append(student_energy)
                
                if j < 2:
                    # Forward pass through teacher networks
                    teacher_output, teacher_energy = model.forward(image, is_teacher=True)
                    teacher_outputs.append(teacher_output)
                    teacher_energies.append(teacher_energy)
            
            # Convert lists of tensors to tensors
            student_outputs_tensor = torch.stack(student_outputs)
            teacher_outputs_tensor = torch.stack(teacher_outputs)
            student_energies_tensor = torch.stack(student_energies)
            teacher_energies_tensor = torch.stack(teacher_energies)
            
            loss = ebm_dino_loss(
                student_output=student_outputs_tensor,
                teacher_output=teacher_outputs_tensor,
                student_energy=student_energies_tensor,
                teacher_energy=teacher_energies_tensor,
                epoch=epoch
            )
            
            total_loss += loss.item() * images[0].size(0)  # Assuming the first set of images in 'images' has the batch size
            total_samples += images[0].size(0)
    
    model.train()
    return total_loss / total_samples


# Training loop
for epoch in range(total_epochs):
    for i, (images, _) in enumerate(train_loader):
        # Separate the two sets of crops
        student_outputs = []
        teacher_outputs = []
        student_energies = []
        teacher_energies = []
        
        for j, image in enumerate(images):
            image = image.to(device)
            
            # Forward pass through student networks
            student_output, student_energy = dino_instance(image)
            student_outputs.append(student_output)
            student_energies.append(student_energy)
            
            
            if j < 2:
                with torch.no_grad():
                    # Forward pass through teacher networks
                    teacher_output, teacher_energy = dino_instance.forward(image, is_teacher=True)
                    teacher_outputs.append(teacher_output)
                    teacher_energies.append(teacher_energy)
        
        # Convert lists of tensors to tensors
        student_outputs_tensor = torch.stack(student_outputs)
        teacher_outputs_tensor = torch.stack(teacher_outputs)
        student_energies_tensor = torch.stack(student_energies)
        teacher_energies_tensor = torch.stack(teacher_energies)

        # Compute EBM DINO loss
        loss = ebm_dino_loss(
            student_output=student_outputs_tensor, 
            teacher_output=teacher_outputs_tensor, 
            student_energy=student_energies_tensor, 
            teacher_energy=teacher_energies_tensor, 
            epoch=epoch
        )
        

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        clip_grad = 3
        freeze_last_layer = 1
        param_norms = utils.clip_gradients(dino_instance.student, clip_grad)
        utils.cancel_gradients_last_layer(epoch, dino_instance.student, freeze_last_layer)
        optimizer.step()

        # Update teacher parameters (exponential moving average of student parameters)
        with torch.no_grad():
            momentum = 0.996  # Or whatever value you've chosen
            for param_student, param_teacher in zip(dino_instance.student.parameters(), dino_instance.teacher.parameters()):
                param_teacher.data.mul_(momentum).add_((1 - momentum) * param_student.detach().data)
                
                
    if epoch % 1 == 0:  # Evaluate every 10 epochs
        val_loss = evaluate(dino_instance, val_loader, device, epoch)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")
        sys.stdout.flush()

    if i%50 == 0:
        print([epoch, i, loss])
        

    if epoch % 5 == 0:
        torch.save(dino_instance.student.state_dict(), f'{path2models}/student_model_e-{str(epoch).zfill(4)}.pth')
        torch.save(dino_instance.teacher.state_dict(), f'{path2models}/teacher_model_e-{str(epoch).zfill(4)}.pth')


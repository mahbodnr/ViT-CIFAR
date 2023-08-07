import torch
from matplotlib import pyplot as plt

from network import Net
from argparse import Namespace
from utils import get_dataloader
from layers import TransformerEncoder
from attention import get_joint_attentions

model_name = "vit"
n_layers = 3
# PATH= "logs/vit_c10/version_0/checkpoints/epoch=27-step=10948.ckpt"
PATH = f"logs/{model_name}_c10_{n_layers}l_aa.ckpt"
# Load model
print("Loading model")
trained_model = torch.load(PATH)
hparams = trained_model["hyper_parameters"]
args = Namespace(**hparams)
args._comet_api_key = None
args.semi_supervised = False #DELETE later
args.download_data = False #DELETE later
args.shuffle = False #DELETE later
args.pin_memory = False #DELETE later
args.unsupervised_steps = 0 #DELETE later
model = Net(args)
model.load_state_dict(trained_model["state_dict"])
model.eval()
# get a batch of data
print("Loading data")
_, test_dl = get_dataloader(args)
imgs, _ = next(iter(test_dl))
print("Data shape:", imgs.shape)
# Enable saving attention maps
for module in model.modules():
    if isinstance(module, TransformerEncoder):
        module.save_attn_map = True
# one forward pass
print("Forward pass")
with torch.no_grad():
    out = model(imgs)

attn_maps = []
for module in model.modules():
    if isinstance(module, TransformerEncoder):
        attn_maps.append(module.get_attention_map())

attention_maps = torch.stack(attn_maps).cpu()
# print("attention layers:", len(attn_maps))
# joint_attentions= attention_rollout(attn_maps, Token= None)
# print("joint attention shape:", joint_attention.shape)

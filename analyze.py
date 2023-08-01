import torch
from network import Net
from argparse import Namespace
from utils import get_dataset
from layers import TransformerEncoder
from attention import attention_rollout

model_name = "ae"
# PATH= "logs/vit_c10/version_0/checkpoints/epoch=27-step=10948.ckpt"
PATH = f"logs/{model_name}_c10_3l_aa.ckpt"
# Load model
print("Loading model")
trained_model = torch.load(PATH)
hparams = trained_model["hyper_parameters"]
args = Namespace(**hparams)
args._comet_api_key = None
model = Net(args)
model.load_state_dict(trained_model["state_dict"])
model.eval()
# get a batch of data
print("Loading data")
_, test_ds = get_dataset(args)
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=args.eval_batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
)
imgs, _ = next(iter(test_dl))
print("Data shape:", imgs.shape)
# Enable saving attention maps
for module in model.modules():
    if isinstance(module, TransformerEncoder):
        module.save_attn_map = True
# one forward pass
with torch.no_grad():
    out = model(imgs)

attn_maps = []
for module in model.modules():
    if isinstance(module, TransformerEncoder):
        attn_maps.append(module.get_attention_map())

print("attention layers:", len(attn_maps))
joint_attentions= attention_rollout(attn_maps, head_dim=1 if model_name == "vit" else None)
        
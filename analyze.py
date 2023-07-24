import torch
from network import Net
from argparse import Namespace

PATH= "logs/vit_c10/version_0/checkpoints/epoch=27-step=10948.ckpt"
# Load model
trained_model = torch.load(PATH)
hparams = trained_model["hyper_parameters"]
args = Namespace(**hparams)
args._comet_api_key = None
model = Net(args)
model.load_state_dict(trained_model["state_dict"])
import torch
from network import Net
from argparse import Namespace
from utils import get_dataloader

def load_run_model(model_name=None, n_layers=None, model_path=None, batch_size= None):
    if model_path is None:
        assert model_name is not None and n_layers is not None
        model_path = f"logs/{model_name}_c10_{n_layers}l_aa.ckpt"

    print("Loading model")
    trained_model = torch.load(model_path)
    print('*' * 100, type(trained_model))
    hparams = trained_model["hyper_parameters"]
    args = Namespace(**hparams)
    args._comet_api_key = None
    args.chunk= args.chunk if hasattr(args, "chunk") else False
    args.legacy_heads = args.heads if hasattr(args, "heads") else False
    if batch_size is not None:
        args.eval_batch_size = batch_size
    model = Net(args)

    # DELETE LATER:
    # Removing norm2 from the model
    for name, module in model.named_modules():
        if "norm2" in name:
            # remove norm2 from the model
            delattr(model, name)
    # removing norm2 from the state dict
    for key in list(trained_model["state_dict"].keys()):
        if "norm2" in key:
            del trained_model["state_dict"][key]
    # END DELETE LATER

    model.load_state_dict(trained_model["state_dict"])
    model.eval()
    # get a batch of data
    print("Loading data")
    _, test_dl = get_dataloader(args)
    imgs, _ = next(iter(test_dl))
    print("Data shape:", imgs.shape)
    # Enable saving attention maps
    for module in model.modules():
        if hasattr(module, "save_attn_map"):
            module.save_attn_map = True
    # one forward pass
    print("Forward pass")
    with torch.no_grad():
        out = model(imgs)

    print("Loading complete")
    return model, imgs, out

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
    hparams = trained_model["hyper_parameters"]
    args = Namespace(**hparams)
    args._comet_api_key = None
    # Delete later
    args.chunk= args.chunk if hasattr(args, "chunk") else False
    args.legacy_heads = args.heads if hasattr(args, "heads") else False
    args.use_nnmf_layers = args.use_nnmf_layers if hasattr(args, "use_nnmf_layers") else False
    args._nnmf_params = {}
    args.nnmf_local_learning = args.nnmf_local_learning if hasattr(args, "nnmf_local_learning") else False
    args.nnmf_scale_grade = args.nnmf_scale_grade if hasattr(args, "nnmf_scale_grade") else False
    args.mask_type = args.mask_type if hasattr(args, "mask_type") else "zeros"
    # END DELETE LATER
    if batch_size is not None:
        args.eval_batch_size = batch_size
    model = Net(args)

    # DELETE LATER:
    # Removing norm2 from the model
    for name, module in model.named_modules():
        if "norm2" in name:
            # remove norm2 from the model
            delattr(model, name)
    # END DELETE LATER

    model.load_state_dict(trained_model["state_dict"], strict=False)
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
        if torch.cuda.is_available():    
            model = model.cuda()
            imgs = imgs.cuda()
            out = model(imgs)
            model = model.cpu()
            imgs = imgs.cpu()
            out = out.cpu()
        else:
            out = model(imgs)

    print("Loading complete")
    return model, imgs, out

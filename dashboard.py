import streamlit as st
from attention.utils import (
    draw_divided_image_with_index,
    get_joint_attentions,
    get_attention_maps,
)
from run_model import load_run_model
import numpy as np
import torch
import cv2
import math
import os

st.set_page_config(layout="wide")

batch_size = 10


def numpy_to_cv2(image):
    if type(image) == torch.Tensor:
        image = image.numpy()
    image = image - image.min()
    image = image / image.max()
    image = (image * 255).astype(np.uint8)
    # make sure channels are last
    if len(image.shape) == 3:
        if image.shape[-1] not in [1, 3]:
            if image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            else:
                raise ValueError(
                    "Image shape not supported. shape: {}".format(image.shape)
                )

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def to_heatmap(image, size, cmap=cv2.COLORMAP_JET, interpolation=cv2.INTER_NEAREST):
    image = numpy_to_cv2(image)
    # resize:
    image = cv2.resize(image, size, interpolation=interpolation)

    return cv2.applyColorMap(image, cmap)


def mask_image(image, mask, interpolation=cv2.INTER_NEAREST, alpha=0.5):
    if type(image) in [torch.Tensor, np.ndarray]:
        image = numpy_to_cv2(image)
    if type(mask) in [torch.Tensor, np.ndarray]:
        mask = numpy_to_cv2(mask)

    # resize mask to image size
    mask = cv2.resize(mask, image.shape[:2][::-1], interpolation=interpolation)

    # mask to BGR
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    return cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)


@st.cache_data()
def load_data(model_name=None, n_layers=None, model_path=None, batch_size=None):
    model, imgs, out = load_run_model(
        model_name=model_name,
        n_layers=n_layers,
        model_path=model_path,
        batch_size=batch_size,
    )
    attention_maps = get_attention_maps(model)
    return model, imgs, attention_maps


def apply_change_to_image(image, seq_len, selected):
    return draw_divided_image_with_index(image, seq_len, selected)


def main():
    # Sidebar
    st.sidebar.title("Visualizing Attention in Transformers")
    st.sidebar.markdown("select the model:")
    # model_name = st.sidebar.selectbox(
    #     "Model:",
    #     ("vit", "ae"),
    # )
    # n_layers = st.sidebar.selectbox(
    #     "Number of Layers:",
    #     (1, 3, 7),
    # )
    # read all file names in the "models" folder
    models = os.listdir("models")
    if not models:
        st.error("No models found in the 'models' folder.")
        return

    model_name, model_name_refresh = st.sidebar.columns([5, 1])
    selected_model = model_name.selectbox(
        "Model",
        models,
        label_visibility= "collapsed"
    )
    model_path = f"models/{selected_model}"

    model_name_refresh.button("🔄")
    if model_name_refresh:
        models = os.listdir("models")
        if not models:
            st.error("No models found in the 'models' folder.")
            return

    model, imgs, attention_maps = load_data(
        batch_size=batch_size,
        model_path=model_path,
        # model_name=model_name, n_layers=n_layers
    )
    if attention_maps.dim() == 4:
        # add dimension for heads
        attention_maps = attention_maps.unsqueeze(2)
    n_layers, _, n_heads, seq_len, _ = attention_maps.shape

    st.sidebar.markdown("Image settings:")
    img_index = st.sidebar.number_input("Select an image:", 1, batch_size, 1) - 1
    input_image_np = imgs[img_index]

    token = st.sidebar.radio("Token:", ("All Tokens", "<CLS> Token", "choose a token"))
    if token == "All Tokens":
        token = None
    elif token == "<CLS> Token":
        token = 0
    elif token == "choose a token":
        token = st.sidebar.number_input("Select a part:", 1, seq_len - 1, 1)

    processed_image = apply_change_to_image(
        input_image_np, int(np.sqrt(seq_len - 1)), token
    )
    input_image_size = processed_image.shape[:2]
    input_image = st.sidebar.image(processed_image, use_column_width=True)

    show_image_maps = st.sidebar.checkbox(
        "Show maps on input image",
        value=False,
        help="Not available when all tokens are selected",
    )

    st.sidebar.markdown("Attention Maps:")
    transpose_attention = st.sidebar.checkbox(
        "Transpose Attention",
        value=False,
        help="Transposes all attention maps, so instead of attention value of the selected token to all other tokens, it shows the attention value of all other tokens to the selected token.",
    )
    if transpose_attention:
        attention_maps = attention_maps.transpose(-1, -2)

    if n_heads > 1:
        heads_transform = st.sidebar.radio(
            "Heads:", ("Show all heads", "Average over heads", "choose a head")
        )
        if heads_transform == "Average over heads":
            attention_maps = attention_maps.mean(dim=2, keepdim=True)
            n_heads = 1
        elif heads_transform == "choose a head":
            head = st.sidebar.selectbox("Select a head:", range(1, n_heads + 1))
            attention_maps = attention_maps[:, :, head - 1 : head]
            n_heads = 1

    with st.sidebar.expander("Advanced Options"):
        cmap = st.selectbox(
            "Color Map:",
            (
                # Default:
                "Jet",
                # Other options:
                "Autumn",
                "Bone",
                "Cool",
                "Hot",
                "HSV",
                "Ocean",
                "Pink",
                "Rainbow",
                "Spring",
                "Summer",
                "Winter",
            ),
        )
        heatmap_cmap = getattr(cv2, f"COLORMAP_{cmap.upper()}")
        interpolation = st.selectbox(
            "Resize Interpolation:",
            (
                # Default:
                "Linear",
                # Other options:
                "Nearest",
                "Area",
                "Cubic",
                "Lanczos4",
            ),
        )
        resize_interpolation = getattr(cv2, f"INTER_{interpolation.upper()}")

        max_columns = st.number_input(
            "Max Number of Maps in one Column:",
            1,
            10,
            5,
        )

        mask_alpha = st.slider(
            "Mask Intensity:",
            0.0,
            1.0,
            0.4,
            0.05,
        )

    # Main page
    if token is None:
        show_image_maps = False
    if not show_image_maps:
        (
            col_joint_attentions,
            col_attention_maps,
        ) = st.columns(2, gap="medium")
        col_joint_attentions.header("Joint Attentions")
        col_attention_maps.header("Attention Maps")
    else:
        (
            col_joint_attentions,
            col_joint_attentions_input,
            col_attention_maps,
            col_attention_maps_input,
        ) = st.columns(4, gap="small")
        col_joint_attentions.header("Joint Attentions")
        col_joint_attentions_input.header(" * Input")
        col_attention_maps.header("Attention Maps")
        col_attention_maps_input.header(" * Input")

    # Display joint attention maps
    joint_attentions = get_joint_attentions(attention_maps, token=token).numpy()
    if token is not None:
        joint_attentions = joint_attentions[..., 1:].reshape(
            n_layers,
            batch_size,
            n_heads,
            int(np.sqrt(seq_len - 1)),
            int(np.sqrt(seq_len - 1)),
        )
    for layer in reversed(range(n_layers)):
        col_joint_attentions.subheader(f"Layer {layer+1}")
        if show_image_maps:
            col_joint_attentions_input.subheader(f"Layer {layer+1}")
        if n_heads < max_columns:
            head_joint_attn = col_joint_attentions.columns(n_heads)
            if show_image_maps:
                head_joint_attn_input = col_joint_attentions_input.columns(n_heads)
            for i, head_image in enumerate(head_joint_attn):
                attention_heatmap = to_heatmap(
                    joint_attentions[layer, img_index, i],
                    input_image_size,
                    heatmap_cmap,
                    resize_interpolation,
                )
                head_image.image(
                    attention_heatmap,
                    caption=f"{i+1}",
                    use_column_width=True,
                    channels="BGR",
                )
                if show_image_maps:
                    head_joint_attn_input[i].image(
                        mask_image(
                            input_image_np,
                            attention_heatmap,
                            interpolation=resize_interpolation,
                            alpha=mask_alpha,
                        ),
                        caption=f"{i+1}",
                        use_column_width=True,
                        channels="BGR",
                    )
        else:
            head_idx = 0
            for row in range(math.ceil(n_heads / max_columns)):
                head_joint_attn = col_joint_attentions.columns(max_columns)
                if show_image_maps:
                    head_joint_attn_input = col_joint_attentions_input.columns(
                        max_columns
                    )
                for head_image_idx in range(max_columns):
                    heatmap = to_heatmap(
                        joint_attentions[layer, img_index, head_idx],
                        input_image_size,
                        heatmap_cmap,
                        resize_interpolation,
                    )
                    head_joint_attn[head_image_idx].image(
                        heatmap,
                        caption=f"{head_idx+1}",
                        use_column_width=True,
                        channels="BGR",
                    )
                    if show_image_maps:
                        head_joint_attn_input[head_image_idx].image(
                            mask_image(
                                input_image_np,
                                heatmap,
                                interpolation=resize_interpolation,
                                alpha=mask_alpha,
                            ),
                            caption=f"{head_idx+1}",
                            use_column_width=True,
                            channels="BGR",
                        )

                    head_idx += 1
                    if head_idx == n_heads:
                        break

    # Display attention maps
    if token is not None:
        attention_maps = attention_maps[..., token, 1:].reshape(
            n_layers,
            batch_size,
            n_heads,
            int(np.sqrt(seq_len - 1)),
            int(np.sqrt(seq_len - 1)),
        )
    for layer in reversed(range(n_layers)):
        col_attention_maps.subheader(f"Layer {layer+1}")
        if show_image_maps:
            col_attention_maps_input.subheader(f"Layer {layer+1}")
        if n_heads < max_columns:
            head_attn = col_attention_maps.columns(n_heads)
            if show_image_maps:
                head_attn_input = col_attention_maps_input.columns(n_heads)
            for i, head_image in enumerate(head_attn):
                heatmap = to_heatmap(
                    attention_maps[layer, img_index, i],
                    input_image_size,
                    heatmap_cmap,
                    resize_interpolation,
                )
                head_image.image(
                    heatmap,
                    caption=f"{i+1}",
                    use_column_width=True,
                    channels="BGR",
                )
                if show_image_maps:
                    head_attn_input[i].image(
                        mask_image(
                            input_image_np,
                            heatmap,
                            interpolation=resize_interpolation,
                            alpha=mask_alpha,
                        ),
                        caption=f"{i+1}",
                        use_column_width=True,
                        channels="BGR",
                    )
        else:
            head_idx = 0
            for row in range(math.ceil(n_heads / max_columns)):
                head_attn = col_attention_maps.columns(max_columns)
                if show_image_maps:
                    head_attn_input = col_attention_maps_input.columns(max_columns)
                for head_image_idx in range(max_columns):
                    heatmap = to_heatmap(
                        attention_maps[layer, img_index, head_idx],
                        input_image_size,
                        heatmap_cmap,
                        resize_interpolation,
                    )
                    head_attn[head_image_idx].image(
                        heatmap,
                        caption=f"{head_idx+1}",
                        use_column_width=True,
                        channels="BGR",
                    )
                    if show_image_maps:
                        head_attn_input[head_image_idx].image(
                            mask_image(
                                input_image_np,
                                heatmap,
                                interpolation=resize_interpolation,
                                alpha=mask_alpha,
                            ),
                            caption=f"{head_idx+1}",
                            use_column_width=True,
                            channels="BGR",
                        )
                    head_idx += 1
                    if head_idx == n_heads:
                        break


if __name__ == "__main__":
    main()

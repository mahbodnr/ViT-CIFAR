import torch
from PIL import Image
import cv2
import numpy as np

def draw_divided_image_with_index(image, N, selected_index):
    # Get the image dimensions
    chans, height, width = image.size()

    image = image.permute(1, 2, 0).numpy()
    image = image - image.min()
    image = image / image.max()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # scale the image if the size is too small
    if height < 256 or width < 256:
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        height, width = image.shape[:2]

    # Calculate the dimensions of each subpart
    part_height = height // N
    part_width = width // N

    # Draw lines to divide the image and add index numbers
    for i in range(N):
        for j in range(N):
            x_start, x_end = j * part_width, (j + 1) * part_width
            y_start, y_end = i * part_height, (i + 1) * part_height
            part_index = i * N + j + 1  # Adjust indexing to start from 1
            part_center_x = (x_start + x_end) // 2
            part_center_y = (y_start + y_end) // 2

            
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 0), 1)  # Draw rectangles
            text_size = np.sqrt(height * width)/1000
            cv2.putText(image, str(part_index), (part_center_x, part_center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 1, cv2.LINE_AA)  # Put index numbers

    if selected_index is None or selected_index == 0:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Calculate the row and column indices for the selected part
    row_index = (selected_index - 1) // N
    col_index = (selected_index - 1) % N

    # Create a yellow layer for the selected part with matching dimensions
    yellow_layer = np.zeros((part_height, part_width, 3), dtype=np.uint8)
    yellow_layer[:, :, 0] = 0
    yellow_layer[:, :, 1] = 255
    yellow_layer[:, :, 2] = 255

    # Add the yellow layer to the selected part
    x_start = col_index * part_width
    x_end = (col_index + 1) * part_width
    y_start = row_index * part_height
    y_end = (row_index + 1) * part_height
    image[y_start:y_end, x_start:x_end] = cv2.addWeighted(image[y_start:y_end, x_start:x_end], 0.5, yellow_layer, 0.5, 0)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_attention_maps(model):
    attn_maps = []
    for module in model.modules():
        if hasattr(module, "get_attention_map"):
            attn_maps.append(module.get_attention_map())

    return torch.stack(attn_maps)

def get_joint_attentions(attn_mat, token = None):
    """
    Rollout the attention weights across multiple heads.


    Args:
        attn_mat: torch.Tensor of shape (num_layers, batch_size, num_heads, seq_len, seq_len)
        token: int, optional. If provided, only the attention weights for this token are rolled out.

    Returns:
        joint_attentions: torch.Tensor 
            If token is None, shape (num_layers, batch_size, num_heads, seq_len, seq_len)
            If token is provided, shape (num_layers, batch_size, num_heads, seq_len)
    """
    # check if the input is a torch tensor
    if not isinstance(attn_mat, torch.Tensor):
        attn_mat = torch.stack(attn_mat).cpu()

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(attn_mat.size(-1))
    aug_att_mat = attn_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    if token is None:
        return joint_attentions
    else:
        return joint_attentions[:, :, :, token, :]


if __name__ == '__main__':
    # Example usage:
    N = 8  # Number of rows and columns to divide the image into
    selected_index = 2  # Index of the selected part (0-indexed)
    image = torch.rand(3, 32, 32)

    result_image = draw_divided_image_with_index(image, N, selected_index)
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

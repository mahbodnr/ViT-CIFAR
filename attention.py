import torch


def attention_rollout(attn_mat, head_dim= None):
    # check if the input is a torch tensor
    if not isinstance(attn_mat, torch.Tensor):
        attn_mat = torch.stack(attn_mat).cpu()

    if head_dim is not None:
        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=head_dim)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(-1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    return joint_attentions[-1]
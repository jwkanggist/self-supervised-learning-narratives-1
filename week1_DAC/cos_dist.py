import torch


def get_cos_dist_matrix(features):
    norm_feat = torch.exp(features - torch.max(features))
    l2norm = torch.norm(norm_feat, dim=1, keepdim=True)
    unit_vec = features / l2norm
    return torch.mm(unit_vec.transpose(0, 1), unit_vec)

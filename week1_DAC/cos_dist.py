import torch


def get_cos_similarity(features, threshold):
    l2norm = torch.norm(features, dim=1, keepdim=True)
    unit_vec = features / l2norm
    cos_similarity = torch.mm(unit_vec, unit_vec.transpose(0, 1))
    return torch.where(cos_similarity > threshold, 1, 0)

from typing import List, Union

import torch

from sdm.config import *


def cos_sim(
    a: Union[List[torch.Tensor], torch.Tensor],
    b: Union[List[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if isinstance(a, list):
        a = torch.stack(a)

    if isinstance(b, list):
        b = torch.stack(b)

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a).float()

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b).float()

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a = a.cuda()
    b = b.cuda()
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)

    return torch.mm(a_norm, b_norm.transpose(0, 1)).cpu()

import torch as tc

from drl.utils.typing import NestedTensor, Indices


def slice_nested_tensor(nt: NestedTensor, indices: Indices) -> NestedTensor:
    if isinstance(nt, tc.Tensor):
        return nt[indices]
    nt_new = {}
    for k in nt:
        nt_new[k] = slice_nested_tensor(nt[k], indices)
    return nt_new


def clone_nested_tensor(nt: NestedTensor) -> NestedTensor:
    if isinstance(nt, tc.Tensor):
        return nt.clone()
    nt_new = {}
    for k in nt:
        nt_new[k] = clone_nested_tensor(nt[k])
    return nt_new

from typing import NamedTuple

import torch

from corener.data.entities import TokenSpan


def get_token(token_embedding: torch.tensor, input_ids: torch.tensor, token: int):
    """Get specific token embedding (e.g. [CLS])"""
    emb_size = token_embedding.shape[-1]

    token_h = token_embedding.view(-1, emb_size)
    flat = input_ids.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[: tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[: tensor_shape[0], : tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[
            : tensor_shape[0], : tensor_shape[1], : tensor_shape[2]
        ] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[
            : tensor_shape[0], : tensor_shape[1], : tensor_shape[2], : tensor_shape[3]
        ] = tensor

    return extended_tensor


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def swap(v1, v2):
    return v2, v1


def get_span_tokens(tokens, span):
    inside = False
    span_tokens = []

    for t in tokens:
        if t.span[0] == span[0]:
            inside = True

        if inside:
            span_tokens.append(t)

        if inside and t.span[1] == span[1]:
            return TokenSpan(span_tokens)

    return None


class TrainBatch(NamedTuple):
    encodings: torch.tensor
    context_masks: torch.tensor
    # entity
    entity_masks: torch.tensor
    entity_sizes: torch.tensor
    entity_types: torch.tensor
    entity_sample_masks: torch.tensor
    # mentions
    mention_masks: torch.tensor
    mention_sizes: torch.tensor
    mention_types: torch.tensor
    mention_sample_masks: torch.tensor
    # relations
    rels: torch.tensor
    rel_masks: torch.tensor
    rel_types: torch.tensor
    rel_sample_masks: torch.tensor
    # references
    refs: torch.tensor
    ref_masks: torch.tensor
    ref_types: torch.tensor
    ref_sample_masks: torch.tensor
    # binary mask
    is_ner: torch.tensor
    is_emd: torch.tensor
    is_re: torch.tensor
    is_cr: torch.tensor

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(*[t.to(device) for t in self])

    def __len__(self):
        return len(self.encodings)


class EvalBatch(NamedTuple):
    encodings: torch.tensor
    context_masks: torch.tensor
    entity_masks: torch.tensor
    entity_sizes: torch.tensor
    entity_spans: torch.tensor
    entity_sample_masks: torch.tensor
    is_ner: torch.tensor
    is_emd: torch.tensor
    is_re: torch.tensor
    is_cr: torch.tensor

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(*[t.to(device) for t in self])

    def __len__(self):
        return len(self.encodings)

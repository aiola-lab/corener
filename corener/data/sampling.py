import random
from functools import partial
from typing import List, Union

import torch

from corener.data.entities import Document
from corener.utils.data import EvalBatch, TrainBatch, padded_stack


def create_entity_sample(
    doc: Document,
    candidates,
    context_size,
    max_span_size,
    token_count,
    neg_entity_count,
):
    # positive entities\mentions
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = (
        [],
        [],
        [],
        [],
    )
    for e in candidates:
        pos_entity_spans.append(e.span)
        pos_entity_types.append(e.entity_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))
        pos_entity_sizes.append(len(e.tokens))

    # negative entities
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i : i + size].span
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

    # sample negative entities
    neg_entity_samples = random.sample(
        list(zip(neg_entity_spans, neg_entity_sizes)),
        min(len(neg_entity_spans), neg_entity_count),
    )
    neg_entity_spans, neg_entity_sizes = (
        zip(*neg_entity_samples) if neg_entity_samples else ([], [])
    )

    neg_entity_masks = [
        create_entity_mask(*span, context_size) for span in neg_entity_spans
    ]
    neg_entity_types = [0] * len(neg_entity_spans)

    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)

    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return (
        entity_masks,
        entity_sizes,
        entity_types,
        entity_sample_masks,
        pos_entity_spans,
    )


def create_relation_sample(
    candidates, pos_entity_spans, rel_type_count, context_size, neg_rel_count
):
    # positive relations
    # collect relations between entity pairs
    entity_pair_relations = dict()
    for rel in candidates:
        pair = (rel.head_entity, rel.tail_entity)
        if pair not in entity_pair_relations:
            entity_pair_relations[pair] = []
        entity_pair_relations[pair].append(rel)

    # build positive relation samples
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for pair, rels in entity_pair_relations.items():
        head_entity, tail_entity = pair
        s1, s2 = head_entity.span, tail_entity.span
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))
        pos_rel_spans.append((s1, s2))

        pair_rel_types = [r.relation_type.index for r in rels]
        pair_rel_types = [int(t in pair_rel_types) for t in range(1, rel_type_count)]
        pos_rel_types.append(pair_rel_types)
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))

        # negative relations
        # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related

    neg_rel_spans = []
    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans:
                neg_rel_spans.append((s1, s2))

    # sample negative relations
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [
        (pos_entity_spans.index(s1), pos_entity_spans.index(s2))
        for s1, s2 in neg_rel_spans
    ]
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_types = [(0,) * (rel_type_count - 1)] * len(neg_rel_spans)

    rels = pos_rels + neg_rels
    rel_types = pos_rel_types + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(rels) == len(rel_masks) == len(rel_types)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.float32)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1, rel_type_count - 1], dtype=torch.float32)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    return rels, rel_masks, rel_types, rel_sample_masks


def create_train_sample(
    doc: Document,
    neg_entity_count: int,
    neg_rel_count: int,
    max_span_size: int,
    rel_type_count: int,
    ref_type_count: int = 2,  # binary for ref/no ref
):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation

    (
        entity_masks,
        entity_sizes,
        entity_types,
        entity_sample_masks,
        pos_entity_spans,
    ) = create_entity_sample(
        doc, doc.entities, context_size, max_span_size, token_count, neg_entity_count
    )
    (
        mention_masks,
        mention_sizes,
        mention_types,
        mention_sample_masks,
        pos_mention_spans,
    ) = create_entity_sample(
        doc, doc.mentions, context_size, max_span_size, token_count, neg_entity_count
    )
    rels, rel_masks, rel_types, rel_sample_masks = create_relation_sample(
        doc.relations, pos_entity_spans, rel_type_count, context_size, neg_rel_count
    )
    refs, ref_masks, ref_types, ref_sample_masks = create_relation_sample(
        doc.references, pos_mention_spans, ref_type_count, context_size, neg_rel_count
    )

    return TrainBatch(
        encodings=encodings,
        context_masks=context_masks,
        entity_masks=entity_masks,
        entity_sizes=entity_sizes,
        entity_types=entity_types,
        entity_sample_masks=entity_sample_masks,
        mention_masks=mention_masks,
        mention_sizes=mention_sizes,
        mention_types=mention_types,
        mention_sample_masks=mention_sample_masks,
        rels=rels,
        rel_masks=rel_masks,
        rel_types=rel_types,
        rel_sample_masks=rel_sample_masks,
        refs=refs,
        ref_masks=ref_masks,
        ref_types=ref_types,
        ref_sample_masks=ref_sample_masks,
        is_ner=torch.tensor([doc.is_ner]),
        is_emd=torch.tensor([doc.is_emd]),
        is_re=torch.tensor([doc.is_re]),
        is_cr=torch.tensor([doc.is_cr]),
    )


def create_eval_sample(doc: Document, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i : i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[: len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[: len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor(
            [1] * entity_masks.shape[0], dtype=torch.bool
        )
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return EvalBatch(
        **dict(
            encodings=encodings,
            context_masks=context_masks,
            entity_masks=entity_masks,
            entity_sizes=entity_sizes,
            entity_spans=entity_spans,
            entity_sample_masks=entity_sample_masks,
            is_ner=torch.tensor([doc.is_ner]),
            is_emd=torch.tensor([doc.is_emd]),
            is_re=torch.tensor([doc.is_re]),
            is_cr=torch.tensor([doc.is_cr]),
        )
    )


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch: List[Union[TrainBatch, EvalBatch]], encoding_padding=0):
    batch_cls = batch[0].__class__
    padded_batch = dict()
    batch = [b.as_dict() for b in batch]
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = padded_stack(
                [s[key] for s in batch],
                padding=encoding_padding if key == "encodings" else 0,
            )

    return batch_cls(**padded_batch)


def partial_collate_fn_padding(encoding_padding):
    """return partially initialized collate_fn_padding to support different padding indices"""
    return partial(collate_fn_padding, encoding_padding=encoding_padding)

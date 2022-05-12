from typing import Callable, Tuple, Union

import torch

from corener.data.dataset import DataParser
from corener.models import ModelOutput
from corener.utils.clusters import convert_to_clusters
from corener.utils.data import EvalBatch, TrainBatch, get_span_tokens


def convert_model_output(
    output: ModelOutput,
    batch: Union[EvalBatch, TrainBatch],
    dataset,
    rel_filter_threshold=0.4,
    ref_filter_threshold=0.5,
    token_to_idx=None,
):
    # convert entities relation predictions
    batch_pred_entities, batch_pred_relations = convert_predictions(
        output.entity_clf,
        output.rel_clf,
        output.relations,
        batch,
        rel_filter_threshold,
        dataset.data_parser,
        is_ner_rel=True,
    )
    batch_pred_mentions, batch_pred_references = convert_predictions(
        output.mention_clf,
        output.references_clf,
        output.references,
        batch,
        ref_filter_threshold,
        dataset.data_parser,
        is_ner_rel=False,
    )

    return parse_predictions(
        documents=dataset.documents,
        pred_entities=batch_pred_entities,
        pred_relations=batch_pred_relations,
        pred_mentions=batch_pred_mentions,
        pred_references=batch_pred_references,
        token_to_idx=token_to_idx,
    )


def convert_predictions(
    batch_entity_clf: torch.tensor,
    batch_rel_clf: torch.tensor,
    batch_rels: torch.tensor,
    batch: EvalBatch,
    rel_filter_threshold: float,
    data_parser: DataParser,
    no_overlapping: bool = False,
    is_ner_rel: bool = True,
):
    """Supports both entities + relations, and mentions + references.

    Parameters
    ----------
    batch_entity_clf :
    batch_rel_clf :
    batch_rels :
    batch :
    rel_filter_threshold :
    data_parser :
    no_overlapping :
    is_ner_rel :

    Returns
    -------

    """
    # get maximum activation (index of predicted entity type)
    batch_entity_types = batch_entity_clf.argmax(dim=-1)
    # apply entity sample mask
    # NOTE: in eval we have only entity_sample_masks and not mention_sample_masks
    batch_entity_types *= batch.entity_sample_masks.long()

    # apply threshold to relations
    batch_rel_clf[batch_rel_clf < rel_filter_threshold] = 0

    batch_pred_entities = []
    batch_pred_relations = []

    for i in range(batch_rel_clf.shape[0]):
        # get model predictions for sample
        entity_types = batch_entity_types[i]
        entity_spans = batch.entity_spans[i]
        entity_clf = batch_entity_clf[i]
        rel_clf = batch_rel_clf[i]
        rels = batch_rels[i]

        # convert predicted entities
        if is_ner_rel:
            entity_types_func = data_parser.get_entity_type
            rel_types_func = data_parser.get_relation_type

        else:
            entity_types_func = data_parser.get_mention_type
            rel_types_func = data_parser.get_reference_type

        sample_pred_entities = _convert_pred_entities(
            entity_types, entity_spans, entity_clf, entity_type_func=entity_types_func
        )

        # convert predicted relations
        sample_pred_relations = _convert_pred_relations(
            rel_clf,
            rels,
            entity_types,
            entity_spans,
            relation_type_func=rel_types_func,
            entity_type_func=entity_types_func,
        )

        if no_overlapping:
            sample_pred_entities, sample_pred_relations = remove_overlapping(
                sample_pred_entities, sample_pred_relations
            )

        batch_pred_entities.append(sample_pred_entities)
        batch_pred_relations.append(sample_pred_relations)

    return batch_pred_entities, batch_pred_relations


def _convert_pred_entities(
    entity_types: torch.tensor,
    entity_spans: torch.tensor,
    entity_scores: torch.tensor,
    entity_type_func: Callable,
):
    # get entities that are not classified as 'None'
    valid_entity_indices = entity_types.nonzero().view(-1)
    pred_entity_types = entity_types[valid_entity_indices]
    pred_entity_spans = entity_spans[valid_entity_indices]
    pred_entity_scores = torch.gather(
        entity_scores[valid_entity_indices], 1, pred_entity_types.unsqueeze(1)
    ).view(-1)

    # convert to tuples (start, end, type, score)
    converted_preds = []
    for i in range(pred_entity_types.shape[0]):
        label_idx = pred_entity_types[i].item()
        entity_type = entity_type_func(label_idx)

        start, end = pred_entity_spans[i].tolist()
        score = pred_entity_scores[i].item()

        converted_pred = (start, end, entity_type, score)
        converted_preds.append(converted_pred)

    return converted_preds


def _convert_pred_relations(
    rel_clf: torch.tensor,
    rels: torch.tensor,
    entity_types: torch.tensor,
    entity_spans: torch.tensor,
    relation_type_func: Callable,
    entity_type_func: Callable,
):
    rel_class_count = rel_clf.shape[1]
    rel_clf = rel_clf.view(-1)

    # get predicted relation labels and corresponding entity pairs
    rel_nonzero = rel_clf.nonzero().view(-1)
    pred_rel_scores = rel_clf[rel_nonzero]

    pred_rel_types = (
        rel_nonzero % rel_class_count
    ) + 1  # model does not predict None class (+1)
    valid_rel_indices = torch.div(
        rel_nonzero, rel_class_count, rounding_mode="trunc"
    )  # rel_nonzero // rel_class_count
    valid_rels = rels[valid_rel_indices]

    # get masks of entities in relation
    pred_rel_entity_spans = entity_spans[valid_rels].long()

    # get predicted entity types
    pred_rel_entity_types = torch.zeros([valid_rels.shape[0], 2])
    if valid_rels.shape[0] != 0:
        pred_rel_entity_types = torch.stack(
            [entity_types[valid_rels[j]] for j in range(valid_rels.shape[0])]
        )

    # convert to tuples ((head start, head end, head type), (tail start, tail end, tail type), rel type, score))
    converted_rels = []
    check = set()

    for i in range(pred_rel_types.shape[0]):
        label_idx = pred_rel_types[i].item()
        pred_rel_type = relation_type_func(label_idx)
        pred_head_type_idx, pred_tail_type_idx = (
            pred_rel_entity_types[i][0].item(),
            pred_rel_entity_types[i][1].item(),
        )
        pred_head_type = entity_type_func(pred_head_type_idx)
        pred_tail_type = entity_type_func(pred_tail_type_idx)
        score = pred_rel_scores[i].item()

        spans = pred_rel_entity_spans[i]
        head_start, head_end = spans[0].tolist()
        tail_start, tail_end = spans[1].tolist()

        converted_rel = (
            (head_start, head_end, pred_head_type),
            (tail_start, tail_end, pred_tail_type),
            pred_rel_type,
        )
        converted_rel = _adjust_rel(converted_rel)

        if converted_rel not in check:
            check.add(converted_rel)
            converted_rels.append(tuple(list(converted_rel) + [score]))

    return converted_rels


def remove_overlapping(entities, relations):
    non_overlapping_entities = []
    non_overlapping_relations = []

    for entity in entities:
        if not _is_overlapping(entity, entities):
            non_overlapping_entities.append(entity)

    for rel in relations:
        e1, e2 = rel[0], rel[1]
        if not _check_overlap(e1, e2):
            non_overlapping_relations.append(rel)

    return non_overlapping_entities, non_overlapping_relations


def _is_overlapping(e1, entities):
    for e2 in entities:
        if _check_overlap(e1, e2):
            return True

    return False


def _check_overlap(e1, e2):
    if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
        return False
    else:
        return True


def _adjust_rel(rel: Tuple):
    adjusted_rel = rel
    if rel[-1].symmetric:
        head, tail = rel[:2]
        if tail[0] < head[0]:
            adjusted_rel = tail, head, rel[-1]

    return adjusted_rel


def convert_entities(sample_pred_entities, tokens):
    converted_entities = []
    extended_converted_entities = []
    for entity in sample_pred_entities:
        entity_span = entity[:2]
        span_tokens = get_span_tokens(tokens, entity_span)
        entity_type = entity[2].identifier
        converted_entity = dict(
            type=entity_type,
            start=span_tokens[0].index,
            end=span_tokens[-1].index + 1,
        )
        converted_entities.append(converted_entity)
        extended_converted_entity = converted_entity.copy()
        extended_converted_entity.update(
            span=[t.phrase for t in span_tokens.tokens], score=entity[3]
        )
        extended_converted_entities.append(extended_converted_entity)
    converted_entities = sorted(converted_entities, key=lambda e: e["start"])
    extended_converted_entities = sorted(
        extended_converted_entities, key=lambda e: e["start"]
    )

    return converted_entities, extended_converted_entities


def convert_relations(sample_pred_relations, tokens, converted_entities):
    converted_relations = []
    for relation in sample_pred_relations:
        head, tail = relation[:2]
        head_span, head_type = head[:2], head[2].identifier
        tail_span, tail_type = tail[:2], tail[2].identifier
        head_span_tokens = get_span_tokens(tokens, head_span)
        tail_span_tokens = get_span_tokens(tokens, tail_span)
        relation_type = relation[2].identifier

        converted_head = dict(
            type=head_type,
            start=head_span_tokens[0].index,
            end=head_span_tokens[-1].index + 1,
        )
        converted_tail = dict(
            type=tail_type,
            start=tail_span_tokens[0].index,
            end=tail_span_tokens[-1].index + 1,
        )

        head_idx = converted_entities.index(converted_head)
        tail_idx = converted_entities.index(converted_tail)

        converted_relation = dict(
            type=relation_type,
            head=head_idx,
            tail=tail_idx,
            head_span=[t.phrase for t in head_span_tokens.tokens],
            tail_span=[t.phrase for t in tail_span_tokens.tokens],
            score=relation[3],
        )
        converted_relations.append(converted_relation)
    converted_relations = sorted(converted_relations, key=lambda r: r["head"])
    return converted_relations


def parse_predictions(
    documents,
    pred_entities,
    pred_relations,
    pred_mentions,
    pred_references,
    token_to_idx=None,
):
    predictions = []

    for i, doc in enumerate(documents):
        tokens = doc.tokens
        sample_pred_entities = pred_entities[i]
        sample_pred_relations = pred_relations[i]
        sample_pred_mentions = pred_mentions[i]
        sample_pred_references = pred_references[i]

        # convert entities
        converted_entities, extended_converted_entities = convert_entities(
            sample_pred_entities, tokens
        )
        # convert relations
        converted_relations = convert_relations(
            sample_pred_relations, tokens, converted_entities
        )

        # convert mentions
        converted_mentions, extended_converted_mentions = convert_entities(
            sample_pred_mentions, tokens
        )
        # convert references
        converted_references = convert_relations(
            sample_pred_references, tokens, converted_mentions
        )

        clusters = convert_to_clusters(
            mentions=extended_converted_mentions,
            references=converted_references,
            filter_top=True,
        )
        clusters = [
            [
                {"start": c[0][0], "end": c[0][1], "span": c[1], "cluster_id": i}
                for c in cluster
            ]
            for i, cluster in enumerate(clusters)
        ]

        if token_to_idx is not None:
            # adding start and end chars to entities/mentions
            curr_token_to_idx = token_to_idx[i]
            spans = [extended_converted_entities, extended_converted_mentions]
            for span in spans:
                for e in span:
                    start_token = curr_token_to_idx[e["start"]]
                    end_token = curr_token_to_idx[e["end"] - 1]
                    e.update(dict(start_char=start_token[0], end_char=end_token[1]))

        doc_predictions = dict(
            tokens=[t.phrase for t in tokens],
            entities=extended_converted_entities,
            relations=converted_relations,
            mentions=extended_converted_mentions,
            references=converted_references,
            clusters=clusters,
        )
        predictions.append(doc_predictions)

    return predictions

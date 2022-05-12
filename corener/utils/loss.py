from torch import nn


def compute_spans_loss(span_logits, span_types, span_sample_masks, is_spans):
    criterion = nn.CrossEntropyLoss(reduction="none")
    entity_loss = criterion(span_logits.permute(0, 2, 1), span_types)
    entity_loss = (
        (entity_loss * span_sample_masks).sum(1) * is_spans.squeeze()
    ).sum() / (span_sample_masks.sum(1) * is_spans.squeeze()).sum()
    return entity_loss


def compute_relations_loss(rel_logits, rel_types, rel_sample_masks, is_rel):
    rel_criterion = nn.BCEWithLogitsLoss(reduction="none")
    rel_loss = rel_criterion(rel_logits, rel_types).mean(-1)
    rel_loss = ((rel_loss * rel_sample_masks).sum(1) * is_rel.squeeze()).sum() / (
        rel_sample_masks.sum(1) * is_rel.squeeze()
    ).sum()
    return rel_loss


def compute_loss(
    # entity
    entity_logits,
    entity_types,
    entity_sample_masks,
    # mention
    mention_logits,
    mention_types,
    mention_sample_masks,
    # rel
    rel_logits,
    rel_types,
    rel_sample_masks,
    # ref
    ref_logits,
    ref_types,
    ref_sample_masks,
    is_ner,
    is_emd,
    is_re,
    is_cr,
):
    # entity loss
    entity_loss = (entity_logits * 0.0).sum()
    if entity_sample_masks.sum() > 0 and is_ner.sum() > 0:
        entity_loss = compute_spans_loss(
            span_logits=entity_logits,
            span_types=entity_types,
            span_sample_masks=entity_sample_masks,
            is_spans=is_ner,
        )

    mention_loss = (mention_logits * 0.0).sum()
    if mention_sample_masks.sum() > 0 and is_emd.sum() > 0:
        mention_loss = compute_spans_loss(
            span_logits=mention_logits,
            span_types=mention_types,
            span_sample_masks=mention_sample_masks,
            is_spans=is_emd,
        )

    rel_loss = (rel_logits * 0.0).sum()
    if rel_sample_masks.sum() > 0 and is_re.sum() > 0:
        rel_loss = compute_relations_loss(
            rel_logits=rel_logits,
            rel_types=rel_types,
            rel_sample_masks=rel_sample_masks,
            is_rel=is_re,
        )

    ref_loss = (ref_logits * 0.0).sum()
    if ref_sample_masks.sum() > 0 and is_cr.sum() > 0:
        ref_loss = compute_relations_loss(
            rel_logits=ref_logits,
            rel_types=ref_types,
            rel_sample_masks=ref_sample_masks,
            is_rel=is_cr,
        )

    return [
        entity_loss,
        mention_loss,
        rel_loss,
        ref_loss,
    ]

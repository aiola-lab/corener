import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from corener.data import sampling
from corener.data.dataset import MTLDataset
from corener.models import Corener, ModelOutput
from corener.utils import get_device, prediction


@dataclass
class InferenceInput:
    data: Union[List[str], str]
    device: Any = get_device()
    rel_filter_threshold: float = 0.4
    ref_filter_threshold: float = 0.5
    batch_size: int = 1
    num_workers: int = 0


def text_to_tokens(documents: List[str], nlp):
    """Convert text to tokens and create a token to char index dict for each doc.

    Parameters
    ----------
    documents :
    nlp : Spacy language model

    Returns
    -------

    """
    docs = [nlp(doc) for doc in documents]
    parsed_docs = [[t.text for t in doc] for doc in docs]
    token_to_idx = [
        {token.i: (token.idx, token.idx + len(token.text), token.text) for token in doc}
        for doc in docs
    ]
    return parsed_docs, token_to_idx


def load_pretrained_model(
    artifact_path: str, device=None, max_context_size: int = None
):
    """

    Parameters
    ----------
    artifact_path :
    device :
    max_context_size : reset of context size (number of tokens) for inference on large docs.

    Returns
    -------

    """
    tokenizer = AutoTokenizer.from_pretrained(artifact_path)
    model = Corener.from_pretrained(artifact_path)

    if (
        max_context_size is not None
        and max_context_size > model.backbone.config.max_position_embeddings
    ):
        old = model.backbone.embeddings.position_embeddings
        new = nn.Embedding(
            max_context_size,
            model.backbone.config.hidden_size,
            padding_idx=old.padding_idx,
        )
        new.weight.data[
            : model.backbone.config.max_position_embeddings, :
        ] = old.weight.data
        model.backbone.embeddings.position_embeddings = new
        model.backbone.embeddings.register_buffer(
            "position_ids", torch.arange(max_context_size).expand((1, -1))
        )
        model.backbone.embeddings.register_buffer(
            "token_type_ids", torch.zeros(1, max_context_size, dtype=torch.int64)
        )
        model.backbone.config.max_position_embeddings = max_context_size
        model = model.to(device)

    dataset = MTLDataset(
        types=model.config.types,
        tokenizer=tokenizer,
        train_mode=False,
    )

    return model, dataset, tokenizer


@torch.no_grad()
def run_inference(
    config: InferenceInput, model: Corener, dataset: MTLDataset, tokenizer
):
    data = args.input
    if not isinstance(data, list):
        data = [data]
    data, token_to_idx = text_to_tokens(data, dataset.data_parser._nlp)

    dataset.read_dataset(data)
    max_doc_contexts = max([len(doc.encoding) for doc in dataset.documents])
    if max_doc_contexts > model.backbone.config.max_position_embeddings:
        raise ValueError(
            f"Max context size (number of tokens) is {model.backbone.config.max_position_embeddings}, "
            f"however a context of {max_doc_contexts} was given. Please break the documents into smaller chunks."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        collate_fn=sampling.partial_collate_fn_padding(
            encoding_padding=tokenizer.pad_token_id
        ),
    )

    pred_entities = []
    pred_relations = []
    pred_mentions = []
    pred_references = []

    model = model.to(config.device)
    model.eval()

    for batch in tqdm(dataloader):
        batch = batch.to(config.device)
        output: ModelOutput = model(
            input_ids=batch.encodings,
            context_masks=batch.context_masks,
            entity_masks=batch.entity_masks,
            entity_sizes=batch.entity_sizes,
            entity_spans=batch.entity_spans,
            entity_sample_masks=batch.entity_sample_masks,
            inference=True,
        )

        # convert entities relation predictions
        batch_pred_entities, batch_pred_relations = prediction.convert_predictions(
            output.entity_clf,
            output.rel_clf,
            output.relations,
            batch,
            config.rel_filter_threshold,
            dataset.data_parser,
            is_ner_rel=True,
        )
        pred_entities.extend(batch_pred_entities)
        pred_relations.extend(batch_pred_relations)

        # convert mentions references predictions
        batch_pred_mentions, batch_pred_references = prediction.convert_predictions(
            output.mention_clf,
            output.references_clf,
            output.references,
            batch,
            config.ref_filter_threshold,
            dataset.data_parser,
            is_ner_rel=False,
        )
        pred_mentions.extend(batch_pred_mentions)
        pred_references.extend(batch_pred_references)

    predictions = prediction.parse_predictions(
        documents=dataset.documents,
        pred_entities=pred_entities,
        pred_relations=pred_relations,
        pred_mentions=pred_mentions,
        pred_references=pred_references,
        token_to_idx=token_to_idx,
    )

    return predictions


def main(args):
    device = get_device(gpus=args.gpu, no_cuda=args.no_cuda)

    model, dataset, tokenizer = load_pretrained_model(
        args.artifact_path, device=device, max_context_size=args.max_context_size
    )
    config = InferenceInput(
        data=args.input,
        device=device,
        rel_filter_threshold=args.rel_filter_threshold,
        ref_filter_threshold=args.ref_filter_threshold,
        batch_size=args.batch_size,
    )
    predictions = run_inference(
        config=config, model=model, dataset=dataset, tokenizer=tokenizer
    )
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="CoReNer inference",
    )
    parser.add_argument(
        "--rel-filter-threshold",
        type=float,
        default=0.4,
        help="Filter threshold for relations",
    )
    parser.add_argument(
        "--ref-filter-threshold",
        type=float,
        default=0.5,
        help="Filter threshold for references",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Optional, sentence for inference",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        help="Path to cached model/tokenizer etc.",
    )
    parser.add_argument(
        "--max-context-size",
        type=int,
        default=None,
        help="reset of context size (number of tokens) for inference on large docs.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="No cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation/Prediction batch size",
    )
    args = parser.parse_args()

    main(args)

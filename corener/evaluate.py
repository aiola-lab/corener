from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from corener.data import MTLDataset, sampling
from corener.models import Corener, ModelOutput
from corener.utils import get_device
from corener.utils.evaluation import EntityRelEvaluator


def load_pretrained_model(
    artifact_path: str,
    device=None,
):
    """

    Parameters
    ----------
    artifact_path :
    device :

    Returns
    -------

    """
    tokenizer = AutoTokenizer.from_pretrained(artifact_path)
    model = Corener.from_pretrained(artifact_path, map_location=device)
    dataset = MTLDataset(
        types=(Path(artifact_path) / "types.json").as_posix(),
        tokenizer=tokenizer,
        train_mode=False,
    )

    return model, dataset, tokenizer


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    rel_filter_threshold,
    ref_filter_threshold,
    no_overlapping=False,
):
    # NOTE: workaround to insure examples are not shuffled
    assert isinstance(
        dataloader.sampler, torch.utils.data.sampler.SequentialSampler
    ), "dataset can't be shuffled for evaluation"
    model.eval()
    model = model.to(device)

    ner_rel_evaluator = EntityRelEvaluator(
        dataloader.dataset,
        rel_filter_threshold=rel_filter_threshold,
        no_overlapping=no_overlapping,
        relations_name="relations",
        spans_name="entities",
        is_ner_rel=True,
    )

    emd_cr_evaluator = EntityRelEvaluator(
        dataloader.dataset,
        rel_filter_threshold=ref_filter_threshold,
        no_overlapping=no_overlapping,
        relations_name="references",
        spans_name="mentions",
        is_ner_rel=False,
    )

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        output: ModelOutput = model(
            input_ids=batch.encodings,
            context_masks=batch.context_masks,
            entity_masks=batch.entity_masks,
            entity_sizes=batch.entity_sizes,
            entity_spans=batch.entity_spans,
            entity_sample_masks=batch.entity_sample_masks,
            inference=True,
        )

        ner_rel_evaluator.eval_batch(
            output.entity_clf, output.rel_clf, output.relations, batch
        )
        emd_cr_evaluator.eval_batch(
            output.mention_clf, output.references_clf, output.references, batch
        )

    ner_eval, rel_eval, rel_nec_eval = ner_rel_evaluator.compute_scores()
    emd_eval, ref_eval, ref_emd_eval = emd_cr_evaluator.compute_scores()
    # todo: also return coref metrics
    return ner_eval, rel_eval, rel_nec_eval, emd_eval, ref_eval, ref_emd_eval


def main(args):
    device = get_device(gpus=args.gpu)
    model, dataset, tokenizer = load_pretrained_model(args.artifact_path, device=device)
    dataset.read_dataset(args.data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=sampling.partial_collate_fn_padding(
            encoding_padding=tokenizer.pad_token_id
        ),
    )
    ner_eval, rel_eval, rel_nec_eval, emd_eval, ref_eval, ref_emd_eval = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        rel_filter_threshold=args.rel_filter_threshold,
        ref_filter_threshold=args.ref_filter_threshold,
        no_overlapping=args.no_overlapping,
    )


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
        "--data-path",
        type=str,
        help="Path to eval. set.",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        help="Path to cached model/tokenizer etc.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--no-overlapping",
        action="store_true",
        default=False,
        help="If true, do not evaluate on overlapping entities "
        "and relations with overlapping entities",
    )
    args = parser.parse_args()

    main(args)

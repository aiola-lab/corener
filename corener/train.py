import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from corener.data import MTLDataset, sampling
from corener.evaluate import evaluate
from corener.models import Corener, ModelOutput
from corener.utils import common_parser, get_device, get_optimizer_params, set_seed
from corener.utils.loss import compute_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):
    # device
    device = get_device(gpus=args.gpu)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case=args.lowercase, cache_dir=args.cache_path
    )

    # load data
    train_dataset = MTLDataset(
        dataset_or_path=args.train_path,
        types=args.types_path,
        tokenizer=tokenizer,
        train_mode=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=sampling.partial_collate_fn_padding(
            encoding_padding=tokenizer.pad_token_id
        ),
    )

    val_loader = None
    if args.val_path is not None and args.do_eval:
        val_dataset = MTLDataset(
            dataset_or_path=args.val_path,
            types=args.types_path,
            tokenizer=tokenizer,
            train_mode=False,  # eval mode
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            collate_fn=sampling.partial_collate_fn_padding(
                encoding_padding=tokenizer.pad_token_id
            ),
        )

    # model
    model = Corener(
        backbone_model_name_or_path=args.model_name_or_path,
        ner_classes=train_dataset.data_parser.entity_type_count,
        # removing the None relation since we do binary classification for each relation to support
        # multiple relations per ner-pair.
        relation_classes=train_dataset.data_parser.relation_type_count - 1,
        cls_token=tokenizer.cls_token_id,
        pad_token=tokenizer.pad_token_id,
        size_embedding=args.size_embedding,
        max_pairs=args.max_pairs,
        cache_dir=args.cache_path,
    )
    # multi-gpu
    # todo: remove data parallel? (since it breaks eval with bs > 1)
    model = nn.DataParallel(model)
    model = model.to(device)

    # create optimizer
    optimizer_params = get_optimizer_params(model, weight_decay=args.weight_decay)
    optimizer = AdamW(
        optimizer_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        correct_bias=False,
    )
    # create scheduler
    updates_total = len(train_loader) * args.n_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup * updates_total,
        num_training_steps=updates_total,
    )
    step_scheduler = None
    if args.milestones is not None:
        step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=0.5
        )

    # training loop
    with tqdm(range(args.n_epoch), total=args.n_epoch) as epoch_iterator:
        nbatchs = len(train_loader)
        for epoch in epoch_iterator:
            for i, batch in enumerate(train_loader):
                model.train()
                model.zero_grad()
                batch = batch.to(device)

                output: ModelOutput = model(
                    input_ids=batch.encodings,
                    context_masks=batch.context_masks,
                    entity_masks=batch.entity_masks,
                    entity_sizes=batch.entity_sizes,
                    relations=batch.rels,
                    relations_masks=batch.rel_masks,
                    mention_masks=batch.mention_masks,
                    mention_sizes=batch.mention_sizes,
                    references=batch.refs,
                    references_masks=batch.ref_masks,
                )

                losses = compute_loss(
                    # NER + RE
                    entity_logits=output.entity_clf,
                    entity_types=batch.entity_types,
                    entity_sample_masks=batch.entity_sample_masks,
                    rel_logits=output.rel_clf,
                    rel_types=batch.rel_types,
                    rel_sample_masks=batch.rel_sample_masks,
                    # EMD + CR
                    mention_logits=output.mention_clf,
                    mention_types=batch.mention_types,
                    mention_sample_masks=batch.mention_sample_masks,
                    ref_logits=output.references_clf,
                    ref_types=batch.ref_types,
                    ref_sample_masks=batch.ref_sample_masks,
                    # binary masks
                    is_ner=batch.is_ner,
                    is_emd=batch.is_emd,
                    is_re=batch.is_re,
                    is_cr=batch.is_cr,
                )

                loss = torch.stack(losses).mean()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                epoch_iterator.set_description(
                    f"[{epoch+1} {i+1}/{nbatchs}] avg train loss: {loss.item():.3f}"
                )

            if (
                args.do_eval
                and val_loader is not None
                and (epoch + 1) % args.eval_every == 0
            ):
                _ = evaluate(
                    model,
                    dataloader=val_loader,
                    device=device,
                    rel_filter_threshold=args.rel_filter_threshold,
                    ref_filter_threshold=args.ref_filter_threshold,
                    no_overlapping=args.no_overlapping,
                )

        if args.milestones is not None:
            step_scheduler.step()

    out_path = Path(args.artifact_path)
    # save model
    model.module.save_pretrained(
        args.artifact_path, types=train_dataset.data_parser.types
    )
    tokenizer.save_pretrained(args.artifact_path)
    # save training args
    with open(out_path / "training_args.json", "w") as f:
        json.dump(args.__dict__, f)
    # save entity types
    with open(out_path / "types.json", "w") as f:
        json.dump(train_dataset.data_parser.types, f)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="CoReNer trainer", parents=[common_parser])
    parser.add_argument(
        "--train-path",
        type=str,
        help="training dataset path",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="validation dataset path",
    )
    parser.add_argument(
        "--types-path",
        type=str,
        help="label types (entities and relations)",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="robert-base",
        help="pretrained model name or model path",
    )
    parser.add_argument(
        "--max-span-size", type=int, default=10, help="Maximum size of spans"
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        default=False,
        help="If true, input is lowercased during preprocessing",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of sampling processes. 0 = no multiprocessing for sampling",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=1000,
        help="Maximum entity pairs to process during training/evaluation",
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
        "--size-embedding",
        type=int,
        default=25,
        help="Dimensionality of size embedding",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="./artifacts",
        help="Path to directory where model checkpoints are stored",
    )

    # Model / Training
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument(
        "--train-batch-size", type=int, default=40, help="Training batch size"
    )
    parser.add_argument("--n-epoch", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--neg-entity-count",
        type=int,
        default=100,
        help="Number of negative entity samples per document (sentence)",
    )
    parser.add_argument(
        "--neg-relation-count",
        type=int,
        default=100,
        help="Number of negative relation samples per document (sentence)",
    )
    parser.add_argument("--lr", type=float, default=7.5e-5, help="Learning rate")
    parser.add_argument(
        "--lr-warmup",
        type=float,
        default=0.1,
        help="Proportion of total train iterations to warmup in linear increase/decrease schedule",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=None,
        help="Steps epochs for step scheduler.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay to apply"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm"
    )
    parser.add_argument(
        "--no-overlapping",
        action="store_true",
        default=False,
        help="If true, do not evaluate on overlapping entities "
        "and relations with overlapping entities",
    )

    # Val
    parser.add_argument("--do-eval", action="store_true", default=False)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluation freq",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Evaluation batch size",
    )
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to cache transformer models (HuggingFace)",
    )

    args = parser.parse_args()

    # seed
    set_seed(args.seed)

    # run training
    main(args)

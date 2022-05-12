import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, logging
from transformers.file_utils import ModelOutput as HFModelOutput

from corener.data.sampling import create_rel_mask
from corener.utils.data import batch_index, get_token, padded_stack


class Corener(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        ner_classes: int,
        relation_classes: int,
        cls_token: int,
        pad_token: int,
        size_embedding: int = 32,
        max_pairs: int = 100,
        cache_dir: str = None,
    ):
        super().__init__()

        self.cls_token = cls_token
        self.pad_token = pad_token
        self._max_pairs = max_pairs
        self.ner_classes = ner_classes
        self.relation_classes = relation_classes
        self.size_embeddings_dim = size_embedding

        self.backbone = AutoModel.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, add_pooling_layer=False
        )
        # update config
        corener_config = dict(
            ner_classes=self.ner_classes,
            relation_classes=self.relation_classes,
            cls_token=self.cls_token,
            pad_token=self.pad_token,
            size_embedding=self.size_embeddings_dim,
            max_pairs=self._max_pairs,
        )
        self.backbone.config.update(dict(corener_config=corener_config))
        self.config = self.backbone.config

        self.dropout = nn.Dropout(self.backbone.config.hidden_dropout_prob)

        # ner
        self.ner_size_embeddings = nn.Embedding(100, size_embedding)
        self.ner_rep = nn.Linear(
            self.backbone.config.hidden_size * 2 + size_embedding,
            self.backbone.config.hidden_size,
        )
        self.ner_head = nn.Sequential(
            nn.ReLU(), nn.Linear(self.backbone.config.hidden_size, ner_classes)
        )
        self.ner_classifier = nn.Sequential(self.ner_rep, self.ner_head)
        # emd
        self.emd_size_embeddings = nn.Embedding(100, size_embedding)
        self.emd_rep = nn.Linear(
            self.backbone.config.hidden_size * 2 + size_embedding,
            self.backbone.config.hidden_size,
        )
        self.emd_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.config.hidden_size, 2)
            # NOTE: hard-coded for now. If we, for some reason, want to support several mention types,
            # we should change it here and in the corresponding loss
        )
        self.emd_classifier = nn.Sequential(self.emd_rep, self.emd_head)

        # relations
        self.rel_classifier = nn.Sequential(
            nn.Linear(
                self.backbone.config.hidden_size * 3 + size_embedding * 2,
                self.backbone.config.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                self.backbone.config.hidden_size,
                self.backbone.config.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                self.backbone.config.hidden_size,
                relation_classes,
            ),
        )

        # co-reference
        self.cr_classifier = nn.Sequential(
            nn.Linear(
                self.backbone.config.hidden_size * 3 + size_embedding * 2,
                self.backbone.config.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                self.backbone.config.hidden_size,
                self.backbone.config.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                self.backbone.config.hidden_size,
                1,  # NOTE: hard-coded for now. If we, for some reason, want to support several co-ref types,
                # we should change it here and in the corresponding loss
            ),
        )

    def save_pretrained(self, path, **kwargs):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin", **kwargs)
        self.config.update(**kwargs)
        self.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path, map_location=None):
        """

        Parameters
        ----------
        path : path to artifacts folder
        map_location :

        Returns
        -------

        """
        path = Path(path)
        with open(path / "config.json", "r") as config_file:
            config = json.load(config_file)
            corener_config = config["corener_config"]

        logging.set_verbosity_error()
        # supress warning: Some weights of the model checkpoint...
        model = Corener(
            model_name_or_path=path.as_posix(),
            ner_classes=corener_config["ner_classes"],
            relation_classes=corener_config["relation_classes"],
            cls_token=corener_config["cls_token"],
            pad_token=corener_config["pad_token"],
            size_embedding=corener_config["size_embedding"],
            max_pairs=corener_config["max_pairs"],
        )
        # go back to warning level...
        logging.set_verbosity_warning()

        model.load_state_dict(
            torch.load(path / "pytorch_model.bin", map_location=map_location)
        )
        return model

    def _classify_spans(
        self, input_ids, token_embedding, spans_masks, size_embeddings, classifier
    ):
        """
        Classify spans (NER, EMD)

        Parameters
        ----------
        input_ids :
        token_embedding :
        spans_masks :
        size_embeddings :
        classifier : NER/EMD classifier head

        Returns
        -------

        """
        # max pool entity candidate spans
        m = (spans_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + token_embedding.unsqueeze(1).repeat(
            1, spans_masks.shape[1], 1, 1
        )
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # get cls token as candidate context representation
        entity_ctx = get_token(token_embedding, input_ids, self.cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_representation = torch.cat(
            [
                entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                entity_spans_pool,
                size_embeddings,
            ],
            dim=2,
        )
        entity_representation = self.dropout(entity_representation)

        # classify entity candidates
        entity_clf = classifier(entity_representation)
        return entity_clf, entity_spans_pool, entity_representation

    def _classify_relations_chunk(
        self,
        entity_spans,
        size_embeddings,
        chunk_relations,
        chunk_rel_masks,
        chunk_token_embedding,
        classifier,
    ):
        batch_size = chunk_relations.shape[0]
        # get pairs of entity candidate representations
        entity_pairs = batch_index(entity_spans, chunk_relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = batch_index(size_embeddings, chunk_relations)
        size_pair_embeddings = size_pair_embeddings.view(
            batch_size, size_pair_embeddings.shape[1], -1
        )

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((chunk_rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + chunk_token_embedding
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[chunk_rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = classifier(rel_repr)
        return chunk_rel_logits

    def _classify_relations(
        self,
        entity_spans,
        size_embeddings,
        relations,
        rel_masks,
        token_embedding,
        classifier,
    ):
        """Classify relations/references

        Parameters
        ----------
        entity_spans :
        size_embeddings :
        relations :
        rel_masks :
        token_embedding :
        classifier :

        Returns
        -------

        """
        batch_size = relations.shape[0]

        relations_logits = torch.zeros(
            [
                batch_size,
                relations.shape[1],
                classifier[-1].weight.shape[0],
            ]  # classifier.weight.shape[0]]
        ).to(
            classifier[-1].weight
        )  # classifier.weight.device)

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            # obtain relation logits
            # chunk processing to reduce memory usage
            for chunk_start in range(0, relations.shape[1], self._max_pairs):
                chunk_relations = relations[
                    :, chunk_start : chunk_start + self._max_pairs
                ]
                chunk_rel_masks = rel_masks[
                    :, chunk_start : chunk_start + self._max_pairs
                ]
                chunk_token_embedding = token_embedding[
                    :, : chunk_relations.shape[1], :
                ]

                chunk_rel_logits = self._classify_relations_chunk(
                    entity_spans=entity_spans,
                    size_embeddings=size_embeddings,
                    chunk_relations=chunk_relations,
                    chunk_rel_masks=chunk_rel_masks,
                    chunk_token_embedding=chunk_token_embedding,
                    classifier=classifier,
                )
                relations_logits[
                    :, chunk_start : chunk_start + self._max_pairs, :
                ] = chunk_rel_logits
        else:
            chunk_rel_logits = self._classify_relations_chunk(
                entity_spans=entity_spans,
                size_embeddings=size_embeddings,
                chunk_relations=relations,
                chunk_rel_masks=rel_masks,
                chunk_token_embedding=token_embedding,
                classifier=classifier,
            )
            relations_logits[:, 0 : 0 + self._max_pairs, :] = chunk_rel_logits

        return relations_logits

    def _forward_train(
        self,
        input_ids: torch.tensor,
        context_masks: torch.tensor,
        entity_masks: torch.tensor,
        entity_sizes: torch.tensor,
        mention_masks: torch.tensor,
        mention_sizes: torch.tensor,
        relations: torch.tensor,
        rel_masks: torch.tensor,
        references: torch.tensor,
        references_masks: torch.tensor,
    ):

        # get contextualized token embeddings from last transformer layer
        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=context_masks,
        )
        # bs, seq_len, hidden_dim
        token_embedding = backbone_out[0]

        # NER + RE
        size_embeddings = self.ner_size_embeddings(
            entity_sizes
        )  # embed entity candidate sizes
        # classify entities
        entity_clf, entity_spans_pool, entity_representation = self._classify_spans(
            input_ids,
            token_embedding,
            entity_masks,
            size_embeddings,
            classifier=self.ner_classifier,
        )

        # classify relations
        token_embedding_large = token_embedding.unsqueeze(1).repeat(
            1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1
        )

        # obtain relation logits
        rel_clf = self._classify_relations(
            entity_spans_pool,
            size_embeddings,
            relations,
            rel_masks,
            token_embedding_large,
            classifier=self.rel_classifier,
        )

        # EMD + CR
        size_embeddings = self.emd_size_embeddings(
            mention_sizes
        )  # embed mention candidate sizes
        # classify mentions
        mention_clf, mention_spans_pool, mention_representation = self._classify_spans(
            input_ids,
            token_embedding,
            mention_masks,
            size_embeddings,
            classifier=self.emd_classifier,
        )

        # classify relations
        token_embedding_large = token_embedding.unsqueeze(1).repeat(
            1, max(min(references.shape[1], self._max_pairs), 1), 1, 1
        )

        # obtain relation logits
        ref_clf = self._classify_relations(
            mention_spans_pool,
            size_embeddings,
            references,
            references_masks,
            token_embedding_large,
            classifier=self.cr_classifier,
        )

        return ModelOutput(
            entity_clf=entity_clf,
            rel_clf=rel_clf,
            mention_clf=mention_clf,
            references_clf=ref_clf,
        )

    def _forward_inference(
        self,
        input_ids: torch.tensor,
        context_masks: torch.tensor,
        entity_masks: torch.tensor,
        entity_sizes: torch.tensor,
        entity_spans: torch.tensor,
        entity_sample_masks: torch.tensor,
    ):
        # get contextualized token embeddings from last transformer layer
        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=context_masks,
        )
        # bs, seq_len, hidden_dim
        token_embedding = backbone_out[0]
        ctx_size = context_masks.shape[-1]

        size_embeddings = self.ner_size_embeddings(
            entity_sizes
        )  # embed entity candidate sizes

        # NER + RE
        # classify entities
        entity_clf, entity_spans_pool, entity_representation = self._classify_spans(
            input_ids,
            token_embedding,
            entity_masks,
            size_embeddings,
            classifier=self.ner_classifier,
        )

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(
            entity_clf,
            entity_spans,
            entity_sample_masks,
            ctx_size,
            device=self.rel_classifier[-1].weight.device,
        )

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        token_embedding_large = token_embedding.unsqueeze(1).repeat(
            1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1
        )

        # obtain relation logits
        rel_logits = self._classify_relations(
            entity_spans_pool,
            size_embeddings,
            relations,
            rel_masks,
            token_embedding_large,
            classifier=self.rel_classifier,
        )
        rel_clf = torch.sigmoid(rel_logits)
        rel_clf = rel_clf * rel_sample_masks  # mask

        # EMD + CR
        size_embeddings = self.emd_size_embeddings(
            entity_sizes
        )  # embed entity candidate sizes
        # classify mentions
        mention_clf, mention_spans_pool, mention_representation = self._classify_spans(
            input_ids,
            token_embedding,
            entity_masks,
            size_embeddings,
            classifier=self.emd_classifier,
        )

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        references, ref_masks, ref_sample_masks = self._filter_spans(
            mention_clf,
            entity_spans,
            entity_sample_masks,
            ctx_size,
            device=self.cr_classifier[-1].weight.device,
        )

        # apply softmax
        mention_clf = torch.softmax(mention_clf, dim=2)

        ref_sample_masks = ref_sample_masks.float().unsqueeze(-1)
        token_embedding_large = token_embedding.unsqueeze(1).repeat(
            1, max(min(references.shape[1], self._max_pairs), 1), 1, 1
        )

        # obtain relation logits
        ref_logits = self._classify_relations(
            mention_spans_pool,
            size_embeddings,
            references,
            ref_masks,
            token_embedding_large,
            classifier=self.cr_classifier,
        )
        ref_clf = torch.sigmoid(ref_logits)
        ref_clf = ref_clf * ref_sample_masks  # mask

        return ModelOutput(
            entity_clf=entity_clf,
            rel_clf=rel_clf,
            mention_clf=mention_clf,
            references_clf=ref_clf,
            relations=relations,
            references=references,
        )

    @staticmethod
    def _filter_spans(
        entity_clf,
        entity_spans,
        entity_sample_masks,
        ctx_size,
        device,
    ):
        batch_size = entity_clf.shape[0]
        entity_logits_max = (
            entity_clf.argmax(dim=-1) * entity_sample_masks.long()
        )  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(
                    torch.tensor(sample_masks, dtype=torch.bool)
                )

        # stack
        batch_relations = padded_stack(batch_relations, padding=0).to(device)
        batch_rel_masks = padded_stack(batch_rel_masks, padding=0).to(device)
        batch_rel_sample_masks = padded_stack(batch_rel_sample_masks, padding=0).to(
            device
        )

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(
        self,
        input_ids: torch.tensor,
        context_masks: torch.tensor,
        entity_masks: torch.tensor,
        entity_sizes: torch.tensor,
        mention_masks: torch.tensor = None,
        mention_sizes: torch.tensor = None,
        relations: torch.tensor = None,
        relations_masks: torch.tensor = None,
        references: torch.tensor = None,
        references_masks: torch.tensor = None,
        entity_sample_masks: torch.tensor = None,
        entity_spans: torch.tensor = None,
        inference=False,
    ):

        if not inference:
            return self._forward_train(
                input_ids=input_ids,
                context_masks=context_masks,
                entity_masks=entity_masks,
                entity_sizes=entity_sizes,
                mention_masks=mention_masks,
                mention_sizes=mention_sizes,
                relations=relations,
                rel_masks=relations_masks,
                references=references,
                references_masks=references_masks,
            )
        else:
            return self._forward_inference(
                input_ids=input_ids,
                context_masks=context_masks,
                entity_masks=entity_masks,
                entity_sizes=entity_sizes,
                entity_spans=entity_spans,
                entity_sample_masks=entity_sample_masks,
            )


class ModelOutput(HFModelOutput):
    # todo: change forward and forward inference so that entity_clf and rel_clf would contain logits ?
    entity_clf: torch.tensor
    rel_clf: torch.tensor
    mention_clf: torch.tensor
    references_clf: torch.tensor
    relations: Optional[torch.tensor] = None
    references: Optional[torch.tensor] = None

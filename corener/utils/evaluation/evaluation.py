from typing import List, Tuple

import torch
from sklearn.metrics import precision_recall_fscore_support as prfs

from corener.data import MTLDataset
from corener.data.entities import Document, EntityType
from corener.utils import prediction
from corener.utils.clusters import references_to_clusters
from corener.utils.data import EvalBatch
from corener.utils.evaluation.coref_metrics import CorefEvaluator


class Evaluator:
    def __init__(
        self,
        dataset: MTLDataset,
        rel_filter_threshold: float,
        no_overlapping: bool,
    ):
        self._dataset = dataset
        self._rel_filter_threshold = rel_filter_threshold
        self._no_overlapping = no_overlapping

        # relations
        self._gt_relations = []  # ground truth
        self._pred_relations = []  # predictions

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # predictions

        self._pseudo_entity_type = EntityType(
            "Entity", 1, "Entity", "Entity"
        )  # for span only evaluation where don't care about entity type

        self._convert_gt(self._dataset.documents)

    def eval_batch(
        self,
        batch_entity_clf: torch.tensor,
        batch_rel_clf: torch.tensor,
        batch_rels: torch.tensor,
        batch: EvalBatch,
    ):
        batch_pred_entities, batch_pred_relations = prediction.convert_predictions(
            batch_entity_clf,
            batch_rel_clf,
            batch_rels,
            batch,
            self._rel_filter_threshold,
            self._dataset.data_parser,
            no_overlapping=self._no_overlapping,
        )

        self._pred_entities.extend(batch_pred_entities)
        self._pred_relations.extend(batch_pred_relations)

    def compute_scores(self):
        print("")
        print("--- Entities (named entity recognition (NER)) ---")
        print(
            "An entity is considered correct if the entity type and span is predicted correctly"
        )
        gt, pred = self._convert_by_setting(
            self._gt_entities, self._pred_entities, include_entity_types=True
        )
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- Relations ---")
        gt, pred = self._convert_by_setting(
            self._gt_relations, self._pred_relations, include_entity_types=False
        )
        rel_eval = self._score(gt, pred, print_results=False)

        print("")
        print("With named entity classification (NEC)")
        print(
            "A relation is considered correct if the relation type and the two "
            "related entities are predicted correctly (in span and entity type)"
        )
        print("")
        gt, pred = self._convert_by_setting(
            self._gt_relations, self._pred_relations, include_entity_types=True
        )
        rel_nec_eval = self._score(gt, pred, print_results=True)

        return ner_eval, rel_eval, rel_nec_eval

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_relations = doc.relations
            gt_entities = doc.entities

            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_gt_entities = [entity.as_tuple() for entity in gt_entities]
            sample_gt_relations = [rel.as_tuple() for rel in gt_relations]

            if self._no_overlapping:
                sample_gt_entities, sample_gt_relations = prediction.remove_overlapping(
                    sample_gt_entities, sample_gt_relations
                )

            self._gt_entities.append(sample_gt_entities)
            self._gt_relations.append(sample_gt_relations)

    def _convert_by_setting(
        self,
        gt: List[List[Tuple]],
        pred: List[List[Tuple]],
        include_entity_types: bool = True,
        include_score: bool = False,
    ):
        assert len(gt) == len(pred)

        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # relation
                    c = [
                        (t[0][0], t[0][1], self._pseudo_entity_type),
                        (t[1][0], t[1][1], self._pseudo_entity_type),
                        t[2],
                    ]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include predictions scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _score(
        self,
        gt: List[List[Tuple]],
        pred: List[List[Tuple]],
        print_results: bool = False,
    ):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        # todo: remove if removing ignored index
        ignore_index = -100
        types = [t for t in types if t.index != ignore_index]

        labels = [t.index for t in types]
        pred_all = [p for i, p in enumerate(pred_all) if gt_all[i] != ignore_index]
        gt_all = [gt for gt in gt_all if gt != -100]

        per_type = prfs(gt_all, pred_all, labels=labels, average=None, zero_division=0)

        micro = prfs(gt_all, pred_all, labels=labels, average="micro", zero_division=0)[
            :-1
        ]
        macro = prfs(gt_all, pred_all, labels=labels, average="macro", zero_division=0)[
            :-1
        ]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(
                per_type,
                list(micro) + [total_support],
                list(macro) + [total_support],
                types,
            )

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ("type", "precision", "recall", "f1-score", "support")

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, "\n"]

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append("\n")

        results.append("\n")

        # micro
        results.append(row_fmt % self._get_row(micro, "micro"))
        results.append("\n")

        # macro
        results.append(row_fmt % self._get_row(macro, "macro"))

        results_str = "".join(results)
        print(results_str)

    @staticmethod
    def _get_row(data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)


class EntityRelEvaluator(Evaluator):
    def __init__(
        self,
        dataset: MTLDataset,
        rel_filter_threshold: float,
        no_overlapping: bool,
        relations_name="references",
        spans_name="mentions",
        is_ner_rel: bool = False,
    ):
        self.is_ner_rel = is_ner_rel
        self.relations_name = relations_name
        self.spans_name = spans_name

        if not is_ner_rel:
            self.coref_evaluator = CorefEvaluator()
            self.coref_evaluator_top = CorefEvaluator()
            self._gt_clusters = []  # ground truth
            self._pred_clusters = []  # predictions
            self._pred_clusters_top = []  # predictions

        super(EntityRelEvaluator, self).__init__(
            dataset=dataset,
            rel_filter_threshold=rel_filter_threshold,
            no_overlapping=no_overlapping,
        )

    def compute_scores(self):
        print("")
        print(f"--- {self.spans_name.title()} classification ---")
        print(
            "An entity is considered correct if the entity type and span is predicted correctly"
        )
        print("")
        gt, pred = self._convert_by_setting(
            self._gt_entities, self._pred_entities, include_entity_types=True
        )
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print(f"--- {self.relations_name.title()} ---")
        gt, pred = self._convert_by_setting(
            self._gt_relations, self._pred_relations, include_entity_types=False
        )
        rel_eval = self._score(gt, pred, print_results=False)
        print("")
        print(f"With {self.spans_name} classification")
        print(
            "A relation is considered correct if the relation type and the two "
            "related entities are predicted correctly (in span and entity type)"
        )
        print("")
        gt, pred = self._convert_by_setting(
            self._gt_relations, self._pred_relations, include_entity_types=True
        )
        rel_nec_eval = self._score(gt, pred, print_results=True)

        # evaluate coref if available
        if not self.is_ner_rel:
            for i in range(len(self._gt_clusters)):
                mention_to_predicted = {
                    m: cluster for cluster in self._pred_clusters[i] for m in cluster
                }
                mention_to_gold = {
                    m: cluster for cluster in self._gt_clusters[i] for m in cluster
                }
                if self._dataset.documents[i].is_cr:
                    self.coref_evaluator.update(
                        predicted=self._pred_clusters[i],
                        gold=self._gt_clusters[i],
                        mention_to_predicted=mention_to_predicted,
                        mention_to_gold=mention_to_gold,
                    )

                    self.coref_evaluator_top.update(
                        predicted=self._pred_clusters_top[i],
                        gold=self._gt_clusters[i],
                        mention_to_predicted=mention_to_predicted,
                        mention_to_gold=mention_to_gold,
                    )
            # print("")
            # print("Co-reference metrics - all predicted references")
            # print("")
            # self.coref_evaluator.print_metrics()
            print("")
            print("Co-reference metrics")
            print("")
            self.coref_evaluator_top.print_metrics()

        return ner_eval, rel_eval, rel_nec_eval

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            span_mask, rel_mask = (
                ("is_ner", "is_re") if self.is_ner_rel else ("is_emd", "is_cr")
            )
            gt_relations = doc.__getattribute__(self.relations_name)
            gt_entities = doc.__getattribute__(self.spans_name)
            entity_mask = doc.__getattribute__(span_mask)
            rel_mask = doc.__getattribute__(rel_mask)

            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_gt_entities = (
                [entity.as_tuple() for entity in gt_entities] if entity_mask else []
            )
            sample_gt_relations = (
                [rel.as_tuple() for rel in gt_relations] if rel_mask else []
            )

            if self._no_overlapping:
                sample_gt_entities, sample_gt_relations = prediction.remove_overlapping(
                    sample_gt_entities, sample_gt_relations
                )

            self._gt_entities.append(sample_gt_entities)
            self._gt_relations.append(sample_gt_relations)
            if not self.is_ner_rel:
                # NOTE: take only gt cluster with more than a single mention
                self._gt_clusters.append(
                    [c for c in doc.clusters if len(c) > 1] if doc.is_cr else []
                )

    def eval_batch(
        self,
        batch_entity_clf: torch.tensor,
        batch_rel_clf: torch.tensor,
        batch_rels: torch.tensor,
        batch: EvalBatch,
        #  evaluator for both NER+RE and EMD+CR
    ):
        batch_pred_entities, batch_pred_relations = prediction.convert_predictions(
            batch_entity_clf,
            batch_rel_clf,
            batch_rels,
            batch,
            self._rel_filter_threshold,
            self._dataset.data_parser,
            no_overlapping=self._no_overlapping,
            is_ner_rel=self.is_ner_rel,
        )

        # remove non-annotated docs
        span_mask, rel_mask = (
            ("is_ner", "is_re") if self.is_ner_rel else ("is_emd", "is_cr")
        )
        batch_pred_entities = [
            b if m.item() == 1 else []
            for b, m in zip(
                batch_pred_entities, batch.__getattribute__(span_mask).squeeze(1)
            )
        ]
        batch_pred_relations = [
            b if m.item() == 1 else []
            for b, m in zip(
                batch_pred_relations, batch.__getattribute__(rel_mask).squeeze(1)
            )
        ]

        self._pred_entities.extend(batch_pred_entities)
        self._pred_relations.extend(batch_pred_relations)

        if not self.is_ner_rel:  # eval co-reference
            # get predicted clusters
            for doc_pred in batch_pred_relations:
                doc_refs = []
                for ref in doc_pred:
                    doc_refs.append(
                        (
                            (
                                ref[0][0],
                                ref[0][1],
                            ),  # head span
                            (
                                ref[1][0],
                                ref[1][1],
                            ),  # tail span
                            ref[3],  # score
                        )
                    )

                doc_clusters = references_to_clusters(
                    references=doc_refs, filter_top=False
                )
                self._pred_clusters.append(doc_clusters)

                doc_clusters = references_to_clusters(
                    references=doc_refs, filter_top=True
                )
                self._pred_clusters_top.append(doc_clusters)

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

"""Source: https://github.com/lxucs/coref-hoi/blob/274167a0def8ccb94dfe024b4b04c73cf6e99904/metrics.py
"""


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class CorefEvaluator(object):
    def __init__(self):
        self.names = ["MUC", r"B^3", "CEAF (φ4)"]
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def reset(self):
        for e in self.evaluators:
            e.reset()

    def update(
        self,
        predicted: List[Tuple[Tuple[int, int], ...]],
        gold: List[Tuple[Tuple[int, int], ...]],
        mention_to_predicted: Dict[Tuple, Tuple],
        mention_to_gold: Dict[Tuple, Tuple],
    ):
        """

        Parameters
        ----------
        predicted : list of predicted cluster. Each cluster is a tuple and each mention is also a tuple of (start, end).
        gold : list of gold cluster. Each cluster is a tuple and each mention is also a tuple of (start, end).
        mention_to_predicted : dict of mention: predicted cluster
        mention_to_gold : dict of mention: gold cluster

        Returns
        -------

        """

        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    @staticmethod
    def _get_row(data, label):
        row = [label]
        for i in range(len(data)):
            row.append("%.2f" % (data[i] * 100))
        return tuple(row)

    def print_metrics(self):
        columns = ("metric", "precision", "recall", "f1-score")
        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = list()
        results.append(row_fmt % columns)
        results.append("\n")

        for name, evaluator in zip(self.names, self.evaluators):
            results.append(row_fmt % self._get_row(evaluator.get_prf(), name))
            results.append("\n")

        results_str = "".join(results)
        print(results_str)


class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def reset(self):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    # NOTE: make sure this is correct. Source:
    #  https://stackoverflow.com/questions/57369848/how-do-i-resolve-use-scipy-optimize-linear-sum-assignment-instead
    matching = np.array(list(zip(*matching)))
    # todo: Need to varify: this is to handle null clusters
    if len(clusters) == 0:
        if len(gold_clusters) == 0:
            return 1.0, 1, 1.0, 1  # todo: set num clusters to 0/1?
        else:
            return 0.0, 1, 0.0, len(gold_clusters)
    elif len(gold_clusters) == 0:
        if len(clusters) == 0:
            return 1.0, 1, 1.0, 1
        else:
            return 0.0, len(clusters), 0.0, 1
    else:
        similarity = sum(scores[matching[:, 0], matching[:, 1]])
        return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1 :]:
                    if (
                        m2 in mention_to_gold
                        and mention_to_gold[m] == mention_to_gold[m2]
                    ):
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem


if __name__ == "__main__":
    predicted = [((0, 1), (4, 5), (11, 15)), ((8, 9), (20, 22))]
    gold = [((0, 1), (11, 15)), ((8, 9), (20, 22))]
    mention_to_predicted = {m: cluster for cluster in predicted for m in cluster}
    mention_to_gold = {m: cluster for cluster in gold for m in cluster}
    evaluator = CorefEvaluator()
    evaluator.update(predicted, gold, mention_to_predicted, mention_to_gold)
    for name, eval in zip(evaluator.names, evaluator.evaluators):
        print(name)
        print(eval.get_prf())

    evaluator.print_metrics()

from collections import defaultdict
from typing import List, Tuple, Union

import networkx as nx

"""Utility functions for convert references/mentions to clusters."""


def references_to_clusters(
    references: List[Union[Tuple[Tuple, Tuple, float], Tuple[Tuple, Tuple]]],
    filter_top=False,
):
    # keep only top reference for each mention
    if filter_top and len(references) > 0:
        assert len(references[0]) == 3
        top_scores = defaultdict(float)
        top_references = {}
        for ref in references:
            if top_scores[ref[0]] < ref[2]:
                top_references[ref[0]] = ref
                top_scores[ref[0]] = ref[2]

        references = list(top_references.values())

    # Initialize graph (mentions are nodes and edges indicate coref linkage)
    graph = nx.Graph()
    for ref in references:
        graph.add_edge(ref[0], ref[1])

    # Extract clusters as nodes that share an edge
    clusters = [tuple(cluster) for cluster in list(nx.connected_components(graph))]
    return clusters


def convert_to_clusters(mentions, references, filter_top=False):
    if filter_top:
        top_scores = defaultdict(float)
        top_references = {}
        for ref in references:
            if top_scores[ref["head"]] < ref["score"]:
                top_references[ref["head"]] = ref
                top_scores[ref["head"]] = ref["score"]
        references = list(top_references.values())

    mention_span_to_mention = {}

    graph = nx.Graph()
    for ref in references:
        head_mention = mentions[ref["head"]]
        tail_mention = mentions[ref["tail"]]
        mention_span_to_mention[
            (head_mention["start"], head_mention["end"])
        ] = head_mention
        mention_span_to_mention[
            (tail_mention["start"], tail_mention["end"])
        ] = tail_mention
        graph.add_edge(
            (head_mention["start"], head_mention["end"]),
            (tail_mention["start"], tail_mention["end"]),
            score=ref["score"],
        )

    clusters = [tuple(cluster) for cluster in list(nx.connected_components(graph))]
    clusters = [
        tuple((m, mention_span_to_mention[m]["span"]) for m in cluster)
        for cluster in clusters
    ]
    return clusters


if __name__ == "__main__":
    mentions = [
        {
            "type": "MENTION",
            "start": 4,
            "end": 5,
            "span": ["Shai"],
            "score": 0.9973962306976318,
        },
        {
            "type": "MENTION",
            "start": 8,
            "end": 10,
            "span": ["Eval", "corp"],
            "score": 0.9940345883369446,
        },
        {
            "type": "MENTION",
            "start": 14,
            "end": 17,
            "span": ["our", "Adtec", "product"],
            "score": 0.999136745929718,
        },
        {
            "type": "MENTION",
            "start": 19,
            "end": 20,
            "span": ["he"],
            "score": 0.998333752155304,
        },
        {
            "type": "MENTION",
            "start": 23,
            "end": 24,
            "span": ["He"],
            "score": 0.9982483386993408,
        },
        {
            "type": "MENTION",
            "start": 27,
            "end": 29,
            "span": ["our", "pricing"],
            "score": 0.9427295327186584,
        },
        {
            "type": "MENTION",
            "start": 30,
            "end": 31,
            "span": ["it"],
            "score": 0.9961351156234741,
        },
        {
            "type": "MENTION",
            "start": 35,
            "end": 37,
            "span": ["a", "comparison"],
            "score": 0.6440563797950745,
        },
        {
            "type": "MENTION",
            "start": 38,
            "end": 40,
            "span": ["this", "product"],
            "score": 0.9981814622879028,
        },
    ]
    references = [
        {
            "type": "COREF",
            "head": 3,
            "tail": 0,
            "head_span": ["he"],
            "tail_span": ["Shai"],
            "score": 0.9832863807678223,
        },
        {
            "type": "COREF",
            "head": 4,
            "tail": 0,
            "head_span": ["He"],
            "tail_span": ["Shai"],
            "score": 0.9226845502853394,
        },
        {
            "type": "COREF",
            "head": 3,
            "tail": 2,
            "head_span": ["he"],
            "tail_span": ["our", "Adtec", "product"],
            "score": 0.6,
        },
        {
            "type": "COREF",
            "head": 6,
            "tail": 2,
            "head_span": ["it"],
            "tail_span": ["our", "Adtec", "product"],
            "score": 0.9750972986221313,
        },
        {
            "type": "COREF",
            "head": 8,
            "tail": 2,
            "head_span": ["this", "product"],
            "tail_span": ["our", "Adtec", "product"],
            "score": 0.8458112478256226,
        },
    ]

    print(convert_to_clusters(mentions, references, filter_top=True))
    print(convert_to_clusters(mentions, references, filter_top=False))

    references = [
        ((0, 1), (4, 5), 0.55),
        ((0, 1), (8, 11), 0.8),
        ((12, 14), (4, 5), 0.7),
        ((22, 23), (15, 16), 0.7),
        ((24, 25), (15, 16), 0.6),
        ((18, 19), (22, 23), 0.7),
    ]
    print(references_to_clusters(references=references, filter_top=True))
    print(references_to_clusters(references=references, filter_top=False))

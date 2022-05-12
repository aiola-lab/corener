import json
from collections import OrderedDict
from typing import List, Union

import spacy
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm

from corener.data.entities import (
    Document,
    Entity,
    EntityType,
    Mention,
    MentionType,
    Reference,
    ReferenceType,
    Relation,
    RelationType,
    Token,
)
from corener.data.sampling import create_eval_sample, create_train_sample
from corener.utils.clusters import references_to_clusters


class DataParser:
    """Parse json to extract examples"""

    def __init__(
        self,
        types: Union[str, dict],
        tokenizer: transformers.PreTrainedTokenizerBase,
        spacy_model="en_core_web_sm",
    ):
        """

        Parameters
        ----------
        types : path to json file containing all relations and entities types, or dict with types.
        tokenizer : transformers.PreTrainedTokenizerBase
        spacy_model :

        """
        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        self._mention_types = OrderedDict()
        self._idx2mention_type = OrderedDict()
        self._reference_types = OrderedDict()
        self._idx2reference_type = OrderedDict()

        # getting types of entities and relations
        if isinstance(types, str):
            with open(types, "r") as f:
                self.types = json.load(f, object_pairs_hook=OrderedDict)
                self._parse_types(self.types)
        else:
            self.types = OrderedDict(types)
            self._parse_types(self.types)

        self._tokenizer = tokenizer
        self._vocabulary_size = tokenizer.vocab_size

        self._nlp = spacy.load(spacy_model)

        self.documents = OrderedDict()
        self.entities = OrderedDict()
        self.relations = OrderedDict()
        self.mentions = OrderedDict()
        self.references = OrderedDict()

        self._token_ids = 0
        self._entity_ids = 0
        self._relation_ids = 0
        self._mention_ids = 0
        self._reference_ids = 0
        self._doc_ids = 0

    def reset(self):
        self.documents = OrderedDict()
        self.entities = OrderedDict()
        self.relations = OrderedDict()
        self.mentions = OrderedDict()
        self.references = OrderedDict()

        self._token_ids = 0
        self._entity_ids = 0
        self._relation_ids = 0
        self._mention_ids = 0
        self._reference_ids = 0
        self._doc_ids = 0

    def _parse_types(self, types):
        # entities
        # add 'None' entity type
        none_entity_type = EntityType("None", 0, "None", "No Entity")
        self._entity_types["None"] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # todo: consider removing. this is a hack...
        none_entity_type = EntityType("IGNORE", -100, "IGNORE", "Ignored Entity")
        self._entity_types["IGNORE"] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types["entities"].items()):
            entity_type = EntityType(key, i + 1, v["short"], v["verbose"])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # MENTIONS
        # add 'None' mention type
        none_mention_type = MentionType("None", 0, "None", "No Mention")
        self._mention_types["None"] = none_mention_type
        self._idx2mention_type[0] = none_mention_type

        # specified mention types
        for i, (key, v) in enumerate(types["mentions"].items()):
            mention_type = MentionType(key, i + 1, v["short"], v["verbose"])
            self._mention_types[key] = mention_type
            self._idx2mention_type[i + 1] = mention_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType("None", 0, "None", "No Relation")
        self._relation_types["None"] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types["relations"].items()):
            relation_type = RelationType(
                key, i + 1, v["short"], v["verbose"], v["symmetric"]
            )
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type

        # reference
        # add 'None' reference type
        none_reference_type = ReferenceType("None", 0, "None", "No Relation")
        self._relation_types["None"] = none_reference_type
        self._idx2reference_type[0] = none_reference_type

        # specified relation types
        for i, (key, v) in enumerate(types["references"].items()):
            reference_type = ReferenceType(
                key,
                i + 1,
                v["short"],
                v["verbose"],
            )
            self._reference_types[key] = reference_type
            self._idx2reference_type[i + 1] = reference_type

    def _create_token(self, index: int, span_start: int, span_end: int, phrase: str):
        token = Token(self._token_ids, index, span_start, span_end, phrase)
        self._token_ids += 1
        return token

    def _create_entity(self, entity_type, tokens, phrase) -> Entity:
        mention = Entity(self._entity_ids, entity_type, tokens, phrase)
        self.entities[self._entity_ids] = mention
        self._entity_ids += 1
        return mention

    def _create_mention(self, mention_type, tokens, phrase) -> Mention:
        mention = Mention(self._mention_ids, mention_type, tokens, phrase)
        self.mentions[self._mention_ids] = mention
        self._mention_ids += 1
        return mention

    def _create_relation(
        self, relation_type, head_entity, tail_entity, reverse=False
    ) -> Relation:
        relation = Relation(
            self._relation_ids, relation_type, head_entity, tail_entity, reverse
        )
        self.relations[self._relation_ids] = relation
        self._relation_ids += 1
        return relation

    def _create_reference(
        self, reference_type, head_mention, tail_mention
    ) -> Reference:
        reference = Reference(
            self._reference_ids, reference_type, head_mention, tail_mention
        )
        self.references[self._reference_ids] = reference
        self._reference_ids += 1
        return reference

    def _create_document(
        self,
        tokens,
        entity_mentions,
        relations,
        mentions,
        references,
        doc_encoding,
        is_ner,
        is_emd,
        is_re,
        is_cr,
        clusters=None,
    ) -> Document:
        document = Document(
            self._doc_ids,
            tokens,
            entity_mentions,
            relations,
            mentions,
            references,
            doc_encoding,
            is_ner=is_ner,
            is_emd=is_emd,
            is_re=is_re,
            is_cr=is_cr,
            clusters=clusters,
        )
        self.documents[self._doc_ids] = document
        self._doc_ids += 1

        return document

    def _parse_tokens(self, jtokens):
        doc_tokens = []

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [self._tokenizer.cls_token_id]

        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(
                token_phrase, add_special_tokens=False
            )
            if not token_encoding:
                token_encoding = [self._tokenizer.unk_token_id]
            span_start, span_end = (
                len(doc_encoding),
                len(doc_encoding) + len(token_encoding),
            )

            token = self._create_token(i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        doc_encoding += [self._tokenizer.sep_token_id]

        return doc_tokens, doc_encoding

    def _parse_spans(
        self, doc_spans, doc_tokens, is_ner=True
    ) -> List[Union[Entity, Mention]]:
        spans = []

        for entity_idx, jentity in enumerate(doc_spans):
            if is_ner:
                entity_type = self._entity_types[jentity["type"]]
            else:
                entity_type = self._mention_types[jentity["type"]]
            start, end = jentity["start"], jentity["end"]

            # create entity/mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            if is_ner:
                entity = self._create_entity(entity_type, tokens, phrase)
            else:
                entity = self._create_mention(entity_type, tokens, phrase)
            spans.append(entity)

        return spans

    def _parse_relations(
        self, doc_relations, entities, is_rel=True
    ) -> List[Union[Relation, Reference]]:
        relations = []

        for jrelation in doc_relations:
            if is_rel:
                relation_type = self._relation_types[jrelation["type"]]
            else:
                relation_type = self._reference_types[jrelation["type"]]

            head_idx = jrelation["head"]
            tail_idx = jrelation["tail"]

            # create relation
            head = entities[head_idx]
            tail = entities[tail_idx]

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                raise NotImplementedError
                # todo: handle symmetric relations
                # head, tail = utils.swap(head, tail)
            if is_rel:
                relation = self._create_relation(
                    relation_type, head_entity=head, tail_entity=tail, reverse=reverse
                )
            else:
                relation = self._create_reference(
                    relation_type,
                    head_mention=head,
                    tail_mention=tail,
                )
            relations.append(relation)

        return relations

    def _create_clusters(self, references):
        ref_spans = [(r.head_entity.span, r.tail_entity.span) for r in references]
        return references_to_clusters(references=ref_spans, filter_top=False)

    def _read_from_list(self, documents):
        if isinstance(documents, list):
            if isinstance(documents[0], list):
                # split into words
                for doc in documents:
                    doc_tokens, doc_encoding = self._parse_tokens(doc)
                    # todo: need refactoring so we won't need dummies
                    _ = self._create_document(
                        doc_tokens, [], [], [], [], doc_encoding, 1, 1, 1, 1, None
                    )
            else:
                # not split into words
                for doc in documents:
                    doc_tokens = [t.text for t in self._nlp(doc)]
                    doc_tokens, doc_encoding = self._parse_tokens(doc_tokens)
                    _ = self._create_document(
                        doc_tokens, [], [], [], [], doc_encoding, 1, 1, 1, 1, None
                    )

    def _read_from_file(self, path):
        with open(path, "r") as input_file:
            documents = json.load(input_file)

        if isinstance(documents[0]["tokens"], list):
            # split into words
            # parse docs
            for doc in tqdm(documents, desc="parsing documents"):
                doc_tokens = doc["tokens"]
                doc_relations = doc["relations"]
                doc_entities = doc["entities"]
                doc_mentions = doc["mentions"]
                doc_references = doc["references"]

                # parse tokens
                doc_tokens, doc_encoding = self._parse_tokens(doc_tokens)

                # parse entities
                entities = self._parse_spans(doc_entities, doc_tokens, is_ner=True)

                # parse relations
                relations = self._parse_relations(doc_relations, entities)

                # parse mentions
                mentions = self._parse_spans(doc_mentions, doc_tokens, is_ner=False)

                # parse reference
                references = self._parse_relations(
                    doc_references, mentions, is_rel=False
                )

                clusters = self._create_clusters(references)

                # create document
                _ = self._create_document(
                    doc_tokens,
                    entity_mentions=entities,
                    relations=relations,
                    mentions=mentions,
                    references=references,
                    doc_encoding=doc_encoding,
                    is_ner=doc["is_ner"],
                    is_emd=doc["is_emd"],
                    is_re=doc["is_re"],
                    is_cr=doc["is_cr"],
                    clusters=clusters,
                )
        else:
            # not split into words
            raise NotImplementedError

    def read(self, dataset_or_path: Union[List[str], List[List[str]], str]):
        """

        Parameters
        ----------
        dataset_or_path : List of examples, list of list of tokens, or path to json file

        Returns
        -------

        """
        # todo refactoring: we don't need all the self._x_ids and the self._entities, self._relations etc.
        #  We only need it in the Dataset object.

        if isinstance(dataset_or_path, list):
            self._read_from_list(dataset_or_path)
        else:
            self._read_from_file(dataset_or_path)

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def get_mention_type(self, idx) -> MentionType:
        entity = self._idx2mention_type[idx]
        return entity

    def get_reference_type(self, idx) -> RelationType:
        relation = self._idx2reference_type[idx]
        return relation

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        # todo: remove check if we get rid of ignored entity
        if "IGNORE" in self._entity_types:
            return len(self._entity_types) - 1
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size


class MTLDataset(Dataset):
    def __init__(
        self,
        types: Union[str, dict],
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset_or_path: Union[List[str], List[List[str]], str] = None,
        neg_entity_count: int = 100,
        neg_rel_count: int = 100,
        max_span_size: int = 10,
        train_mode=True,
        spacy_model="en_core_web_sm",
    ):
        """

        Parameters
        ----------
        types : path to json file containing all relations/references and entities/mentions types, or dict with types.
        tokenizer : transformers.PreTrainedTokenizerBase
        dataset_or_path : List of examples, list of list of tokens, or path to json file
        neg_entity_count : number of negative entities/mentions
        neg_rel_count : number of negative relations/references
        max_span_size : maximal span size for NER/EMD
        spacy_model :
        """

        self.data_parser = DataParser(
            types=types,
            tokenizer=tokenizer,
            spacy_model=spacy_model,
        )

        if dataset_or_path is not None:
            self.read_dataset(dataset_or_path)

        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size
        self.train_mode = train_mode
        self.eval_mode = not train_mode

        self._documents = self.data_parser.documents
        self._entities = self.data_parser.entities
        self._relations = self.data_parser.relations
        self._mentions = self.data_parser.mentions
        self._references = self.data_parser.references

    def _set_attrs(self):
        self._documents = self.data_parser.documents
        self._entities = self.data_parser.entities
        self._relations = self.data_parser.relations
        self._mentions = self.data_parser.mentions
        self._references = self.data_parser.references

    def reset(self):
        self.data_parser.reset()
        self._set_attrs()

    def read_dataset(self, dataset_or_path: Union[List[str], List[List[str]], str]):
        self.data_parser.read(dataset_or_path)
        self._set_attrs()

    def __len__(self):
        return len(self._documents)

    def get_example(self, index: int):
        return self.__getitem__(index=index)

    def __getitem__(self, index: int):
        doc = self._documents[index]

        if self.train_mode:
            return create_train_sample(
                doc,
                self._neg_entity_count,
                self._neg_rel_count,
                self._max_span_size,
                len(self.data_parser.relation_types),
            )

        else:
            return create_eval_sample(doc, self._max_span_size)

    def train(self):
        self.eval_mode = False
        self.train_mode = True

    def eval(self):
        self.eval_mode = True
        self.train_mode = False

    @property
    def input_reader(self):
        return self.data_parser

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def entities(self):
        return list(self._entities.values())

    @property
    def mentions(self):
        return list(self._mentions.values())

    @property
    def relations(self):
        return list(self._relations.values())

    @property
    def references(self):
        return list(self._references.values())

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def entity_count(self):
        return len(self._entities)

    @property
    def relation_count(self):
        return len(self._relations)

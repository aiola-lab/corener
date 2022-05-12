from dataclasses import dataclass
from typing import List, Tuple, Union

"""
Modification of: https://github.com/lavis-nlp/spert/blob/master/spert/entities.py
"""


@dataclass
class RelationType:
    identifier: Union[int, str]
    index: int
    short_name: str
    verbose_name: str
    symmetric: bool = False

    def __int__(self):
        return self.index

    def __eq__(self, other):
        if isinstance(other, RelationType):
            return self.identifier == other.identifier
        return False

    def __hash__(self):
        return hash(self.identifier)


@dataclass
class EntityType:
    identifier: Union[int, str]
    index: int
    short_name: str
    verbose_name: str

    def __int__(self):
        return self.index

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self.identifier == other.identifier
        return False

    def __hash__(self):
        return hash(self.identifier)


class MentionType(EntityType):
    def __init__(
        self,
        identifier: Union[int, str],
        index: int,
        short_name="MENTION",
        verbose_name="MENTION",
    ):
        super().__init__(
            identifier=identifier,
            index=index,
            short_name=short_name,
            verbose_name=verbose_name,
        )


class ReferenceType(RelationType):
    def __init__(
        self,
        identifier: Union[int, str],
        index: int,
        short_name="COREF",
        verbose_name="COREF",
    ):
        super().__init__(
            identifier=identifier,
            index=index,
            short_name=short_name,
            verbose_name=verbose_name,
            symmetric=False,
        )


@dataclass
class Token:
    tid: int  # ID within the corresponding dataset
    id: int  # original token index in document
    span_start: int  # start of token span in document (inclusive)
    span_end: int  # end of token span in document (exclusive)
    phrase: str

    def __post_init__(self):
        self.span = (self.span_start, self.span_end)

    @property
    def index(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.tid == other.tid
        return False

    def __hash__(self):
        return hash(self.tid)

    def __str__(self):
        return self.phrase

    def __repr__(self):
        return self.phrase


@dataclass
class TokenSpan:
    tokens: List[Token]

    @property
    def span_start(self):
        return self.tokens[0].span_start

    @property
    def span_end(self):
        return self.tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self.tokens[s.start : s.stop : s.step])
        else:
            return self.tokens[s]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)


@dataclass
class Entity:
    eid: int
    entity_type: EntityType
    tokens: Union[List[Token], TokenSpan]
    phrase: str

    def __post_init__(self):
        if not isinstance(self.tokens, TokenSpan):
            self.tokens = TokenSpan(self.tokens)

    def as_tuple(self):
        return self.span_start, self.span_end, self.entity_type

    @property
    def span_start(self):
        return self.tokens.span_start

    @property
    def span_end(self):
        return self.tokens.span_end

    @property
    def span(self):
        return self.tokens.span

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.eid == other.eid
        return False

    def __hash__(self):
        return hash(self.eid)

    def __str__(self):
        return self.phrase


class Mention(Entity):
    def __init__(
        self,
        eid: int,
        entity_type: MentionType,
        tokens: Union[List[Token], TokenSpan],
        phrase: str,
    ):
        super().__init__(eid=eid, entity_type=entity_type, tokens=tokens, phrase=phrase)


@dataclass
class Relation:
    rid: int
    relation_type: RelationType
    head_entity: Entity
    tail_entity: Entity
    reverse: bool = False

    def __post_init__(self):
        self.first_entity = self.head_entity if not self.reverse else self.tail_entity
        self.second_entity = self.tail_entity if not self.reverse else self.head_entity

    def as_tuple(self):
        head = self.head_entity
        tail = self.tail_entity
        head_start, head_end = (head.span_start, head.span_end)
        tail_start, tail_end = (tail.span_start, tail.span_end)

        t = (
            (head_start, head_end, head.entity_type),
            (tail_start, tail_end, tail.entity_type),
            self.relation_type,
        )
        return t

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self.rid == other.rid
        return False

    def __hash__(self):
        return hash(self.rid)


class Reference(Relation):
    def __init__(
        self,
        rid: int,
        relation_type: RelationType,
        head_entity: Mention,
        tail_entity: Mention,
    ):
        super().__init__(
            rid=rid,
            relation_type=relation_type,
            head_entity=head_entity,
            tail_entity=tail_entity,
            reverse=False,
        )


@dataclass
class Document:
    doc_id: int  # ID within the corresponding dataset
    tokens: Union[List[Token], TokenSpan]
    entities: List[Entity]
    relations: List[Relation]
    mentions: List[Mention]
    references: List[Reference]
    encoding: List[
        int
    ]  # byte-pair document encoding including special tokens ([CLS] and [SEP])
    is_ner: bool
    is_emd: bool
    is_re: bool
    is_cr: bool
    clusters: List[Tuple[Tuple]] = None

    def __post_init__(self):
        if not isinstance(self.tokens, TokenSpan):
            self.tokens = TokenSpan(self.tokens)

        if self.clusters is None:
            self.clusters = []

    def __eq__(self, other):
        if isinstance(other, Document):
            return self.doc_id == other.doc_id
        return False

    def __hash__(self):
        return hash(self.doc_id)

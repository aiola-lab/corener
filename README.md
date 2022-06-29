CoReNer
==============================

[![Python Version](https://img.shields.io/badge/python-3.7+-blue?logo=python)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A multi-task model for named-entity recognition, relation extraction, entity mention detection and coreference resolution.
Our model extends the [SpERT](https://github.com/lavis-nlp/spert) framework to: (i) add two additional tasks, namely entity mention detection (EMD) and coreference resolution (CR), and (ii) support different pretrained backbones from the Huggingface model hub (e.g. `roberta-base`).

We model NER as a span classification task, and relation extraction as a multi-label classification of (NER) span tuples. 
Similarly, model EMD as a span classification task and CR as a binary classification of (EMD) span tuples.
To construct the CR clusters, we keep the top antecedent of each mention, then compute the connected components of the mentions' undirected graph.

## Model checkpoints

We release RoBERTa-based CoReNer models, finetuned on the 4 tasks (NER, RE, EMD and CR) using the Ontonotes and CoNLL04 datasets.
The model checkpoint are available at Huggingface's model hub: 
- [aiola/roberta-base-corener](https://huggingface.co/aiola/roberta-base-corener).
- [aiola/roberta-large-corener](https://huggingface.co/aiola/roberta-large-corener).


## Installation

```bash
git clone https://github.com/aiola-lab/corener.git
cd corener
pip install --upgrade pip
pip install -e .
# also install spacy en model
python -m spacy download en_core_web_sm
```

## Usage

```python
import json

from transformers import AutoTokenizer
from corener.models import Corener, ModelOutput
from corener.data import MTLDataset
from corener.utils.prediction import convert_model_output


tokenizer = AutoTokenizer.from_pretrained("aiola/roberta-base-corener")
model = Corener.from_pretrained("aiola/roberta-base-corener")
model.eval()

examples = [
    "In 2009, ABC increased its margin by 10%. The company used to manufacture its car in Thailand but moved the factories to China."
]

dataset = MTLDataset(
    types=model.config.types, 
    tokenizer=tokenizer,
    train_mode=False,
)
dataset.read_dataset(examples)
example = dataset.get_example(0)  # get first example

output: ModelOutput = model(
    input_ids=example.encodings,
    context_masks=example.context_masks,
    entity_masks=example.entity_masks,
    entity_sizes=example.entity_sizes,
    entity_spans=example.entity_spans,
    entity_sample_masks=example.entity_sample_masks,
    inference=True,
)

print(json.dumps(convert_model_output(output=output, batch=example, dataset=dataset), indent=2))
```


## Training

Training CLI example:

```bash
python train.py --train-path path/to/train.json \
  --val-path path/to/val.json \
  --types-path path/to/types.json \
  --model-name-or-path roberta-base \
  --artifact-path path/to/artifacts \
  --do-eval
```

## Inference

Inference example and output.

```bash
python inference.py 
  --artifact-path path/to/artifacts \ 
  --input "In 2009, ABC increased its margin by 10%. The company used to manufacture its car in Thailand but moved the factories to China."
```

Output:

```json5
[
  {
    "tokens": [
      "In",
      "2009",
      ",",
      ...
    ],
    "entities": [
      {
        "type": "DATE",
        "start": 1,
        "end": 2,
        "span": [
          "2009"
        ],
        "score": 0.9997476935386658
      },
      ...
    ],
    "relations": [
      {
        "type": "OrgBased_In",
        "head": 1,
        "tail": 4,
        "head_span": [
          "ABC"
        ],
        "tail_span": [
          "China"
        ],
        "score": 0.9945483803749084
      }
    ],
    "mentions": [
      {
        "type": "MENTION",
        "start": 3,
        "end": 4,
        "span": [
          "ABC"
        ],
        "score": 0.9999425411224365
      },
      ...
    ],
    "references": [
      {
        "type": "COREF",
        "head": 1,
        "tail": 0,
        "head_span": [
          "its"
        ],
        "tail_span": [
          "ABC"
        ],
        "score": 1.0
      },
      ...
    ],
    "clusters": [
      [
        {
          "start": 11,
          "end": 13,
          "span": [
            "The",
            "company"
          ],
          "cluster_id": 0
        },
        {
          "start": 16,
          "end": 17,
          "span": [
            "its"
          ],
          "cluster_id": 0
        },
        ...
      ]
    ]
  }
]
```

## Data

Training data is a `json` file of the following form:

```json5
[
  {
    "tokens": ["John", "met", "Jane", ".", "He", "asked", "her", "a", "question", "."],
    "entities": [
      {"type": "PERSON", "start": 0, "end": 1}, // John
      {"type": "PERSON", "start": 2, "end": 3}  // Jane
    ],
    "relations": [
      {"type": "MET", "head": 0, "tail": 1}  // "head"/"tail" is the index of the head/tail entities.
    ],
    "mentions": [
      {"type": "MENTION", "start": 0, "end": 1}, // John
      {"type": "MENTION", "start": 2, "end": 3}, // Jane
      {"type": "MENTION", "start": 4, "end": 5}, // he
      {"type": "MENTION", "start": 6, "end": 7}  // her
    ],
    "references": [
      {"type": "COREF", "head": 2, "tail": 0}, // He -> John
      {"type": "COREF", "head": 3, "tail": 1} // her -> Jane
    ],
    "is_ner": 1, // boolean for whether the doc is labeled for the NER task
    "is_emd": 1, // boolean for whether the doc is labeled for the EMD task
    "is_re": 1, // boolean for whether the doc is labeled for the relation extraction task
    "is_cr": 1, // boolean for whether the doc is labeled for the co-reference task
  },
  {
    // second document.
  }
]
```

In addition, you will need to provide a `types.json` file will all entity/relation types presented in the training data. For example, to train CoReNer on the Ontonotes + Conll04 datasets we use the following file:

```json5
{
  "entities": {
    "ORG": {"short": "ORG", "verbose": "ORGANIZATION"},
    "PERSON": {"short": "PER", "verbose":"PERSON"},
    "NORP": {"short": "NORP", "verbose":"Nationalities or religious or political groups"},
    "FAC": {"short": "FAC", "verbose":"Buildings, airports, highways, bridges"},
    "GPE": {"short": "GPE", "verbose":"Countries, cities, states."},
    "LOC": {"short": "LOC", "verbose":"LOCATION"},
    "PRODUCT": {"short": "PROD", "verbose": "PRODUCT"},
    "DATE": {"short": "DATE", "verbose": "DATE"},
    "TIME": {"short": "TIME", "verbose": "TIME"},
    "PERCENT": {"short": "PERCENT", "verbose": "PERCENT"},
    "MONEY": {"short": "MONEY", "verbose": "MONEY"},
    "QUANTITY": {"short": "QUANTITY", "verbose": "QUANTITY"},
    "ORDINAL": {"short": "ORDINAL", "verbose": "ORDINAL"},
    "CARDINAL": {"short": "CARDINAL", "verbose": "CARDINAL"},
    "EVENT": {"short": "EVENT", "verbose": "EVENT"},
    "WORK_OF_ART": {"short": "WORK_OF_ART", "verbose": "WORK_OF_ART"},
    "LAW": {"short": "LAW", "verbose": "LAW"},
    "LANGUAGE": {"short": "LANGUAGE", "verbose": "LANGUAGE"}
  },
  "relations": {
    "Work_For": {"short": "Work", "verbose": "Work for", "symmetric": false},
    "Kill": {"short": "Kill", "verbose": "Kill", "symmetric": false},
    "OrgBased_In": {"short": "OrgBI", "verbose": "Organization based in", "symmetric": false},
    "Live_In": {"short": "Live", "verbose": "Live in", "symmetric": false},
    "Located_In": {"short": "LocIn", "verbose": "Located in", "symmetric": false}
  },
  "references": {
    "COREF": {"short": "COREF", "verbose": "COREF"}
  },
  "mentions": {
    "MENTION": {"short": "MENTION", "verbose": "MENTION"}
  }
}
```
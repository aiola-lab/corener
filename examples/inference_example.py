import json

from transformers import AutoTokenizer

from corener.data import MTLDataset
from corener.models import Corener, ModelOutput
from corener.utils.prediction import convert_model_output

tokenizer = AutoTokenizer.from_pretrained(
    "../artifacts/roberta-base-corener-ontonotes-60e"
)  # aiola/roberta-base-corener
model = Corener.from_pretrained("../artifacts/roberta-base-corener-ontonotes-60e")
model.eval()

examples = [
    "Steve Jobs was a charismatic pioneer of the personal computer era. With Steve Wozniak, Jobs founded Apple Inc. "
    "in 1976 and transformed the company into a world leader in telecommunications. Widely considered a visionary and a"
    " genius, he oversaw the launch of such revolutionary products as the iPod and the iPhone.",
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

print(
    json.dumps(
        convert_model_output(output=output, batch=example, dataset=dataset), indent=2
    )
)

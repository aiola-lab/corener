import json

from transformers import AutoTokenizer

from corener.data import MTLDataset
from corener.models import Corener, ModelOutput
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

print(
    json.dumps(
        convert_model_output(output=output, batch=example, dataset=dataset), indent=2
    )
)

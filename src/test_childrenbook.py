
#%%
from transformers import pipeline 
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import pandas as pd
from transformers import DefaultDataCollator
import json
import torch
import evaluate
from tqdm.auto import tqdm
import collections
import numpy as np

import torch
from torchmetrics.functional.classification import multiclass_f1_score
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", return_tesnors='pt', truncation_side='left')
model = AutoModelForMultipleChoice.from_pretrained("distilbert-base-uncased")

#%%
from datasets import load_dataset

dataset = load_dataset("cbt", 'CN', split='train[:10000]')
dataset = dataset.train_test_split(test_size=0.2)


#%%
def preprocess_function(examples) :
    # length_of_options = [len(option) for option in examples["options"]]
    # first trunctate the sentences and options
    contexts = [["".join(sentence)]*10 for sentence in examples["sentences"]] # list of sentences
    questions = [question for question in examples['question']]

    options = [option for option in examples['options']] # list of list
    # the number of choices is 10
    questions = [[questions[idx].replace("XXXXX", option) for option in options] for idx, options in enumerate(options)]
    # labels =  [0]*10

    answers = examples['answer']
    labels = [options[idx].index(answer) for idx, answer in enumerate(answers)]

    # now we flatten this
    contexts = sum(contexts, [])
    questions = sum(questions, [])

    tokenized_examples = tokenizer(contexts, questions, truncation="only_first", max_length=256)

    tokenized_examples = {k: [v[i : i + 10] for i in range(0, len(v), 10)] for k, v in tokenized_examples.items()}
    tokenized_examples['label'] = labels
    return tokenized_examples

#%%
tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=4, remove_columns=dataset['train'].column_names)

#%%

import evaluate
accuracy = evaluate.load("accuracy")
metrics = evaluate.combine(["accuracy", "f1"])

def compute_metrics(eval_pred) :
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metrics.compute(predictions=predictions, references=labels)


#%%
# I don't need this because I already did the padding

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch



#%%
# Define training arguments
training_args = TrainingArguments(
    output_dir="../model/cbt_distilbert",
    overwrite_output_dir=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,
    weight_decay=0.01,
    push_to_hub=False,
)

# Define trainer:
trainer = Trainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["test"],
    data_collator=DataCollatorForMultipleChoice(tokenizer),
)

#%%
# Start training
trainer.train()


#%% This is the evaluation code
checkpoint = "../model/cbt_distilbert/checkpoint-25000"
model = AutoModelForMultipleChoice.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# #%%
# data_collator = DefaultDataCollator()
training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer)
)

#%%
sample = tokenized_dataset['test']

#%%
predictions = trainer.predict(sample)
pred = predictions.predictions.argmax(-1)
pred = pred.tolist()
y = sample['label']

#%%
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

acc = accuracy_metric.compute(predictions=pred, references=y)
print(f"accuracy is {acc['accuracy']}")

#%%
pred = torch.Tensor(pred)
y = torch.Tensor(y)
f1_score = multiclass_f1_score(pred, y, num_classes=10, average="macro")
print(f"the f1 score is {f1_score.item()}")


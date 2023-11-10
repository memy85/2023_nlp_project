
#%%
from transformers import pipeline
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import pandas as pd
from metrics import compute_metrics
from preprocess import preprocess_trivia

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt", return_tesnors='pt')
model = OpenAIGPTModel.from_pretrained("openai-gpt")

#%%
model.config

#%%
from datasets import load_dataset

dataset = load_dataset("trivia_qa", "unfiltered")

#%%
dataset['train'].to_pandas()

#%%
dataset['train']['search_results'][0]

#%%
dataset['train']['answer'][0]

#%%
a = tokenizer(dataset['train']['question'][1])


#%%
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained("openai-gpt")


#%%

def tokenize(batch) :
    return tokenizer(batch['question'], padding=True, truncation=True)

dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)

#%%

from transformers import Trainer, TrainingArguments, AutoModelForQuestionAnswering

batch_size = 16
logging_steps = len(dataset_encoded["train"])
model_name = f"openai-gpt-finetuned-triviaqa"
training_args = TrainingArguments(output_dir= model_name,
                                  evaluation_strategy="epoch",
                                  num_train_epochs = 2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps = logging_steps,
                                  push_to_hub=False, # if you want to push to the hub
                                  log_level="error")


#%% We now use the Trainer

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=dataset_encoded["train"],
                  eval_dataset=dataset_encoded["validation"],
                  tokenizer=tokenizer,
                  data_collator=)

trainer.train()

#%% After the training, you can output

preds_outputs = trainer.predict(emotions_encoded["validation"])
preds_outputs.metrics

